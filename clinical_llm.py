"""Utilities for turning free-text clinical notes into structured data."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List


SYMPTOM_KEYS = [
    "dolor_toracico",
    "dificultad_respiratoria",
    "deficit_neurologico",
    "abdomen_agudo",
    "sangrado_activo",
    "traumatismo_grave",
    "convulsiones",
    "intoxicacion_sobredosis",
    "anafilaxia_alergia",
    "emergencia_obstetrica",
    "quemadura_extensa",
    "riesgo_psiquiatrico",
    "vomitos_diarrea_severa",
]

SPECIALTY_TO_CODE = {
    "Desconocida": 0,
    "Cardiología": 1,
    "Neurología": 2,
    "Traumatología": 3,
    "Psiquiatría": 4,
    "Obstetricia": 5,
    "Medicina Interna": 6,
}

FEATURE_ORDER = [
    "edad",
    "sexo",
    "frecuencia_cardiaca",
    "presion_sistolica",
    "presion_diastolica",
    "saturacion_oxigeno",
    "temperatura",
    "frecuencia_respiratoria",
    "glucemia",
    *SYMPTOM_KEYS,
    "inicio_subito",
    "antecedentes_graves",
    "specialty_code",
]

DEFAULT_MODEL = os.getenv("AMBULANCIA_LLM_MODEL", "llama3.1:8b-instruct")
DEFAULT_PROVIDER = os.getenv("AMBULANCIA_LLM_PROVIDER", "ollama")
DEFAULT_BASE_URL = os.getenv("AMBULANCIA_LLM_BASE_URL", "http://localhost:11434")


def build_prompt(diagnosis_text: str) -> str:
    """Build the instruction prompt for the model."""

    schema = {
        "urgency_level": "critica|alta|media|baja|indeterminada",
        "summary": "string breve",
        "patient": {"age": None, "sex": "M|F|unknown"},
        "vitals": {
            "frecuencia_cardiaca": None,
            "presion_sistolica": None,
            "presion_diastolica": None,
            "saturacion_oxigeno": None,
            "temperatura": None,
            "frecuencia_respiratoria": None,
            "glucemia": None,
        },
        "symptoms": {key: 0 for key in SYMPTOM_KEYS},
        "specialty_suspected": "Cardiología|Neurología|Traumatología|Psiquiatría|Obstetricia|Medicina Interna|Desconocida",
        "risk_flags": ["red flags"],
        "confidence": 0.0,
    }

    return (
        "Eres un extractor clínico muy estricto. Convierte la nota del paciente en JSON válido y devuelve solo el JSON, sin markdown ni texto extra.\n"
        "Reglas: 1) no inventes datos; 2) si un dato aparece en la nota, extráelo; 3) si un dato no aparece, usa null en campos numéricos y 'unknown' en sexo; 4) usa la urgencia más alta compatible con la nota; 5) ST elevado, dolor torácico opresivo, disnea y diaforesis deben marcarse como alta o crítica.\n"
        "Marca cada síntoma con 1 si está presente o se infiere con alta probabilidad, si no 0.\n"
        "Usa este esquema exacto como referencia y no añadas claves nuevas:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Nota clínica:\n"
        f"{diagnosis_text.strip()}"
    )


def analyze_clinical_diagnosis(
    diagnosis_text: str,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    base_url: str = DEFAULT_BASE_URL,
) -> Dict[str, Any]:
    """Convert free text into structured JSON and a numeric feature vector."""

    cleaned_text = diagnosis_text.strip()
    if not cleaned_text:
        raise ValueError("La nota clínica no puede estar vacía.")

    raw_output = ""
    used_fallback = False

    if provider == "ollama":
        try:
            raw_output = _call_ollama(cleaned_text, model=model, base_url=base_url)
        except Exception:
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback:
        structured = _heuristic_extract(cleaned_text)
        raw_output = json.dumps(structured, ensure_ascii=False)
    else:
        structured = _normalize_structured_output(raw_output, cleaned_text)

    extracted_baseline = _extract_clinical_signals(cleaned_text)
    structured = _merge_structured_outputs(extracted_baseline, structured)
    structured = _normalize_structured_output(json.dumps(structured, ensure_ascii=False), cleaned_text)

    feature_map = _build_feature_map(structured)
    feature_vector = [feature_map[key] for key in FEATURE_ORDER]

    return {
        "provider": provider if not used_fallback else "fallback",
        "model": model,
        "fallback_used": used_fallback,
        "source_text": cleaned_text,
        "raw_model_output": raw_output,
        "structured_output": structured,
        "feature_map": feature_map,
        "feature_order": FEATURE_ORDER,
        "feature_vector": feature_vector,
    }


def _call_ollama(diagnosis_text: str, model: str, base_url: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": "Devuelve exclusivamente JSON válido."},
            {"role": "user", "content": build_prompt(diagnosis_text)},
        ],
    }

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    content = response_payload.get("message", {}).get("content", "")
    if not content:
        raise ValueError("El modelo no devolvió contenido.")
    return content


def _normalize_structured_output(raw_output: str, diagnosis_text: str) -> Dict[str, Any]:
    parsed = _extract_json_object(raw_output)
    if parsed is None:
        parsed = _heuristic_extract(diagnosis_text)

    parsed.setdefault("urgency_level", "indeterminada")
    parsed.setdefault("summary", diagnosis_text[:240])
    parsed.setdefault("patient", {})
    parsed.setdefault("vitals", {})
    parsed.setdefault("symptoms", {})
    parsed.setdefault("specialty_suspected", "Desconocida")
    parsed.setdefault("risk_flags", [])
    parsed.setdefault("confidence", 0.5)

    patient = parsed["patient"]
    patient["age"] = _to_int_or_none(patient.get("age"))
    patient["sex"] = _normalize_sex(patient.get("sex"))

    vitals = parsed["vitals"]
    for key in [
        "frecuencia_cardiaca",
        "presion_sistolica",
        "presion_diastolica",
        "saturacion_oxigeno",
        "temperatura",
        "frecuencia_respiratoria",
        "glucemia",
    ]:
        vitals[key] = _to_float_or_none(vitals.get(key))

    symptoms = parsed["symptoms"]
    normalized_symptoms = {key: 1 if int(bool(symptoms.get(key, 0))) else 0 for key in SYMPTOM_KEYS}

    return {
        "urgency_level": str(parsed.get("urgency_level", "indeterminada")),
        "summary": str(parsed.get("summary", "")).strip(),
        "patient": patient,
        "vitals": vitals,
        "symptoms": normalized_symptoms,
        "specialty_suspected": _normalize_specialty(parsed.get("specialty_suspected")),
        "risk_flags": _normalize_risk_flags(parsed.get("risk_flags", [])),
        "confidence": _clamp_confidence(parsed.get("confidence", 0.5)),
    }


def _merge_structured_outputs(base: Dict[str, Any], preferred: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(preferred)
    merged.setdefault("patient", {})
    merged.setdefault("vitals", {})
    merged.setdefault("symptoms", {})

    for key, value in base.get("patient", {}).items():
        if merged["patient"].get(key) in {None, "", "unknown"} and value not in {None, "", "unknown"}:
            merged["patient"][key] = value

    for key, value in base.get("vitals", {}).items():
        if merged["vitals"].get(key) in {None, "", 0} and value not in {None, "", 0}:
            merged["vitals"][key] = value

    for key, value in base.get("symptoms", {}).items():
        if int(value) == 1:
            merged["symptoms"][key] = 1
        else:
            merged["symptoms"].setdefault(key, 0)

    if merged.get("specialty_suspected", "Desconocida") == "Desconocida" and base.get("specialty_suspected"):
        merged["specialty_suspected"] = base["specialty_suspected"]

    if merged.get("urgency_level", "indeterminada") in {"indeterminada", "baja"} and base.get("urgency_level"):
        merged["urgency_level"] = base["urgency_level"]

    if not merged.get("risk_flags") and base.get("risk_flags"):
        merged["risk_flags"] = base["risk_flags"]

    if merged.get("summary", "").strip() == "":
        merged["summary"] = base.get("summary", "")

    return merged


def _heuristic_extract(diagnosis_text: str) -> Dict[str, Any]:
    baseline = _extract_clinical_signals(diagnosis_text)
    symptoms = baseline["symptoms"]

    urgency_level = baseline["urgency_level"]
    specialty = _guess_specialty(symptoms)
    risk_flags = baseline["risk_flags"]

    return {
        "urgency_level": urgency_level,
        "summary": baseline["summary"],
        "patient": baseline["patient"],
        "vitals": baseline["vitals"],
        "symptoms": symptoms,
        "specialty_suspected": specialty,
        "risk_flags": risk_flags,
        "confidence": baseline["confidence"],
    }


def _extract_clinical_signals(diagnosis_text: str) -> Dict[str, Any]:
    text = _normalize_text(diagnosis_text)
    symptoms = {key: 0 for key in SYMPTOM_KEYS}

    symptom_rules = {
        "dolor_toracico": [
            r"dolor(?:\s+fuerte|\s+intenso|\s+opresivo)?\s+(?:en\s+el\s+)?pecho",
            r"dolor\s+toracico",
            r"opresiv[oa]",
            r"infarto",
            r"st\s+elevado",
            r"elevacion\s+del\s+segmento\s+st",
        ],
        "dificultad_respiratoria": [r"disnea", r"dificultad\s+para\s+respirar", r"falta\s+de\s+aire"],
        "deficit_neurologico": [r"debilidad\s+de\s*un\s*lado", r"afasia", r"deterioro\s+neurologico", r"deficit\s+neurologico"],
        "abdomen_agudo": [r"abdomen\s+agudo", r"dolor\s+abdominal\s+intenso"],
        "sangrado_activo": [r"sangrado", r"hemorragia"],
        "traumatismo_grave": [r"traumatism[oa]", r"fractura", r"politrauma", r"accidente"],
        "convulsiones": [r"convulsion", r"crisis\s+convulsiv"],
        "intoxicacion_sobredosis": [r"sobredosis", r"intoxicacion"],
        "anafilaxia_alergia": [r"anafilaxia", r"reaccion\s+alergica"],
        "emergencia_obstetrica": [r"embaraz", r"parto", r"hemorragia\s+vaginal"],
        "quemadura_extensa": [r"quemadura"],
        "riesgo_psiquiatrico": [r"riesgo\s+psiquiatrico", r"suicid"],
        "vomitos_diarrea_severa": [r"vomit", r"diarrea"],
    }

    for symptom_key, patterns in symptom_rules.items():
        if any(re.search(pattern, text) for pattern in patterns):
            symptoms[symptom_key] = 1

    age = _extract_age(text)
    sex = _extract_sex(text)
    vitals = {
        "frecuencia_cardiaca": _extract_vital_value(text, [r"frecuencia\s+cardiaca", r"fc"], unit_patterns=[r"lpm", r"latidos?\s+por\s+minuto"]),
        "presion_sistolica": _extract_bp(text, systolic=True),
        "presion_diastolica": _extract_bp(text, systolic=False),
        "saturacion_oxigeno": _extract_vital_value(text, [r"saturacion\s+de\s+oxigeno", r"saturacion", r"spo2", r"sa[oó]2"], unit_patterns=[r"%"]),
        "temperatura": _extract_vital_value(text, [r"temperatura"], unit_patterns=[r"°?c"]),
        "frecuencia_respiratoria": _extract_vital_value(text, [r"frecuencia\s+respiratoria", r"fr"], unit_patterns=[r"rpm", r"respiraciones\s+por\s+minuto"]),
        "glucemia": _extract_vital_value(text, [r"glucemia", r"glucosa"], unit_patterns=[r"mg/dl"]),
    }

    urgency_score = 0
    if symptoms["dolor_toracico"]:
        urgency_score += 2
    if symptoms["dificultad_respiratoria"]:
        urgency_score += 2
    if symptoms["sangrado_activo"] or symptoms["traumatismo_grave"] or symptoms["convulsiones"] or symptoms["emergencia_obstetrica"]:
        urgency_score += 3
    if symptoms["intoxicacion_sobredosis"] or symptoms["anafilaxia_alergia"]:
        urgency_score += 2
    if symptoms["quemadura_extensa"] or symptoms["deficit_neurologico"]:
        urgency_score += 2
    if _to_float_or_none(vitals["saturacion_oxigeno"]) is not None and _to_float_or_none(vitals["saturacion_oxigeno"]) < 94:
        urgency_score += 1
    if _to_float_or_none(vitals["frecuencia_cardiaca"]) is not None and _to_float_or_none(vitals["frecuencia_cardiaca"]) > 100:
        urgency_score += 1
    if _to_float_or_none(vitals["frecuencia_respiratoria"]) is not None and _to_float_or_none(vitals["frecuencia_respiratoria"]) > 20:
        urgency_score += 1
    if re.search(r"st\s+elevado|elevacion\s+del\s+segmento\s+st", text):
        urgency_score += 3

    if urgency_score >= 6:
        urgency_level = "critica"
    elif urgency_score >= 3:
        urgency_level = "alta"
    elif any(symptoms.values()):
        urgency_level = "media"
    else:
        urgency_level = "baja"

    risk_flags = []
    if symptoms["dolor_toracico"]:
        risk_flags.append("dolor_toracico")
    if symptoms["dificultad_respiratoria"]:
        risk_flags.append("dificultad_respiratoria")
    if re.search(r"st\s+elevado|elevacion\s+del\s+segmento\s+st", text):
        risk_flags.append("st_elevado")

    if age is not None:
        patient_age = age
    else:
        patient_age = None

    if sex is None:
        sex_value = "unknown"
    else:
        sex_value = sex

    specialty = _guess_specialty(symptoms)
    if symptoms.get("dolor_toracico") or re.search(r"st\s+elevado|elevacion\s+del\s+segmento\s+st", text):
        specialty = "Cardiología"

    summary = _build_summary_text(diagnosis_text, symptoms, vitals, urgency_level)

    return {
        "urgency_level": urgency_level,
        "summary": summary,
        "patient": {"age": patient_age, "sex": sex_value},
        "vitals": vitals,
        "symptoms": symptoms,
        "specialty_suspected": specialty,
        "risk_flags": risk_flags,
        "confidence": 0.9 if urgency_level in {"critica", "alta"} else 0.7,
    }


def _normalize_text(text: str) -> str:
    text = text.lower()
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_age(text: str) -> int | None:
    patterns = [
        r"(?:paciente|varon|mujer|hombre|mujer)?\s*(?:de|unos|de unos)\s*(\d{1,3})\s*anos",
        r"edad\s*(?:de)?\s*(\d{1,3})\s*anos",
        r"(\d{1,3})\s*anos",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            age = int(match.group(1))
            if 0 < age < 120:
                return age
    return None


def _extract_sex(text: str) -> str | None:
    if re.search(r"\bvaron\b|\bhombre\b|\bmasculino\b", text):
        return "M"
    if re.search(r"\bmujer\b|\bfemenino\b", text):
        return "F"
    return None


def _extract_vital_value(text: str, label_patterns: List[str], unit_patterns: List[str] | None = None) -> float | None:
    label_regex = r"(?:" + r"|".join(label_patterns) + r")"
    unit_regex = r"(?:\s*(?:" + r"|".join(unit_patterns) + r")\b)?" if unit_patterns else r""
    connector_regex = r"(?:\s+(?:de|del|la|el|al))?"

    patterns = [
        rf"{label_regex}{connector_regex}\s*(?:es\s+)?(?:aproximadamente\s+)?(?:[=:]?\s*)?(\d{{1,3}}(?:[\.,]\d+)?){unit_regex}",
        rf"(\d{{1,3}}(?:[\.,]\d+)?)\s*(?:[=:])\s*{label_regex}{unit_regex}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def _extract_bp(text: str, systolic: bool) -> float | None:
    patterns = [
        r"presion\s+arterial\s+de\s+(\d{2,3})\s+(?:sobre|/|y)\s+(\d{2,3})",
        r"presion\s+arterial\s+(\d{2,3})\s*/\s*(\d{2,3})",
        r"pa\s*(\d{2,3})\s*/\s*(\d{2,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1 if systolic else 2))
    return None


def _build_summary_text(diagnosis_text: str, symptoms: Dict[str, int], vitals: Dict[str, Any], urgency_level: str) -> str:
    parts = []
    if symptoms.get("dolor_toracico"):
        parts.append("dolor toracico")
    if symptoms.get("dificultad_respiratoria"):
        parts.append("disnea")
    if symptoms.get("sangrado_activo"):
        parts.append("sangrado activo")
    if symptoms.get("traumatismo_grave"):
        parts.append("traumatismo grave")
    if re.search(r"st\s+elevado|elevacion\s+del\s+segmento\s+st", _normalize_text(diagnosis_text)):
        parts.append("st elevado")

    vitals_parts = []
    if vitals.get("frecuencia_cardiaca") is not None:
        vitals_parts.append(f"FC {int(float(vitals['frecuencia_cardiaca']))}")
    if vitals.get("presion_sistolica") is not None and vitals.get("presion_diastolica") is not None:
        vitals_parts.append(f"TA {int(float(vitals['presion_sistolica']))}/{int(float(vitals['presion_diastolica']))}")
    if vitals.get("saturacion_oxigeno") is not None:
        vitals_parts.append(f"SatO2 {int(float(vitals['saturacion_oxigeno']))}%")
    if vitals.get("frecuencia_respiratoria") is not None:
        vitals_parts.append(f"FR {int(float(vitals['frecuencia_respiratoria']))}")

    summary_bits = []
    if parts:
        summary_bits.append(", ".join(parts))
    if vitals_parts:
        summary_bits.append("; ".join(vitals_parts))
    summary_bits.append(f"urgencia {urgency_level}")

    summary = " | ".join(summary_bits)
    return summary[:240]


def _guess_specialty(symptoms: Dict[str, int]) -> str:
    if symptoms.get("emergencia_obstetrica"):
        return "Obstetricia"
    if symptoms.get("dolor_toracico"):
        return "Cardiología"
    if symptoms.get("deficit_neurologico") or symptoms.get("convulsiones"):
        return "Neurología"
    if symptoms.get("traumatismo_grave") or symptoms.get("quemadura_extensa"):
        return "Traumatología"
    if symptoms.get("riesgo_psiquiatrico"):
        return "Psiquiatría"
    return "Medicina Interna"


def _build_feature_map(structured_output: Dict[str, Any]) -> Dict[str, Any]:
    patient = structured_output.get("patient", {})
    vitals = structured_output.get("vitals", {})
    symptoms = structured_output.get("symptoms", {})

    return {
        "edad": _coalesce_number(patient.get("age"), 0),
        "sexo": _encode_sex(patient.get("sex")),
        "frecuencia_cardiaca": _coalesce_number(vitals.get("frecuencia_cardiaca"), 0),
        "presion_sistolica": _coalesce_number(vitals.get("presion_sistolica"), 0),
        "presion_diastolica": _coalesce_number(vitals.get("presion_diastolica"), 0),
        "saturacion_oxigeno": _coalesce_number(vitals.get("saturacion_oxigeno"), 0),
        "temperatura": _coalesce_number(vitals.get("temperatura"), 0),
        "frecuencia_respiratoria": _coalesce_number(vitals.get("frecuencia_respiratoria"), 0),
        "glucemia": _coalesce_number(vitals.get("glucemia"), 0),
        **{key: int(bool(symptoms.get(key, 0))) for key in SYMPTOM_KEYS},
        "inicio_subito": _infer_instant_onset(symptoms),
        "antecedentes_graves": _infer_grave_history(structured_output.get("risk_flags", [])),
        "specialty_code": SPECIALTY_TO_CODE.get(_normalize_specialty(structured_output.get("specialty_suspected")), 0),
    }


def _infer_instant_onset(symptoms: Dict[str, int]) -> int:
    instant_keys = [
        "dolor_toracico",
        "dificultad_respiratoria",
        "deficit_neurologico",
        "abdomen_agudo",
        "sangrado_activo",
        "traumatismo_grave",
        "convulsiones",
        "anafilaxia_alergia",
        "emergencia_obstetrica",
        "quemadura_extensa",
    ]
    return 1 if any(symptoms.get(key, 0) for key in instant_keys) else 0


def _infer_grave_history(risk_flags: List[str]) -> int:
    return 1 if risk_flags else 0


def _normalize_sex(value: Any) -> str:
    sex = str(value).strip().lower()
    if sex in {"m", "male", "masculino", "hombre", "varon", "varón"}:
        return "M"
    if sex in {"f", "female", "femenino", "mujer"}:
        return "F"
    return "unknown"


def _encode_sex(value: Any) -> int:
    sex = _normalize_sex(value)
    if sex == "M":
        return 1
    return 0


def _normalize_specialty(value: Any) -> str:
    specialty = str(value).strip()
    return specialty if specialty in SPECIALTY_TO_CODE else "Desconocida"


def _normalize_risk_flags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def _clamp_confidence(value: Any) -> float:
    confidence = _coerce_float(value, 0.5)
    return max(0.0, min(1.0, confidence))


def _extract_json_object(raw_output: str) -> Dict[str, Any] | None:
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _to_int_or_none(value: Any) -> int | None:
    if value in {None, "", "null", "None"}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    if value in {None, "", "null", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    coerced = _to_float_or_none(value)
    return default if coerced is None else coerced


def _coalesce_number(value: Any, default: float) -> float:
    if value in {None, "", "null", "None"}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
