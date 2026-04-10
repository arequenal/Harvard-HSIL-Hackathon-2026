"""Utilities for extracting dataset-ready features from 112 call narratives."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Tuple


VITAL_COLUMNS = [
    "edad",
    "sexo",
    "frecuencia_cardiaca",
    "presion_sistolica",
    "presion_diastolica",
    "saturacion_oxigeno",
    "frecuencia_respiratoria",
    "temperatura",
    "glucemia",
    "escala_glasgow",
]

BINARY_FACTORS = [
    "inicio_subito",
    "antecedentes_graves",
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

BINARY_PAIR_COLUMNS = [
    column
    for factor in BINARY_FACTORS
    for column in (f"{factor}_presente", f"{factor}_negado")
]

FEATURE_ORDER = [*VITAL_COLUMNS, *BINARY_PAIR_COLUMNS]

DEFAULT_MODEL = os.getenv("AMBULANCIA_LLM_MODEL", "llama3.1:8b-instruct")
DEFAULT_PROVIDER = os.getenv("AMBULANCIA_LLM_PROVIDER", "ollama")
DEFAULT_BASE_URL = os.getenv("AMBULANCIA_LLM_BASE_URL", "http://localhost:11434")

FACTOR_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "inicio_subito": {
        "terms": [
            r"inicio\s+subit",
            r"de\s+repente",
            r"brusc",
            r"repentin",
            r"comenzo\s+de\s+golpe",
            r"empezo\s+de\s+golpe",
        ],
        "explicit_negations": [
            r"inicio\s+gradual",
            r"progresiv",
            r"de\s+evolucion\s+lenta",
            r"desde\s+hace\s+\d+\s*(?:dias|semanas)",
        ],
    },
    "antecedentes_graves": {
        "terms": [
            r"antecedentes",
            r"hta",
            r"hipertension",
            r"diabetes",
            r"epoc",
            r"cardiopati",
            r"insuficiencia\s+cardiaca",
            r"ictus\s+previo",
            r"anticoagul",
            r"cancer",
        ],
        "explicit_negations": [
            r"sin\s+antecedentes",
            r"niega\s+antecedentes",
            r"sin\s+enfermedades\s+previas",
            r"sin\s+patologia\s+previa",
        ],
    },
    "dolor_toracico": {
        "terms": [
            r"dolor\s+toracic",
            r"dolor\s+(?:en\s+el\s+)?pecho",
            r"opresiv[oa]\s+toracic",
        ]
    },
    "dificultad_respiratoria": {
        "terms": [
            r"disnea",
            r"dificultad\s+para\s+respirar",
            r"falta\s+de\s+aire",
            r"ahogo",
        ]
    },
    "deficit_neurologico": {
        "terms": [
            r"debilidad\s+de\s+un\s+lado",
            r"hemipares",
            r"afasia",
            r"boca\s+torcida",
            r"deficit\s+neurologic",
            r"no\s+puede\s+hablar\s+bien",
        ]
    },
    "abdomen_agudo": {"terms": [r"abdomen\s+agudo", r"dolor\s+abdominal\s+intenso"]},
    "sangrado_activo": {"terms": [r"sangrado", r"hemorragia", r"sangra\s+activamente"]},
    "traumatismo_grave": {
        "terms": [
            r"politrauma",
            r"traumatism",
            r"accidente\s+grave",
            r"fractura\s+abierta",
        ]
    },
    "convulsiones": {"terms": [r"convulsion", r"crisis\s+convulsiv", r"epileptic"]},
    "intoxicacion_sobredosis": {"terms": [r"sobredosis", r"intoxicacion", r"ingesta\s+de\s+toxic"]},
    "anafilaxia_alergia": {"terms": [r"anafilaxia", r"reaccion\s+alergica\s+grave", r"edema\s+de\s+glotis"]},
    "emergencia_obstetrica": {"terms": [r"embaraz", r"parto", r"hemorragia\s+vaginal", r"preeclampsia"]},
    "quemadura_extensa": {"terms": [r"quemadura", r"gran\s+quemado"]},
    "riesgo_psiquiatrico": {
        "terms": [
            r"riesgo\s+psiquiatrico",
            r"ideacion\s+suicida",
            r"intento\s+autolitic",
            r"agresividad\s+severa",
        ]
    },
    "vomitos_diarrea_severa": {
        "terms": [
            r"vomit",
            r"diarrea",
            r"gastroenteritis\s+severa",
            r"deshidratacion\s+severa",
        ]
    },
}


def build_prompt(diagnosis_text: str) -> str:
    """Build a strict extraction prompt for Llama 3."""

    schema = _empty_feature_map()
    ordered_columns = ", ".join(FEATURE_ORDER)

    return (
        "Eres un operador del 112 en Espana con mas de 20 anos de experiencia en emergencias prehospitalarias. "
        "Has atendido miles de llamadas reales y sabes diferenciar lo afirmado, lo negado y lo no mencionado con precision clinica. "
        "Tu tarea ahora es extraer datos estructurados para un dataset de entrenamiento.\n\n"
        "OBJETIVO: convertir el texto de una llamada/nota clinica en un JSON con TODAS las columnas del dataset (excepto especialidad y nivel_urgencia).\n"
        "Debes devolver SIEMPRE todas las claves, sin omitir ninguna, y con valor numerico.\n\n"
        "REGLAS CLAVE (OBLIGATORIAS):\n"
        "1) No inventes informacion: solo usa lo que este en el texto.\n"
        "2) Para cada factor binario duplicado (X_presente / X_negado):\n"
        "   - Si el sintoma/factor aparece afirmado: X_presente=1 y X_negado=0.\n"
        "   - Si aparece negado explicitamente: X_presente=0 y X_negado=1.\n"
        "   - Si no aparece de ninguna forma: X_presente=0 y X_negado=0.\n"
        "3) Si en el texto aparecen tanto afirmacion como negacion del mismo factor en momentos distintos, marca ambos en 1.\n"
        "4) Todas las columnas no mencionadas deben quedarse en 0.\n"
        "5) Sexo: hombre/varon/masculino=1, mujer/femenino=0, desconocido=0.\n"
        "6) Valores numericos (edad y constantes vitales):\n"
        "   - Si se mencionan claramente, extraelos como numero.\n"
        "   - Si no se mencionan, pon 0.\n"
        "   - Presion arterial tipo 178/102 -> presion_sistolica=178 y presion_diastolica=102.\n"
        "   - Escala Glasgow: extrae el numero si aparece (por ejemplo GCS 13, Glasgow 15/15).\n"
        "7) Responde SOLO JSON valido. Sin markdown, sin explicaciones, sin texto extra.\n"
        "8) No anadas claves nuevas ni cambies nombres.\n\n"
        "Columnas exactas y orden objetivo del vector:\n"
        f"{ordered_columns}\n\n"
        "JSON esperado (ejemplo de estructura; usa tus valores extraidos):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Texto de la llamada/nota clinica a extraer:\n"
        f"{diagnosis_text.strip()}"
    )


def analyze_clinical_diagnosis(
    diagnosis_text: str,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    base_url: str = DEFAULT_BASE_URL,
) -> Dict[str, Any]:
    """Convert free text into full dataset columns and a numeric feature vector."""

    cleaned_text = diagnosis_text.strip()
    if not cleaned_text:
        raise ValueError("La nota clinica no puede estar vacia.")

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
        llm_feature_map = _heuristic_extract_feature_map(cleaned_text)
        raw_output = json.dumps(llm_feature_map, ensure_ascii=False)
    else:
        llm_feature_map = _normalize_structured_output(raw_output, cleaned_text)

    fallback_feature_map = _heuristic_extract_feature_map(cleaned_text)
    feature_map = _merge_feature_maps(base=fallback_feature_map, preferred=llm_feature_map)
    feature_vector = [feature_map[key] for key in FEATURE_ORDER]

    return {
        "provider": provider if not used_fallback else "fallback",
        "model": model,
        "fallback_used": used_fallback,
        "source_text": cleaned_text,
        "raw_model_output": raw_output,
        "structured_output": feature_map,
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
            {
                "role": "system",
                "content": (
                    "Eres un extractor de datos clinicos para entrenamiento de modelos. "
                    "Devuelve exclusivamente un JSON valido con las claves solicitadas."
                ),
            },
            {"role": "user", "content": build_prompt(diagnosis_text)},
        ],
    }

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    content = response_payload.get("message", {}).get("content", "")
    if not content:
        raise ValueError("El modelo no devolvio contenido.")
    return content


def _normalize_structured_output(raw_output: str, diagnosis_text: str) -> Dict[str, Any]:
    payload = _extract_json_payload(raw_output)
    if payload is None:
        return _heuristic_extract_feature_map(diagnosis_text)

    candidate = _select_candidate_map(payload)
    if candidate is None:
        return _heuristic_extract_feature_map(diagnosis_text)

    return _coerce_candidate_to_feature_map(candidate=candidate, payload=payload)


def _select_candidate_map(payload: Any) -> Dict[str, Any] | None:
    if isinstance(payload, list):
        if len(payload) == len(FEATURE_ORDER):
            return dict(zip(FEATURE_ORDER, payload))
        return None

    if not isinstance(payload, dict):
        return None

    for key in ("feature_map", "features", "columns", "output"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
        if isinstance(nested, list) and len(nested) == len(FEATURE_ORDER):
            return dict(zip(FEATURE_ORDER, nested))

    if all(column in payload for column in FEATURE_ORDER):
        return payload

    return payload


def _coerce_candidate_to_feature_map(candidate: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    feature_map = _empty_feature_map()

    payload_dict = payload if isinstance(payload, dict) else {}
    patient = payload_dict.get("patient", {}) if isinstance(payload_dict.get("patient", {}), dict) else {}
    vitals = payload_dict.get("vitals", {}) if isinstance(payload_dict.get("vitals", {}), dict) else {}
    symptoms = payload_dict.get("symptoms", {}) if isinstance(payload_dict.get("symptoms", {}), dict) else {}
    negated_symptoms = payload_dict.get("symptoms_negados", {})
    if not isinstance(negated_symptoms, dict):
        negated_symptoms = payload_dict.get("symptoms_negated", {})
    if not isinstance(negated_symptoms, dict):
        negated_symptoms = {}

    age_value = candidate.get("edad", patient.get("age"))
    sex_value = candidate.get("sexo", patient.get("sex"))

    feature_map["edad"] = _coerce_numeric(age_value, default=0)
    feature_map["sexo"] = _coerce_sex(sex_value)

    for vital in [
        "frecuencia_cardiaca",
        "presion_sistolica",
        "presion_diastolica",
        "saturacion_oxigeno",
        "frecuencia_respiratoria",
        "temperatura",
        "glucemia",
        "escala_glasgow",
    ]:
        source_value = candidate.get(vital, vitals.get(vital))
        feature_map[vital] = _coerce_numeric(source_value, default=0)

    for factor in BINARY_FACTORS:
        present_key = f"{factor}_presente"
        negated_key = f"{factor}_negado"

        present_raw = candidate.get(present_key)
        negated_raw = candidate.get(negated_key)

        if present_raw is None:
            present_raw = candidate.get(factor, symptoms.get(factor))

        if negated_raw is None:
            negated_raw = negated_symptoms.get(factor)

        feature_map[present_key] = _coerce_binary(present_raw)
        feature_map[negated_key] = _coerce_binary(negated_raw)

    return _ensure_feature_map_shape(feature_map)


def _merge_feature_maps(base: Dict[str, Any], preferred: Dict[str, Any]) -> Dict[str, Any]:
    merged = _empty_feature_map()

    base = _ensure_feature_map_shape(base)
    preferred = _ensure_feature_map_shape(preferred)

    for column in FEATURE_ORDER:
        if column in BINARY_PAIR_COLUMNS:
            merged[column] = 1 if _coerce_binary(base.get(column)) or _coerce_binary(preferred.get(column)) else 0
            continue

        if column == "sexo":
            merged[column] = 1 if _coerce_sex(base.get(column)) == 1 or _coerce_sex(preferred.get(column)) == 1 else 0
            continue

        preferred_value = _coerce_numeric(preferred.get(column), default=0)
        base_value = _coerce_numeric(base.get(column), default=0)
        merged[column] = preferred_value if preferred_value != 0 else base_value

    return merged


def _heuristic_extract_feature_map(diagnosis_text: str) -> Dict[str, Any]:
    text = _normalize_text(diagnosis_text)
    feature_map = _empty_feature_map()

    age = _extract_age(text)
    if age is not None:
        feature_map["edad"] = age

    sex = _extract_sex(text)
    feature_map["sexo"] = _coerce_sex(sex)

    bp_systolic, bp_diastolic = _extract_bp(text)

    feature_map["frecuencia_cardiaca"] = _coerce_numeric(
        _extract_vital_value(text, [r"frecuencia\s+cardiaca", r"\bfc\b", r"pulso"]),
        default=0,
    )
    feature_map["presion_sistolica"] = _coerce_numeric(bp_systolic, default=0)
    feature_map["presion_diastolica"] = _coerce_numeric(bp_diastolic, default=0)
    feature_map["saturacion_oxigeno"] = _coerce_numeric(
        _extract_vital_value(text, [r"saturacion\s+de\s+oxigeno", r"saturacion", r"spo2", r"sao2"]),
        default=0,
    )
    feature_map["frecuencia_respiratoria"] = _coerce_numeric(
        _extract_vital_value(text, [r"frecuencia\s+respiratoria", r"\bfr\b"]),
        default=0,
    )
    feature_map["temperatura"] = _coerce_numeric(_extract_temperature(text), default=0)
    feature_map["glucemia"] = _coerce_numeric(_extract_vital_value(text, [r"glucemia", r"glucosa"]), default=0)
    feature_map["escala_glasgow"] = _coerce_numeric(_extract_glasgow(text), default=0)

    for factor in BINARY_FACTORS:
        present, negated = _extract_factor_pair(text, factor)
        feature_map[f"{factor}_presente"] = present
        feature_map[f"{factor}_negado"] = negated

    return _ensure_feature_map_shape(feature_map)


def _extract_factor_pair(text: str, factor: str) -> Tuple[int, int]:
    rules = FACTOR_PATTERNS.get(factor, {})
    terms = rules.get("terms", [])
    explicit_negations = rules.get("explicit_negations", [])

    present = 0
    negated = 0

    for neg_pattern in explicit_negations:
        if re.search(neg_pattern, text):
            negated = 1

    for term_pattern in terms:
        for match in re.finditer(term_pattern, text):
            if _is_negated_mention(text, match.start(), match.end()):
                negated = 1
            else:
                present = 1

    return present, negated


def _is_negated_mention(text: str, start_idx: int, end_idx: int) -> bool:
    left_context = text[max(0, start_idx - 80) : start_idx]
    right_context = text[end_idx : min(len(text), end_idx + 40)]

    left_negation_patterns = [
        r"(?:\bno\b|\bsin\b|\bniega(?:n)?\b|\bnegativo\s+para\b|\bausencia\s+de\b|\bdescarta(?:n)?\b)(?:\W+\w+){0,4}\W*$",
        r"(?:\bno\s+presenta\b|\bno\s+tiene\b)(?:\W+\w+){0,4}\W*$",
    ]

    if any(re.search(pattern, left_context) for pattern in left_negation_patterns):
        return True

    if re.search(r"^\W*(?:ausente|ausencia|negativo|descartado)\b", right_context):
        return True

    return False


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
        r"(?:paciente|varon|hombre|mujer)?\s*(?:de|unos|de unos)?\s*(\d{1,3})\s*anos",
        r"edad\s*(?:de)?\s*(\d{1,3})\s*anos",
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


def _extract_vital_value(text: str, label_patterns: List[str]) -> float | None:
    label_regex = r"(?:" + r"|".join(label_patterns) + r")"
    patterns = [
        rf"{label_regex}\s*(?:de|del|:|=)?\s*(\d{{1,3}}(?:[\.,]\d+)?)",
        rf"(\d{{1,3}}(?:[\.,]\d+)?)\s*(?:de|del|:|=)?\s*{label_regex}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _to_float_or_none(match.group(1))
    return None


def _extract_temperature(text: str) -> float | None:
    patterns = [
        r"temperatura\s*(?:de|:|=)?\s*(\d{2}(?:[\.,]\d+)?)",
        r"(\d{2}(?:[\.,]\d+)?)\s*(?:grados|c)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _to_float_or_none(match.group(1))
    return None


def _extract_glasgow(text: str) -> float | None:
    patterns = [
        r"(?:escala\s+de\s+glasgow|glasgow|gcs)\s*(?:de|:|=)?\s*(\d{1,2})",
        r"glasgow\s*(\d{1,2})\s*/\s*15",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 15:
                return float(value)
    return None


def _extract_bp(text: str) -> Tuple[float | None, float | None]:
    patterns = [
        r"presion\s+arterial\s*(?:de|:|=)?\s*(\d{2,3})\s*(?:/|sobre|y)\s*(\d{2,3})",
        r"\bpa\b\s*(\d{2,3})\s*/\s*(\d{2,3})",
        r"\bta\b\s*(\d{2,3})\s*/\s*(\d{2,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return (_to_float_or_none(match.group(1)), _to_float_or_none(match.group(2)))
    return (None, None)


def _extract_json_payload(raw_output: str) -> Any | None:
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _coerce_sex(value: Any) -> int:
    if isinstance(value, (int, float)):
        return 1 if float(value) == 1 else 0

    text = str(value).strip().lower()
    if text in {"m", "male", "masculino", "hombre", "varon", "varón", "1"}:
        return 1
    if text in {"f", "female", "femenino", "mujer", "0", "unknown", "desconocido", "none", ""}:
        return 0
    return 0


def _coerce_binary(value: Any) -> int:
    if value in {None, "", "null", "None"}:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) > 0 else 0

    text = str(value).strip().lower()
    if text in {"1", "true", "si", "sí", "presente", "positivo", "afirmativo", "yes"}:
        return 1
    if text in {"0", "false", "no", "negado", "ausente", "negativo"}:
        return 0
    return 0


def _coerce_numeric(value: Any, default: float = 0) -> float | int:
    parsed = _to_float_or_none(value)
    if parsed is None:
        parsed = float(default)

    if float(parsed).is_integer():
        return int(parsed)
    return round(float(parsed), 2)


def _to_float_or_none(value: Any) -> float | None:
    if value in {None, "", "null", "None"}:
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", ".")
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_feature_map() -> Dict[str, Any]:
    return {column: 0 for column in FEATURE_ORDER}


def _ensure_feature_map_shape(feature_map: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _empty_feature_map()
    for column in FEATURE_ORDER:
        raw = feature_map.get(column, 0)
        if column == "sexo":
            normalized[column] = _coerce_sex(raw)
        elif column in BINARY_PAIR_COLUMNS:
            normalized[column] = _coerce_binary(raw)
        else:
            normalized[column] = _coerce_numeric(raw, default=0)
    return normalized