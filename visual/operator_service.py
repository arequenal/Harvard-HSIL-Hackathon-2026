from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_llm import FEATURE_ORDER, analyze_clinical_diagnosis
try:
    from ml.urgency_specialty_classifier import transcribe_audio_file
except ImportError:
    transcribe_audio_file = None
from visual.dispatch_shared import load_state, update_state

PROCESSED_HOSPITALES_PATH = (
    PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"
)
AUDIO_SAMPLES_PATH = PROJECT_ROOT / "audio" / "samples"


st.set_page_config(page_title="AmbulancIA - Operador", layout="wide")

st.markdown(
    """
    <style>
            .block-container {max-width: 96%;}
            .section-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #58a6ff;
                margin-bottom: 0.4rem;
            }
      .card {
        border: 1px solid rgba(88,166,255,0.2);
        border-radius: 12px;
        padding: 14px 16px;
        background: #1a1f3a;
      }
      .warn {
        border-left: 4px solid #f39c12;
        background: #2a2415;
        padding: 10px 12px;
        border-radius: 8px;
        color: #fbd86b;
      }
      .ok {
        border-left: 4px solid #3fb950;
        background: #0d3f1a;
        padding: 10px 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        color: #7ee787;
            }
            .soft {
                border: 1px solid rgba(88,166,255,0.15);
                background: #1a1f3a;
                border-radius: 10px;
                padding: 10px 12px;
                color: #e0e0e0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models(models_dir: str = "ml/models") -> Dict[str, Any]:
    base = PROJECT_ROOT / models_dir

    urgency_model = xgb.XGBClassifier()
    urgency_model.load_model(str(base / "urgency_model.json"))

    specialty_model = xgb.XGBClassifier()
    specialty_model.load_model(str(base / "specialty_model.json"))

    with open(base / "metadata.pkl", "rb") as file:
        metadata = pickle.load(file)

    label_encoder = None
    label_encoder_path = base / "label_encoder_urgency.pkl"
    if label_encoder_path.exists():
        try:
            with open(label_encoder_path, "rb") as file:
                label_encoder = pickle.load(file)
        except Exception:
            import joblib  # type: ignore

            label_encoder = joblib.load(label_encoder_path)

    urgency_names = {int(k): str(v) for k, v in metadata.get("urgency_names", {}).items()}
    specialty_names = {int(k): str(v) for k, v in metadata.get("specialty_names", {}).items()}

    return {
        "urgency_model": urgency_model,
        "specialty_model": specialty_model,
        "metadata": metadata,
        "label_encoder": label_encoder,
        "urgency_names": urgency_names,
        "specialty_names": specialty_names,
    }


@st.cache_data
def load_hospitals() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_HOSPITALES_PATH, sep=";")
    if "centro_tipo" in df.columns:
        df = df[df["centro_tipo"].isin({"Hospital general", "Hospital especializado"})].copy()
    for col in ["telefono", "email"]:
        if col not in df.columns:
            df[col] = ""
    for idx, row in df.iterrows():
        cid = str(row.get("centro_id", idx))
        if not str(row.get("telefono", "")).strip():
            df.at[idx, "telefono"] = f"+34 91{str(1000000 + idx)[-7:]}"
        if not str(row.get("email", "")).strip():
            df.at[idx, "email"] = f"admisiones.{cid.lower()}@hospital.madrid.es"
    return df


def decode_urgency_label(raw_prediction: int, label_encoder: Any, urgency_names: Dict[int, str]) -> int:
    if label_encoder is not None:
        return int(label_encoder.inverse_transform([raw_prediction])[0])
    if raw_prediction in urgency_names:
        return raw_prediction
    if (raw_prediction + 1) in urgency_names:
        return raw_prediction + 1
    return raw_prediction


def prob_map(values: List[float], names: Dict[int, str], decoder: Any = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for idx, value in enumerate(values):
        mapped = decoder(idx) if decoder else idx
        out[names.get(mapped, str(mapped))] = float(value)
    return out


def predict_from_feature_map(feature_map: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    row = {col: feature_map.get(col, 0) for col in FEATURE_ORDER}
    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    urg_model = models["urgency_model"]
    spec_model = models["specialty_model"]
    urg_names = models["urgency_names"]
    spec_names = models["specialty_names"]
    label_encoder = models["label_encoder"]

    urg_raw = int(urg_model.predict(df)[0])
    urg_proba = urg_model.predict_proba(df)[0].tolist()
    urg_label = decode_urgency_label(urg_raw, label_encoder, urg_names)

    spec_raw = int(spec_model.predict(df)[0])
    spec_proba = spec_model.predict_proba(df)[0].tolist()

    return {
        "urgencia": {
            "raw_class": urg_raw,
            "label": urg_label,
            "name": urg_names.get(urg_label, str(urg_label)),
            "confidence": float(urg_proba[urg_raw]),
            "probabilities": prob_map(
                urg_proba,
                urg_names,
                decoder=lambda i: decode_urgency_label(i, label_encoder, urg_names),
            ),
        },
        "especialidad": {
            "raw_class": spec_raw,
            "label": spec_raw,
            "name": spec_names.get(spec_raw, str(spec_raw)),
            "confidence": float(spec_proba[spec_raw]),
            "probabilities": prob_map(spec_proba, spec_names),
        },
    }


def parse_editor(df: pd.DataFrame) -> Dict[str, Any]:
    row = df.iloc[0].to_dict()
    out: Dict[str, Any] = {}
    for col in FEATURE_ORDER:
        value = row.get(col, 0)
        if col == "sexo" or col.endswith("_presente") or col.endswith("_negado"):
            try:
                out[col] = 1 if int(float(value)) > 0 else 0
            except Exception:
                out[col] = 0
        else:
            try:
                n = float(value)
                out[col] = int(n) if n.is_integer() else round(n, 2)
            except Exception:
                out[col] = 0
    return out


def deterministic_explanation(feature_map: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    urg = prediction["urgencia"]
    spec = prediction["especialidad"]

    evidence: List[str] = []
    cautions: List[str] = []

    age = float(feature_map.get("edad", 0) or 0)
    sat = float(feature_map.get("saturacion_oxigeno", 0) or 0)
    fc = float(feature_map.get("frecuencia_cardiaca", 0) or 0)
    ps = float(feature_map.get("presion_sistolica", 0) or 0)
    pd = float(feature_map.get("presion_diastolica", 0) or 0)

    if age >= 65:
        evidence.append(f"Edad avanzada ({age:.0f} anos)")
    if sat and sat < 92:
        evidence.append(f"Saturacion baja ({sat:.0f}%)")
    if fc and fc > 110:
        evidence.append(f"Taquicardia ({fc:.0f} lpm)")
    if ps >= 180 or pd >= 110:
        evidence.append(f"PA alta ({ps:.0f}/{pd:.0f})")

    symptom_map = {
        "dolor_toracico_presente": "Dolor toracico presente",
        "dificultad_respiratoria_presente": "Dificultad respiratoria presente",
        "deficit_neurologico_presente": "Deficit neurologico presente",
        "sangrado_activo_presente": "Sangrado activo presente",
        "convulsiones_presente": "Convulsiones presentes",
        "traumatismo_grave_presente": "Traumatismo grave presente",
        "emergencia_obstetrica_presente": "Emergencia obstetrica presente",
    }
    for key, text in symptom_map.items():
        if int(feature_map.get(key, 0)) == 1:
            evidence.append(text)

    for key in [
        "frecuencia_cardiaca",
        "saturacion_oxigeno",
        "frecuencia_respiratoria",
        "temperatura",
        "glucemia",
        "escala_glasgow",
    ]:
        if float(feature_map.get(key, 0) or 0) == 0:
            cautions.append(f"{key} no informado (valor 0)")

    if urg["confidence"] < 0.60:
        cautions.append("Confianza baja en urgencia (<0.60): se recomienda validacion humana adicional")
    if spec["confidence"] < 0.60:
        cautions.append("Confianza baja en especialidad (<0.60): revisar derivacion")

    asociacion_por_sintoma = {
        "dolor_toracico_presente": "patologia cardiovascular aguda o dolor toracico de alto riesgo",
        "dificultad_respiratoria_presente": "insuficiencia respiratoria o descompensacion cardiorrespiratoria",
        "deficit_neurologico_presente": "evento neurologico agudo",
        "sangrado_activo_presente": "hemorragia activa con riesgo hemodinamico",
        "convulsiones_presente": "crisis neurologica aguda",
        "traumatismo_grave_presente": "lesion traumatica mayor",
        "emergencia_obstetrica_presente": "complicacion obstetrica urgente",
    }

    asociaciones: List[str] = []
    for key, texto in asociacion_por_sintoma.items():
        if int(feature_map.get(key, 0)) == 1:
            asociaciones.append(texto)

    if sat and sat < 92:
        asociaciones.append("hipoxemia clinicamente relevante")
    if ps >= 180 or pd >= 110:
        asociaciones.append("inestabilidad hemodinamica por hipertension severa")

    asociaciones = list(dict.fromkeys(asociaciones))

    urgency_text = (
        f"La urgencia estimada es {urg['name']} con confianza {urg['confidence']:.2f}. "
        f"Se apoya en hallazgos del vector como: {', '.join(evidence[:6]) if evidence else 'ausencia de marcadores criticos claros'}. "
        "Esta explicacion es orientativa y debe validarse con criterio clinico del equipo asistencial."
    )

    specialty_text = (
        f"Se le quiere llevar a {spec['name']} con confianza {spec['confidence']:.2f}, "
        f"debido a que presenta sintomas de {', '.join(evidence[:5]).lower() if evidence else 'sintomatologia no concluyente'} "
        f"asociadas normalmente a problemas como {', '.join(asociaciones[:4]) if asociaciones else 'cuadros clinicos que requieren valoracion hospitalaria general'}."
    )

    return {
        "urgencia_text": urgency_text,
        "especialidad_text": specialty_text,
        "evidence": evidence,
        "cautions": cautions,
    }


def optional_llm_refine(base_explanation: Dict[str, Any], model_name: str) -> str:
    """Refina redaccion usando LLM sin introducir hechos nuevos.

    Se usa solo si el operador lo activa y devuelve texto conservador.
    """

    prompt = (
        "Reescribe en castellano clinico operativo, maximo 6 lineas, sin inventar datos, "
        "sin diagnosticos nuevos y sin recomendaciones terapeuticas.\n\n"
        "HECHOS PERMITIDOS:\n"
        f"- Urgencia: {base_explanation['urgencia_text']}\n"
        f"- Especialidad: {base_explanation['especialidad_text']}\n"
        f"- Evidencias: {', '.join(base_explanation['evidence']) if base_explanation['evidence'] else 'ninguna'}\n"
        f"- Precauciones: {', '.join(base_explanation['cautions']) if base_explanation['cautions'] else 'ninguna'}\n"
        "Salida: solo texto plano, no markdown."
    )

    try:
        data = analyze_clinical_diagnosis(prompt, model=model_name)
        # No confiamos en que devuelva prose, asi que esto puede no ser util con este extractor.
        # Devolvemos fallback determinista cuando no aporta contenido textual util.
        raw = str(data.get("raw_model_output", "")).strip()
        if raw and len(raw) > 40:
            return raw[:900]
    except Exception:
        pass

    return (
        base_explanation["urgencia_text"]
        + "\n"
        + base_explanation["especialidad_text"]
        + ("\nPrecauciones: " + "; ".join(base_explanation["cautions"]) if base_explanation["cautions"] else "")
    )


def select_hospital(hospitals: pd.DataFrame, specialty_name: str) -> pd.Series:
    filtered = hospitals
    if specialty_name:
        m = hospitals["especialidades_texto"].astype(str).str.contains(specialty_name, case=False, na=False)
        if m.any():
            filtered = hospitals[m]
    return filtered.iloc[0]


def get_audio_samples() -> List[Path]:
    AUDIO_SAMPLES_PATH.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    for pattern in ["*.wav", "*.mp3", "*.m4a", "*.ogg"]:
        files.extend(sorted(AUDIO_SAMPLES_PATH.glob(pattern)))
    return files


def main() -> None:
    st.title("AmbulancIA - Servicio Operador")
    st.caption("Triaje asistido por IA para generar vector clinico, explicar decision y publicar destino al conductor.")

    models = load_models()
    hospitals = load_hospitals()

    if "operator_text" not in st.session_state:
        st.session_state["operator_text"] = ""

    top_a, top_b = st.columns([1, 1], gap="large")
    with top_a:
        st.markdown('<div class="soft"><b>Flujo:</b> audio/texto → vector → prediccion → explicabilidad → publicacion al conductor.</div>', unsafe_allow_html=True)
    with top_b:
        state_now = load_state()
        st.markdown(
            f'<div class="soft"><b>Version compartida actual:</b> {state_now.get("version", 0)} | '
            f'<b>Ultima actualizacion:</b> {state_now.get("updated_at", "-")}</div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns([1.4, 1], gap="large")

    with c1:
        st.markdown('<div class="section-title">1) Entrada de audio y texto</div>', unsafe_allow_html=True)
        audio_samples = get_audio_samples()
        audio_options = ["-- seleccionar --"] + [str(path.relative_to(PROJECT_ROOT)) for path in audio_samples]
        selected_audio = st.selectbox("Sample de audio", audio_options)

        a1, a2 = st.columns(2)
        with a1:
            stt_model = st.selectbox("Modelo STT", ["tiny", "base", "small", "medium", "large-v3"], index=3)
        with a2:
            stt_lang = st.selectbox("Idioma", ["es", "en", "auto"], index=0)

        if st.button("Transcribir", use_container_width=True):
            if transcribe_audio_file is None:
                st.error(
                    "La funcion de transcripcion no esta disponible en ml/urgency_specialty_classifier.py. "
                    "Puedes seguir usando el dispatch con texto manual."
                )
                st.stop()
            if selected_audio == "-- seleccionar --":
                st.warning("Selecciona un archivo en audio/samples")
            else:
                try:
                    lang = None if stt_lang == "auto" else stt_lang
                    tr = transcribe_audio_file(
                        str(PROJECT_ROOT / selected_audio),
                        model_size=stt_model,
                        language=lang,
                        device="auto",
                        compute_type="auto",
                    )
                    st.session_state["operator_text"] = tr["text"]
                    st.success("Transcripcion completada")
                    st.caption(f"Idioma detectado: {tr['language']} (p={tr['language_probability']:.2f})")
                except Exception as exc:
                    st.error(f"Error de transcripcion: {exc}")

        llm_model = st.text_input("Modelo LLM extractor", value=os.getenv("AMBULANCIA_LLM_MODEL", "llama3.1:8b-instruct"))
        text = st.text_area("Texto clinico editable", value=st.session_state["operator_text"], height=210)
        st.session_state["operator_text"] = text

        if st.button("Generar vector + prediccion", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("Escribe texto clinico o transcribe audio")
            else:
                with st.spinner("Analizando caso..."):
                    extraction = analyze_clinical_diagnosis(text, model=llm_model)
                    feature_map = extraction["feature_map"]
                    prediction = predict_from_feature_map(feature_map, models)
                    explanation = deterministic_explanation(feature_map, prediction)

                st.session_state["extraction"] = extraction
                st.session_state["feature_map"] = feature_map
                st.session_state["prediction"] = prediction
                st.session_state["explanation"] = explanation

    with c2:
        st.markdown('<div class="section-title">2) Resultado rapido y explicabilidad</div>', unsafe_allow_html=True)
        if "prediction" in st.session_state:
            pred = st.session_state["prediction"]
            st.metric("Urgencia", pred["urgencia"]["name"], f"Conf. {pred['urgencia']['confidence']:.2f}")
            st.metric("Especialidad", pred["especialidad"]["name"], f"Conf. {pred['especialidad']['confidence']:.2f}")

            exp = st.session_state["explanation"]
            st.markdown("**Explicabilidad de urgencia**")
            st.markdown('<div class="ok">' + exp["urgencia_text"] + "</div>", unsafe_allow_html=True)

            st.markdown("**Explicabilidad de especialidad**")
            st.markdown('<div class="ok" style="margin-top:8px;">' + exp["especialidad_text"] + "</div>", unsafe_allow_html=True)

            if exp["cautions"]:
                st.markdown("<div class='warn'><b>Precauciones de seguridad:</b><br>" + "<br>".join(exp["cautions"]) + "</div>", unsafe_allow_html=True)

            use_llm_polish = st.checkbox("Refinar explicacion con LLM (controlado)", value=False)
            if use_llm_polish:
                polished = optional_llm_refine(exp, llm_model)
                st.text_area("Resumen narrativo", value=polished, height=140)
        else:
            st.info("Aun no hay resultado. Al generar el vector apareceran aqui las explicaciones de urgencia y especialidad.")

    if "feature_map" in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-title">3) Edicion de vector y decision final</div>', unsafe_allow_html=True)

        fm = st.session_state["feature_map"]
        vector_df = pd.DataFrame([{c: fm.get(c, 0) for c in FEATURE_ORDER}], columns=FEATURE_ORDER)
        edited = st.data_editor(vector_df, use_container_width=True, num_rows="fixed")

        if st.button("Recalcular desde vector editado"):
            new_fm = parse_editor(edited)
            new_pred = predict_from_feature_map(new_fm, models)
            new_exp = deterministic_explanation(new_fm, new_pred)
            st.session_state["feature_map"] = new_fm
            st.session_state["prediction"] = new_pred
            st.session_state["explanation"] = new_exp
            st.success("Recalculado")

        pred = st.session_state["prediction"]
        urg_options = list(models["urgency_names"].values())
        spec_options = list(models["specialty_names"].values())

        d1, d2 = st.columns(2)
        with d1:
            urg_idx = urg_options.index(pred["urgencia"]["name"]) if pred["urgencia"]["name"] in urg_options else 0
            final_urg = st.selectbox("Urgencia final (editable)", urg_options, index=urg_idx)
        with d2:
            spec_idx = spec_options.index(pred["especialidad"]["name"]) if pred["especialidad"]["name"] in spec_options else 0
            final_spec = st.selectbox("Especialidad final (editable)", spec_options, index=spec_idx)

        notes = st.text_area("Notas del operador", height=90, placeholder="Contexto adicional y justificacion")

        chosen_hosp = select_hospital(hospitals, final_spec)
        st.markdown('<div class="section-title">4) Hospital de referencia y contacto</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Hospital": chosen_hosp.get("nombre", ""),
                        "Telefono": chosen_hosp.get("telefono", ""),
                        "Email": chosen_hosp.get("email", ""),
                        "Direccion": chosen_hosp.get("direccion_completa", ""),
                        "Especialidades": chosen_hosp.get("especialidades_texto", ""),
                    }
                ]
            ),
            use_container_width=True,
        )

        st.markdown('<div class="section-title">5) Publicar al servicio del conductor</div>', unsafe_allow_html=True)

        default_alerts = [
            "Atasco severo en eje principal",
            "Obras en carril de acceso",
            "Evento masivo con cortes puntuales",
        ]
        selected_alerts = st.multiselect("Alertas para conductor", default_alerts, default=[default_alerts[0]])

        eta = st.slider("ETA estimada al hospital (min)", min_value=4, max_value=45, value=12)

        if st.button("Publicar decision y notificar conductor", type="primary"):
            patch = {
                "case": {
                    "summary": f"{final_urg} / {final_spec}",
                    "source_text": st.session_state.get("operator_text", "").strip(),
                    "urgencia": final_urg,
                    "especialidad": final_spec,
                    "explicacion_urgencia": st.session_state["explanation"]["urgencia_text"],
                    "explicacion_especialidad": st.session_state["explanation"]["especialidad_text"],
                },
                "destination": {
                    "centro_id": str(chosen_hosp.get("centro_id", "")),
                    "nombre": str(chosen_hosp.get("nombre", "")),
                    "telefono": str(chosen_hosp.get("telefono", "")),
                    "email": str(chosen_hosp.get("email", "")),
                    "direccion": str(chosen_hosp.get("direccion_completa", "")),
                    "eta_min": int(eta),
                },
                "traffic_alerts": selected_alerts,
                "operator_notes": notes.strip(),
            }
            new_state = update_state(patch)
            st.success(f"Publicado al conductor. Version compartida: {new_state['version']}")

    st.markdown("---")
    st.markdown('<div class="section-title">Estado compartido actual</div>', unsafe_allow_html=True)
    st.json(load_state())


if __name__ == "__main__":
    main()
