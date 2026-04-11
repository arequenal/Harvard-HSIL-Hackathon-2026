from __future__ import annotations

import html
import os
import pickle
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import osmnx as ox
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
GRAPH_PATH = PROJECT_ROOT / "visual" / "madrid_grafo.graphml"
SAMUR_BASES_PATH = PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "bases_samur_madrid.csv"
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
            .kpi-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-bottom: 10px;
            }
            .kpi {
                border: 1px solid rgba(88,166,255,0.18);
                border-radius: 10px;
                background: #161c34;
                padding: 10px 12px;
            }
            .kpi-label {
                font-size: 0.72rem;
                color: #8bb9ff;
                margin-bottom: 4px;
            }
            .kpi-main {
                font-size: 1rem;
                font-weight: 700;
                color: #e6f0ff;
            }
            .kpi-sub {
                font-size: 0.78rem;
                color: #9db6d6;
            }
            .exp-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin: 8px 0 10px;
            }
            .exp-card {
                border: 1px solid rgba(63,185,80,0.25);
                border-radius: 10px;
                background: #0d3f1a;
                padding: 10px 12px;
                min-height: 142px;
            }
            .exp-title {
                font-size: 0.78rem;
                font-weight: 700;
                color: #baf2c3;
                margin-bottom: 6px;
            }
            .exp-body {
                font-size: 0.84rem;
                line-height: 1.35;
                color: #7ee787;
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

    for col in ["lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    graph = load_dispatch_graph()
    if graph is not None and "lat" in df.columns and "lon" in df.columns:
        if "nodo_red" not in df.columns:
            df["nodo_red"] = pd.NA

        mask = df["lat"].notna() & df["lon"].notna() & df["nodo_red"].isna()
        if mask.any():
            df.loc[mask, "nodo_red"] = df.loc[mask].apply(
                lambda row: int(ox.distance.nearest_nodes(graph, X=float(row["lon"]), Y=float(row["lat"]))),
                axis=1,
            )

    return df


@st.cache_resource
def load_dispatch_graph() -> nx.MultiDiGraph | None:
    if not GRAPH_PATH.exists():
        return None
    return ox.load_graphml(GRAPH_PATH)


@st.cache_data
def load_samur_bases_for_dispatch(_graph: nx.MultiDiGraph | None) -> List[Dict[str, Any]]:
    if _graph is None:
        return []

    if SAMUR_BASES_PATH.exists():
        bases_df = pd.read_csv(SAMUR_BASES_PATH, sep=";")
        bases_df = bases_df.rename(columns={"latitud": "lat", "longitud": "lon"})
        for col in ["lat", "lon"]:
            bases_df[col] = pd.to_numeric(bases_df[col], errors="coerce")
        bases_df = bases_df[bases_df["lat"].notna() & bases_df["lon"].notna()].copy()

        out: List[Dict[str, Any]] = []
        for _, row in bases_df.iterrows():
            out.append(
                {
                    "nombre": str(row.get("nombre", "Base SAMUR")).strip() or "Base SAMUR",
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "nodo_red": int(ox.distance.nearest_nodes(_graph, X=float(row["lon"]), Y=float(row["lat"]))),
                }
            )
        if out:
            return out

    return [
        {
            "nombre": "Base Central",
            "lat": 40.4117,
            "lon": -3.7430,
            "nodo_red": int(ox.distance.nearest_nodes(_graph, X=-3.7430, Y=40.4117)),
        }
    ]


def _normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    clean = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(clean.lower().strip().split())


def _multi_source_min_costs(graph: nx.MultiDiGraph, base_nodes: List[int]) -> Dict[int, float]:
    unique_sources = [int(n) for n in dict.fromkeys(base_nodes)]
    if not unique_sources:
        return {}

    best_costs: Dict[int, float] = {}
    for weight in ("weighted_length", "length"):
        try:
            partial = nx.multi_source_dijkstra_path_length(graph, unique_sources, weight=weight)
        except Exception:
            continue

        for node, value in partial.items():
            v = float(value)
            if node not in best_costs or v < best_costs[node]:
                best_costs[node] = v

    return best_costs


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


def select_hospital(
    hospitals: pd.DataFrame,
    specialty_name: str,
    sos_lat: float | None = None,
    sos_lon: float | None = None,
) -> pd.Series:
    """Devuelve el hospital más cercano al punto SOS que tenga la especialidad requerida.

    Estrategia de filtrado por especialidad (en orden de preferencia):
      1. Coincidencia exacta por substring normalizado.
      2. Fuzzy matching (SequenceMatcher >= 0.55) sobre cada token de especialidades.
      3. Sin filtro: se usa toda la red de hospitales.

    El coste de viaje se calcula desde el nodo SOS (posición del paciente) hasta cada
    hospital candidato usando Dijkstra sobre el grafo vial con tráfico.  Si no se
    proporciona la posición SOS, se cae al comportamiento anterior (origen = bases SAMUR).
    """
    if hospitals.empty:
        raise ValueError("No hay hospitales disponibles para seleccionar")

    graph = load_dispatch_graph()
    specialty_norm = _normalize_text(specialty_name)

    # ── 1. Filtrar por especialidad ──────────────────────────────────────────
    def _exact_match(text: str) -> bool:
        return specialty_norm in _normalize_text(text)

    def _fuzzy_match(text: str) -> bool:
        norm = _normalize_text(text)
        # Comparar contra cada token del campo especialidades
        tokens = norm.split()
        for token in tokens:
            if SequenceMatcher(None, specialty_norm, token).ratio() >= 0.55:
                return True
        # También comparar contra la cadena completa
        return SequenceMatcher(None, specialty_norm, norm).ratio() >= 0.62

    specialty_candidates: pd.DataFrame = pd.DataFrame()
    if specialty_norm and "especialidades_texto" in hospitals.columns:
        spec_series = hospitals["especialidades_texto"].astype(str)
        # Paso 1: coincidencia exacta por substring
        specialty_candidates = hospitals[spec_series.apply(_exact_match)]
        # Paso 2: fuzzy si no hay coincidencia exacta
        if specialty_candidates.empty:
            specialty_candidates = hospitals[spec_series.apply(_fuzzy_match)]

    search_space = specialty_candidates if not specialty_candidates.empty else hospitals

    if graph is None or "lat" not in search_space.columns or "lon" not in search_space.columns:
        return search_space.iloc[0] if not search_space.empty else hospitals.iloc[0]

    # ── 2. Calcular costes desde el SOS (o desde las bases SAMUR como fallback) ──
    if sos_lat is not None and sos_lon is not None:
        try:
            sos_node = int(ox.distance.nearest_nodes(graph, X=float(sos_lon), Y=float(sos_lat)))
            origin_nodes = [sos_node]
        except Exception:
            origin_nodes = []
    else:
        origin_nodes = []

    if not origin_nodes:
        # Fallback: usar bases SAMUR como origen múltiple
        bases = load_samur_bases_for_dispatch(graph)
        origin_nodes = [int(b["nodo_red"]) for b in bases if "nodo_red" in b]

    travel_costs = _multi_source_min_costs(graph, origin_nodes)

    # ── 3. Seleccionar el hospital con menor coste de viaje ──────────────────
    best_idx = None
    best_cost = float("inf")
    for idx, row in search_space.iterrows():
        target_node = row.get("nodo_red")
        if pd.isna(target_node):
            lat, lon = row.get("lat"), row.get("lon")
            if pd.isna(lat) or pd.isna(lon):
                continue
            try:
                target_node = int(ox.distance.nearest_nodes(graph, X=float(lon), Y=float(lat)))
            except Exception:
                continue

        cost = float(travel_costs.get(int(target_node), float("inf")))
        if cost < best_cost:
            best_cost = cost
            best_idx = idx

    if best_idx is not None:
        return search_space.loc[best_idx]

    return search_space.iloc[0] if not search_space.empty else hospitals.iloc[0]


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

    with st.container():
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

        left_in, right_in = st.columns([1.15, 1], gap="medium")
        with left_in:
            selected_audio = st.selectbox("Sample de audio", audio_options)
            if selected_audio != "-- seleccionar --":
                st.audio(str(PROJECT_ROOT / selected_audio))

            a1, a2 = st.columns(2)
            with a1:
                stt_model = st.selectbox("Modelo STT", ["tiny", "base", "small", "medium", "large-v3"], index=1)
            with a2:
                stt_lang = st.selectbox("Idioma", ["es", "en", "auto"], index=0)

            run_transcribe = st.button("Transcribir", use_container_width=True)

        with right_in:
            llm_model = st.text_input(
                "Modelo LLM extractor",
                value=os.getenv("AMBULANCIA_LLM_MODEL", "llama3.1:8b-instruct"),
            )
            st.markdown(
                '<div class="soft" style="margin-top:6px;">Ajusta el modelo de extraccion antes de generar el vector clinico.</div>',
                unsafe_allow_html=True,
            )

        if run_transcribe:
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
            st.markdown(
                f"""
                <div class="kpi-grid">
                    <div class="kpi">
                        <div class="kpi-label">Urgencia</div>
                        <div class="kpi-main">{html.escape(str(pred['urgencia']['name']))}</div>
                        <div class="kpi-sub">Conf. {pred['urgencia']['confidence']:.2f}</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-label">Especialidad</div>
                        <div class="kpi-main">{html.escape(str(pred['especialidad']['name']))}</div>
                        <div class="kpi-sub">Conf. {pred['especialidad']['confidence']:.2f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            exp = st.session_state["explanation"]
            st.markdown(
                f"""
                <div class="exp-grid">
                    <div class="exp-card">
                        <div class="exp-title">Explicabilidad de urgencia</div>
                        <div class="exp-body">{html.escape(exp['urgencia_text'])}</div>
                    </div>
                    <div class="exp-card">
                        <div class="exp-title">Explicabilidad de especialidad</div>
                        <div class="exp-body">{html.escape(exp['especialidad_text'])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if exp["cautions"]:
                st.markdown("<div class='warn'><b>Precauciones de seguridad:</b><br>" + "<br>".join(exp["cautions"]) + "</div>", unsafe_allow_html=True)

            use_llm_polish = st.checkbox("Refinar explicacion con LLM (controlado)", value=False)
            if use_llm_polish:
                polished = optional_llm_refine(exp, llm_model)
                st.text_area("Resumen narrativo", value=polished, height=140)
        else:
            st.markdown(
                '<div class="soft">Aun no hay resultado. Al generar el vector apareceran aqui las explicaciones de urgencia y especialidad.</div>',
                unsafe_allow_html=True,
            )

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

        # Leer posición SOS publicada por el conductor (si está disponible)
        _shared = load_state()
        _sos_nav = _shared.get("navigation", {}).get("sos", {}) if isinstance(_shared.get("navigation"), dict) else {}
        _sos_lat = float(_sos_nav["lat"]) if _sos_nav.get("lat") is not None else None
        _sos_lon = float(_sos_nav["lon"]) if _sos_nav.get("lon") is not None else None

        chosen_hosp = select_hospital(hospitals, final_spec, sos_lat=_sos_lat, sos_lon=_sos_lon)
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

        c_send_1, c_send_2 = st.columns([1, 1], gap="small")
        with c_send_1:
            publish_primary = st.button("Mandar informacion al conductor", type="primary", use_container_width=True)
        with c_send_2:
            publish_secondary = st.button("Publicar decision y notificar conductor", use_container_width=True)

        if publish_primary or publish_secondary:
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
                    "especialidades": str(chosen_hosp.get("especialidades_texto", "")),
                    "perfiles": str(chosen_hosp.get("perfiles_atencion", "")),
                    "municipio": str(chosen_hosp.get("municipio", "")),
                    "centro_tipo": str(chosen_hosp.get("centro_tipo", "")),
                    "lat": float(chosen_hosp.get("lat", 0.0) or 0.0),
                    "lon": float(chosen_hosp.get("lon", 0.0) or 0.0),
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