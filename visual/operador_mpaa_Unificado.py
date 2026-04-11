from __future__ import annotations

import json
import importlib
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import osmnx as ox
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from visual.dispatch_shared import load_state

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "madrid_grafo.graphml"
PROCESSED_HOSPITALES_PATH = (
    PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"
)
SAMUR_BASES_PATH = PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "bases_samur_madrid.csv"


def load_operator_service_embedded() -> Any:
    """Importa operator_service sin aplicar efectos globales de estilo/config."""

    original_set_page_config = st.set_page_config
    try:
        st.set_page_config = lambda *args, **kwargs: None  # type: ignore[assignment]
        if "visual.operator_service" in sys.modules:
            module = importlib.reload(sys.modules["visual.operator_service"])
        else:
            module = importlib.import_module("visual.operator_service")
    finally:
        st.set_page_config = original_set_page_config

    return module


def render_operator_service_embedded(op_module: Any) -> None:
    """Renderiza op.main() sin hero duplicado dentro del panel unificado."""

    original_markdown = st.markdown
    hero_hidden = False

    def _patched_markdown(body: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal hero_hidden
        if isinstance(body, str) and "<div class=\"hero\">" in body and not hero_hidden:
            hero_hidden = True
            return None
        return original_markdown(body, *args, **kwargs)

    st.markdown = _patched_markdown  # type: ignore[assignment]
    try:
        op_module.main()
    finally:
        st.markdown = original_markdown


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def render_clinical_visual_output() -> None:
    text = str(st.session_state.get("operator_text", "") or "").strip()
    extraction = st.session_state.get("extraction", {})
    prediction = st.session_state.get("prediction", {})
    explanation = st.session_state.get("explanation", {})
    feature_map = st.session_state.get("feature_map", {})

    has_prediction = isinstance(prediction, dict) and bool(prediction)
    has_text = bool(text)

    words = len([w for w in text.split() if w.strip()])
    chars = len(text)

    urg = prediction.get("urgencia", {}) if has_prediction else {}
    spec = prediction.get("especialidad", {}) if has_prediction else {}

    urg_conf = max(0.0, min(1.0, _safe_float(urg.get("confidence", 0.0))))
    spec_conf = max(0.0, min(1.0, _safe_float(spec.get("confidence", 0.0))))

    positives = 0
    negatives = 0
    vitals_total = 0
    vitals_reported = 0
    if isinstance(feature_map, dict) and feature_map:
        for key, value in feature_map.items():
            iv = int(_safe_float(value, 0.0))
            if key.endswith("_presente") and iv == 1:
                positives += 1
            if key.endswith("_negado") and iv == 1:
                negatives += 1

        vital_keys = [
            "edad",
            "frecuencia_cardiaca",
            "presion_sistolica",
            "presion_diastolica",
            "saturacion_oxigeno",
            "frecuencia_respiratoria",
            "temperatura",
            "glucemia",
            "escala_glasgow",
        ]
        vitals_total = len(vital_keys)
        vitals_reported = sum(1 for k in vital_keys if _safe_float(feature_map.get(k, 0.0), 0.0) > 0)

    st.markdown('<div class="u-section"><span class="sec-icon">✦</span> Salida visual de transcripción y LLM</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1.05, 1.2], gap="large")
    with c_left:
        st.markdown(
            f"""
            <div class="v-card">
                <div class="v-title">Transcripción</div>
                <div class="v-kpis">
                    <div><span class="v-k-label">Estado</span><span class="v-k-val">{"Lista" if has_text else "Sin texto"}</span></div>
                    <div><span class="v-k-label">Palabras</span><span class="v-k-val">{words}</span></div>
                    <div><span class="v-k-label">Caracteres</span><span class="v-k-val">{chars}</span></div>
                </div>
                <div class="v-block">{text[:900] if has_text else 'No hay transcripción todavía. Selecciona audio y pulsa Transcribir.'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        provider = str(extraction.get("provider", "-") or "-")
        model = str(extraction.get("model", "-") or "-")
        fallback = "sí" if bool(extraction.get("fallback_used", False)) else "no"
        st.markdown(
            f"""
            <div class="v-card compact">
                <div class="v-title">Trazabilidad extracción</div>
                <div class="trace-row"><span>Proveedor</span><b>{provider}</b></div>
                <div class="trace-row"><span>Modelo</span><b>{model}</b></div>
                <div class="trace-row"><span>Fallback</span><b>{fallback}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_right:
        if has_prediction:
            urg_name = str(urg.get("name", "-") or "-")
            spec_name = str(spec.get("name", "-") or "-")
            st.markdown(
                f"""
                <div class="v-card">
                    <div class="v-title">Salida LLM + IA</div>
                    <div class="pred-head">
                        <span class="pred-badge">Urgencia: {urg_name}</span>
                        <span class="pred-badge">Especialidad: {spec_name}</span>
                    </div>
                    <div class="bar-wrap"><span>Confianza urgencia</span><div class="bar"><i style="width:{urg_conf * 100:.1f}%"></i></div><b>{urg_conf * 100:.1f}%</b></div>
                    <div class="bar-wrap"><span>Confianza especialidad</span><div class="bar"><i style="width:{spec_conf * 100:.1f}%"></i></div><b>{spec_conf * 100:.1f}%</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            evidence = explanation.get("evidence", []) if isinstance(explanation, dict) else []
            cautions = explanation.get("cautions", []) if isinstance(explanation, dict) else []
            chips = "".join(f"<span class='chip good'>{str(item)}</span>" for item in list(evidence)[:6])
            warns = "".join(f"<span class='chip warn'>{str(item)}</span>" for item in list(cautions)[:4])
            st.markdown(
                f"""
                <div class="v-card compact">
                    <div class="v-title">Señales clínicas detectadas</div>
                    <div class="chip-wrap">{chips if chips else '<span class="chip">Sin evidencias destacadas</span>'}</div>
                    <div class="v-subtitle">Precauciones</div>
                    <div class="chip-wrap">{warns if warns else '<span class="chip">Sin alertas de seguridad</span>'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="v-card">
                    <div class="v-title">Salida LLM + IA</div>
                    <div class="v-block">Genera vector y predicción para ver aquí la salida visual de urgencia, especialidad, confidencias y señales clínicas.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        ratio = (vitals_reported / vitals_total) if vitals_total else 0.0
        st.markdown(
            f"""
            <div class="v-card compact">
                <div class="v-title">Vector clínico</div>
                <div class="trace-row"><span>Constantes reportadas</span><b>{vitals_reported}/{vitals_total}</b></div>
                <div class="trace-row"><span>Factores presentes</span><b>{positives}</b></div>
                <div class="trace-row"><span>Factores negados</span><b>{negatives}</b></div>
                <div class="bar-wrap"><span>Completitud de constantes</span><div class="bar"><i style="width:{ratio * 100:.1f}%"></i></div><b>{ratio * 100:.1f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

@st.cache_resource
def map_load_graph_with_traffic() -> nx.MultiDiGraph:
    graph = ox.load_graphml(GRAPH_PATH)
    for _, _, _, data in graph.edges(keys=True, data=True):
        tf = random.uniform(1.0, 3.0)
        data["traffic_factor"] = tf
        data["weighted_length"] = data.get("length", 1.0) * tf
    return graph


@st.cache_data
def map_load_hospitals(_graph: nx.MultiDiGraph) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_HOSPITALES_PATH, sep=";")
    if "centro_tipo" in df.columns:
        df = df[df["centro_tipo"].isin({"Hospital general", "Hospital especializado"})].copy()
    if "municipio" in df.columns:
        municipio_norm = df["municipio"].astype(str).str.lower().str.strip()
        df = df[~municipio_norm.isin({"alcorcon", "alcorcón"})].copy()
    for col in ["lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    df["nodo_red"] = df.apply(
        lambda row: int(ox.distance.nearest_nodes(_graph, X=float(row["lon"]), Y=float(row["lat"]))), axis=1
    )
    if "nombre" not in df.columns:
        df["nombre"] = "Hospital"
    for col in ["direccion_completa", "especialidades_texto", "perfiles_atencion", "centro_tipo", "municipio", "telefono", "email"]:
        if col not in df.columns:
            df[col] = ""
    return df


@st.cache_data
def map_load_samur_bases(_graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
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


def map_get_scenario_incidents() -> List[Dict[str, Any]]:
    return [
        {
            "tipo": "Atasco severo",
            "lat": 40.4215,
            "lon": -3.6590,
            "radio": 1400,
            "nivel": "alto",
            "detalle": "Retencion por hora punta",
        },
        {
            "tipo": "Obras",
            "lat": 40.3920,
            "lon": -3.6850,
            "radio": 900,
            "nivel": "medio",
            "detalle": "Carril cortado por mantenimiento",
        },
        {
            "tipo": "Evento deportivo",
            "lat": 40.4531,
            "lon": -3.6883,
            "radio": 1200,
            "nivel": "medio",
            "detalle": "Afluencia masiva en zona estadio",
        },
        {
            "tipo": "Lluvia intensa",
            "lat": 40.4080,
            "lon": -3.6750,
            "radio": 1000,
            "nivel": "bajo",
            "detalle": "Visibilidad reducida",
        },
    ]


def map_build_operational_scenario(
    graph: nx.MultiDiGraph,
    hospitals_df: pd.DataFrame,
    bases: List[Dict[str, Any]],
    shared_state: Dict[str, Any],
) -> Dict[str, Any]:
    destination = shared_state.get("destination", {})
    case_info = shared_state.get("case", {}) if isinstance(shared_state.get("case", {}), dict) else {}
    navigation = shared_state.get("navigation", {}) if isinstance(shared_state.get("navigation", {}), dict) else {}
    sos_shared = navigation.get("sos", {}) if isinstance(navigation.get("sos", {}), dict) else {}
    destination_id = str(destination.get("centro_id", "")).strip().upper()
    destination_name = str(destination.get("nombre", "")).strip().lower()
    specialty_name = str(case_info.get("especialidad", "")).strip()
    urgency_value = case_info.get("urgencia", "")

    def _normalize_text(value: Any) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()

    def _extract_level(value: Any) -> int | None:
        text = str(value).strip()
        try:
            return int(float(text))
        except Exception:
            for ch in text:
                if ch.isdigit():
                    return int(ch)
        return None

    specialty_candidates = pd.DataFrame()
    specialty_norm = _normalize_text(specialty_name)
    if specialty_norm and specialty_norm not in {"", "no definida", "no definida."} and "especialidades_texto" in hospitals_df.columns:
        specialty_candidates = hospitals_df[
            hospitals_df["especialidades_texto"].astype(str).apply(lambda text: specialty_norm in _normalize_text(text))
        ]

    candidate = pd.DataFrame()
    if destination_id:
        candidate = hospitals_df[hospitals_df["centro_id"].astype(str).str.strip().str.upper() == destination_id]
    if candidate.empty and destination_name:
        candidate = hospitals_df[
            hospitals_df["nombre"].astype(str).str.lower().str.strip().eq(destination_name)
            | hospitals_df["nombre"].astype(str).str.lower().str.contains(destination_name, regex=False)
        ]

    def _seed_for_row(row: pd.Series) -> int:
        cid = str(row.get("centro_id", ""))
        return sum(ord(ch) for ch in cid) + 17

    def _metrics_for_row(row: pd.Series) -> Dict[str, float]:
        seed = _seed_for_row(row)
        occ = float(35 + (seed * 7) % 63)
        wait = float(8 + (seed * 11) % 95)
        doctors = float(6 + (seed * 5) % 27)
        return {"occ": occ, "wait": wait, "doctors": doctors}

    def _best_base_and_travel(target_node: int) -> tuple[Dict[str, Any], float]:
        chosen_base = bases[0]
        best_cost = float("inf")
        for base in bases:
            for weight in ("weighted_length", "length"):
                try:
                    c = float(nx.shortest_path_length(graph, source=base["nodo_red"], target=target_node, weight=weight))
                    if c < best_cost:
                        best_cost = c
                        chosen_base = base
                except nx.NetworkXNoPath:
                    continue
        return chosen_base, best_cost

    def _multi_source_min_costs() -> Dict[int, float]:
        source_nodes = [int(base["nodo_red"]) for base in bases if "nodo_red" in base]
        if not source_nodes:
            return {}

        best_costs: Dict[int, float] = {}
        for weight in ("weighted_length", "length"):
            try:
                partial = nx.multi_source_dijkstra_path_length(graph, source_nodes, weight=weight)
            except Exception:
                continue

            for node, value in partial.items():
                v = float(value)
                if node not in best_costs or v < best_costs[node]:
                    best_costs[node] = v

        return best_costs

    if candidate.empty:
        # Regla de negocio:
        # - Priorizar hospital mas cercano con la especialidad indicada.
        # - Si no hay coincidencias de especialidad, usar hospital mas cercano sin filtro.
        best_idx = None
        best_cost = float("inf")
        origin_base = bases[0]
        travel_costs = _multi_source_min_costs()

        search_space = specialty_candidates if not specialty_candidates.empty else hospitals_df
        for idx, row in search_space.iterrows():
            target_node_i = int(row["nodo_red"])
            travel_i = float(travel_costs.get(target_node_i, float("inf")))
            if travel_i == float("inf"):
                continue

            if travel_i < best_cost:
                best_cost = travel_i
                best_idx = idx

        target_row = hospitals_df.loc[best_idx] if best_idx is not None else hospitals_df.iloc[0]
        if not specialty_candidates.empty:
            status = f"Destino recomendado por especialidad y cercanía: {specialty_name}"
        else:
            status = "Destino recomendado por cercanía (sin coincidencia de especialidad)"
    else:
        target_row = candidate.iloc[0]
        status = "Destino sincronizado con conductor"

    target_node = int(target_row["nodo_red"])

    sos_lat = sos_shared.get("lat")
    sos_lon = sos_shared.get("lon")
    has_sos_sync = isinstance(sos_lat, (int, float)) and isinstance(sos_lon, (int, float))

    if has_sos_sync:
        sos_node = int(ox.distance.nearest_nodes(graph, X=float(sos_lon), Y=float(sos_lat)))
        origin_base, _ = _best_base_and_travel(sos_node)
    else:
        sos_node = None
        origin_base, _ = _best_base_and_travel(target_node)

    route_nodes: List[int] | None = None
    route_graphs: List[nx.MultiDiGraph | nx.Graph] = [graph, graph.to_undirected()]
    route_source = int(sos_node) if sos_node is not None else int(origin_base["nodo_red"])
    for g in route_graphs:
        for weight in ("weighted_length", "length"):
            try:
                route_nodes = nx.shortest_path(g, source=route_source, target=target_node, weight=weight)
                break
            except nx.NetworkXNoPath:
                continue
        if route_nodes:
            break

    if route_nodes:
        route_coords = [[float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])] for n in route_nodes]
    else:
        if has_sos_sync:
            route_coords = [[float(sos_lat), float(sos_lon)], [float(target_row["lat"]), float(target_row["lon"])]]
        else:
            route_coords = [[float(origin_base["lat"]), float(origin_base["lon"])], [float(target_row["lat"]), float(target_row["lon"])]]

    final_point = [float(target_row["lat"]), float(target_row["lon"])]
    if route_coords[-1] != final_point:
        route_coords.append(final_point)

    if has_sos_sync and route_coords:
        first_point = [float(sos_lat), float(sos_lon)]
        if route_coords[0] != first_point:
            route_coords.insert(0, first_point)

    hospitals_view: List[Dict[str, Any]] = []
    for idx, row in hospitals_df.iterrows():
        m = _metrics_for_row(row)
        hospitals_view.append(
            {
                "centro_id": str(row.get("centro_id", idx)),
                "nombre": str(row.get("nombre", f"Hospital {row.get('centro_id', idx)}")),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "nodo_red": int(row["nodo_red"]),
                "direccion": str(row.get("direccion_completa", "") or "No disponible"),
                "especialidades": str(row.get("especialidades_texto", "") or "No disponible"),
                "perfiles": str(row.get("perfiles_atencion", "") or "No disponible"),
                "municipio": str(row.get("municipio", "") or "No disponible"),
                "centro_tipo": str(row.get("centro_tipo", "") or "No disponible"),
                "telefono": str(row.get("telefono", "") or "No disponible"),
                "email": str(row.get("email", "") or "No disponible"),
                "medicos_disponibles": int(m["doctors"]),
                "occ": int(m["occ"]),
                "wait": int(m["wait"]),
            }
        )

    return {
        "hospitales": hospitals_view,
        "gps_ida": route_coords,
        "destino": {
            "centro_id": str(target_row.get("centro_id", "")),
            "nombre": str(target_row.get("nombre", destination.get("nombre", "Hospital destino"))),
            "lat": float(target_row["lat"]),
            "lon": float(target_row["lon"]),
            "direccion": str(target_row.get("direccion_completa", destination.get("direccion", ""))),
        },
        "origen": origin_base,
        "sos": {
            "nombre": str(sos_shared.get("nombre", "SOS")),
            "lat": float(sos_lat) if has_sos_sync else float(route_coords[0][0]),
            "lon": float(sos_lon) if has_sos_sync else float(route_coords[0][1]),
            "sync": bool(has_sos_sync),
        },
        "status": status,
        "eta_min": int(destination.get("eta_min", max(6, round(len(route_coords) / 12))) or max(6, round(len(route_coords) / 12))),
        "alerts": [str(a) for a in shared_state.get("traffic_alerts", []) if str(a).strip()],
    }


def map_render_driver_map(
    scenario: Dict[str, Any],
    bases: List[Dict[str, Any]],
    incidents: List[Dict[str, Any]],
    hospitals_count: int,
) -> None:
    scenario_json = json.dumps(scenario)
    bases_json = json.dumps(bases)
    incidents_json = json.dumps(incidents)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@600;700&display=swap');
    html, body, #map {{ height:100%; width:100%; margin:0; padding:0; background:#080e18; overflow:hidden; }}
    @keyframes pulseRed  {{ 0%,100%{{box-shadow:0 0 0 0 rgba(255,71,87,0.8)}}  60%{{box-shadow:0 0 0 14px rgba(255,71,87,0)}} }}
    @keyframes glowTarget{{ 0%,100%{{box-shadow:0 0 6px 2px rgba(0,200,255,0.4)}} 50%{{box-shadow:0 0 22px 8px rgba(0,200,255,0.75)}} }}
    @keyframes pulseLive {{ 0%,100%{{opacity:1}} 50%{{opacity:0.5}} }}
    .sos-marker {{
        background:linear-gradient(135deg,#c0001a,#ff2240); border:2px solid rgba(255,255,255,0.9);
        border-radius:50%; color:white; font-family:'Rajdhani',sans-serif; font-weight:700;
        font-size:11px; text-align:center; line-height:28px; letter-spacing:0.06em;
        animation:pulseRed 1.4s infinite;
    }}
    .ambu-marker {{ background:#0d1624; border:2px solid #00c8ff; border-radius:6px; text-align:center; line-height:22px; font-size:15px; }}
    .amb-icon {{ font-size:30px; text-align:center; filter:drop-shadow(0 0 8px rgba(0,200,255,0.9)); z-index:1001 !important; }}
    .hosp-marker {{ background:#0d1624; border:2px solid; border-radius:8px; text-align:center; line-height:26px; font-size:15px; box-shadow:0 2px 8px rgba(0,0,0,0.6); transition:all 0.4s ease; }}
    .hosp-green  {{ border-color:#00e896; box-shadow:0 0 8px rgba(0,232,150,0.25); }}
    .hosp-orange {{ border-color:#f5a623; box-shadow:0 0 8px rgba(245,166,35,0.25); }}
    .hosp-red    {{ border-color:#ff4757; box-shadow:0 0 8px rgba(255,71,87,0.25); }}
    .hosp-target {{ border-color:#00c8ff !important; animation:glowTarget 1.5s infinite; z-index:900 !important; transform:scale(1.25); }}
    .custom-tip {{
        font-family:'JetBrains Mono',monospace; font-size:12px;
        border-radius:8px; border:1px solid rgba(0,200,255,0.25) !important;
        background:#0d1624 !important; color:#d6e8f5 !important;
        box-shadow:0 4px 16px rgba(0,0,0,0.6), 0 0 12px rgba(0,200,255,0.12);
    }}
    .leaflet-tooltip.custom-tip {{ padding:8px 12px; }}
    .traffic-tip {{ font-weight:700; }}
    .progress-bg   {{ background:#1a2a3a; width:100%; height:6px; border-radius:3px; margin-top:5px; overflow:hidden; }}
    .progress-fill {{ height:100%; border-radius:3px; transition:width 0.5s ease; }}
    .leaflet-popup-content-wrapper {{
        background:#0d1624 !important; color:#d6e8f5 !important;
        border:1px solid rgba(0,200,255,0.25) !important; border-radius:12px !important;
        box-shadow:0 8px 32px rgba(0,0,0,0.7), 0 0 16px rgba(0,200,255,0.10) !important;
    }}
    .leaflet-popup-tip {{ background:#0d1624 !important; }}
    .leaflet-popup-close-button {{ color:#4d6a85 !important; }}
    .hospital-popup {{ min-width:280px; max-width:340px; font-family:'JetBrains Mono',monospace; }}
    .h-title {{ font-family:'Rajdhani',sans-serif; font-size:15px; font-weight:700; color:#fff; margin-bottom:6px; }}
    .h-meta  {{ color:#4d6a85; font-size:11px; margin-bottom:8px; letter-spacing:0.06em; }}
    .h-row   {{ margin:6px 0; font-size:11px; color:rgba(214,232,245,0.75); }}
    .h-label {{ font-weight:700; color:#00c8ff; display:block; margin-bottom:2px; }}
    .h-box   {{ margin-top:4px; padding:8px; border-radius:6px; background:rgba(0,200,255,0.04); border:1px solid rgba(0,200,255,0.12); max-height:80px; overflow:auto; font-size:10px; color:#4d6a85; }}
    #hud {{
        position:absolute; top:14px; right:14px; z-index:1200;
        background:rgba(8,14,24,0.94); border:1px solid rgba(0,200,255,0.22);
        border-radius:12px; padding:14px 16px; width:300px;
        font-family:'JetBrains Mono',monospace;
        box-shadow:0 0 24px rgba(0,200,255,0.12), 0 8px 32px rgba(0,0,0,0.7);
        backdrop-filter:blur(8px);
    }}
    #hud::before {{
        content:''; position:absolute; inset:0; border-radius:12px;
        background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,200,255,0.015) 3px,rgba(0,200,255,0.015) 4px);
        pointer-events:none;
    }}
    .hud-title {{
        font-family:'Rajdhani',sans-serif; font-weight:700; font-size:0.95rem;
        letter-spacing:0.1em; text-transform:uppercase; color:#fff;
        margin-bottom:10px; border-bottom:1px solid rgba(0,200,255,0.15); padding-bottom:8px;
        display:flex; justify-content:space-between; align-items:center;
    }}
    .live-badge {{
        font-size:0.6rem; color:#00e896; border:1px solid rgba(0,232,150,0.35);
        padding:2px 6px; border-radius:3px; letter-spacing:0.1em;
        animation:pulseLive 2s infinite;
    }}
    .hud-row {{ display:flex; justify-content:space-between; align-items:center; margin:6px 0; font-size:0.72rem; }}
    .hud-label {{ color:rgba(100,160,200,0.7); letter-spacing:0.08em; text-transform:uppercase; }}
    .hud-val   {{ color:#00c8ff; font-weight:700; font-size:0.8rem; }}
    .hud-sep   {{ border:none; border-top:1px solid rgba(0,200,255,0.10); margin:8px 0; }}
    .alerts-title {{ color:rgba(245,166,35,0.8); font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px; }}
    .alerts {{ font-size:0.68rem; max-height:80px; overflow:auto; color:rgba(214,232,245,0.65); }}
    .alerts li {{ margin:3px 0; list-style:none; padding-left:0; }}
    .alerts li::before {{ content:'⚠ '; color:#f5a623; }}
    #map-attrib {{
        position:absolute; left:12px; bottom:10px; z-index:1200;
        font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.04em;
        color:rgba(214,232,245,0.48); background:rgba(8,14,24,0.52);
        border:1px solid rgba(0,200,255,0.10); border-radius:6px;
        padding:3px 8px; pointer-events:none;
    }}
  </style>
</head>
<body>
  <div id=\"hud\">
    <div class=\"hud-title\">Operador AmbulancIA <span class=\"live-badge\">EN VIVO</span></div>
    <div class=\"hud-row\"><span class=\"hud-label\">Destino</span><b class=\"hud-val\" id=\"kDestino\">—</b></div>
    <div class=\"hud-row\"><span class=\"hud-label\">ETA</span><b class=\"hud-val\" id=\"kEta\">—</b></div>
    <div class=\"hud-row\"><span class=\"hud-label\">Distancia</span><b class=\"hud-val\" id=\"kDist\">—</b></div>
    <div class=\"hud-row\"><span class=\"hud-label\">Bases SAMUR</span><b class=\"hud-val\" id=\"kBases\">0</b></div>
    <hr class=\"hud-sep\">
    <div class=\"alerts-title\">Alertas de tráfico</div>
    <ul class=\"alerts\" id=\"alerts\"></ul>
  </div>
    <div id=\"map-attrib\">Based on real data</div>
    <div id=\"map\"></div>
  <script>
    const madridBounds = L.latLngBounds([[40.31, -3.88], [40.54, -3.56]]);
    const map = L.map('map', {{ preferCanvas: true }}).setView([40.4168, -3.7038], 13.8);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '© CartoDB' }}).addTo(map);
    map.setMaxBounds(madridBounds.pad(0.15));
    map.fitBounds(madridBounds.pad(0.06));
    const scenario  = {scenario_json};
    const bases     = {bases_json};
    const incidents = {incidents_json};
    document.getElementById('kBases').textContent = String(bases.length);
    const incidentColors = {{ alto:'#ff4757', medio:'#f5a623', bajo:'#00c8ff' }};
    incidents.forEach((inc) => {{
        const color = incidentColors[inc.nivel] || '#888';
        L.circle([inc.lat, inc.lon], {{ radius:inc.radio, color, fillColor:color, fillOpacity:0.10, weight:1, opacity:0.30 }})
          .bindTooltip(`🚥 <b>${{inc.tipo}}</b><br>${{inc.detalle}}`, {{ direction:'center', className:'custom-tip traffic-tip' }})
          .addTo(map);
    }});
    const alertsEl = document.getElementById('alerts');
    if ((scenario.alerts || []).length) {{
        scenario.alerts.forEach((a) => {{ const li = document.createElement('li'); li.textContent = a; alertsEl.appendChild(li); }});
    }} else {{
        const li = document.createElement('li'); li.textContent = '✓ Sin alertas activas';
        li.style.color = 'rgba(0,232,150,0.7)'; li.style.listStyle = 'none'; alertsEl.appendChild(li);
    }}
    bases.forEach((b) => {{
        const icon = L.divIcon({{ className:'ambu-marker', html:'🩺', iconSize:[26,26], iconAnchor:[13,13] }});
        L.marker([b.lat, b.lon], {{ icon }}).bindTooltip(`<b style="color:#00c8ff">BASE SVB</b><br>${{b.nombre}}`, {{ direction:'top', className:'custom-tip' }}).addTo(map);
    }});
    const route = scenario.gps_ida || [];
    const sosPoint = scenario.sos || null;
    const sosIcon = L.divIcon({{ className:'sos-marker', html:'SOS', iconSize:[28,28], iconAnchor:[14,14] }});
    if (sosPoint && Number.isFinite(Number(sosPoint.lat)) && Number.isFinite(Number(sosPoint.lon))) {{
        L.marker([Number(sosPoint.lat), Number(sosPoint.lon)], {{ icon: sosIcon }}).addTo(map)
            .bindTooltip('<b style="color:#ff4757">SOS sincronizado</b><br>' + (sosPoint.nombre || 'Punto operativo'), {{ direction:'top', className:'custom-tip' }});
    }} else if (route.length) {{
        L.marker(route[0], {{ icon: sosIcon }}).addTo(map).bindTooltip('<b style="color:#ff4757">PUNTO OPERATIVO</b>', {{ direction:'top', className:'custom-tip' }});
    }}
    const ambulancePoint = route.length ? route[0] : (sosPoint && Number.isFinite(Number(sosPoint.lat)) && Number.isFinite(Number(sosPoint.lon)) ? [Number(sosPoint.lat), Number(sosPoint.lon)] : [40.4168, -3.7038]);
    const ambulanceZoom = route.length >= 2 ? 15.0 : 15.2;
    map.setView(ambulancePoint, ambulanceZoom);
    window.setTimeout(() => {{
        try {{
            map.invalidateSize();
            map.setView(ambulancePoint, ambulanceZoom);
        }} catch (e) {{}}
    }}, 120);
    const hospitalMarkers = {{}};
    scenario.hospitales.forEach((h) => {{
        const colorClass = (h.occ > 85 ? 'hosp-red' : (h.occ > 50 ? 'hosp-orange' : 'hosp-green'));
        const barColor   = h.occ > 85 ? '#ff4757' : (h.occ > 50 ? '#f5a623' : '#00e896');
        const icon = L.divIcon({{ className:'hosp-marker ' + colorClass, html:'🏥', iconSize:[30,30], iconAnchor:[15,15] }});
        const tipHTML = `<div style="min-width:180px"><div style="font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;color:#fff;margin-bottom:4px;">${{h.nombre}}</div>🛏 Ocupación: <b style="color:${{barColor}}">${{h.occ}}%</b><div class="progress-bg"><div class="progress-fill" style="width:${{h.occ}}%;background:linear-gradient(90deg,${{barColor}},${{barColor}}88);"></div></div><div style="margin-top:5px;">⏱ Espera: <b style="color:#f5a623">${{h.wait}} min</b></div></div>`;
        const popupHTML = `<div class="hospital-popup"><div class="h-title">🏥 ${{h.nombre}}</div><div class="h-meta">${{h.centro_id}} · ${{h.centro_tipo}} · ${{h.municipio}}</div><div class="h-row"><span class="h-label">Dirección</span>${{h.direccion}}</div><div class="h-row"><span class="h-label">Contacto</span>${{h.telefono}} · ${{h.email}}</div><div class="h-row"><span class="h-label">Médicos disponibles</span><b style="color:#00c8ff">${{h.medicos_disponibles}}</b></div><div class="h-row"><span class="h-label">Estado operativo</span>Ocupación <b style="color:${{barColor}}">${{h.occ}}%</b> · Espera <b style="color:#f5a623">${{h.wait}} min</b></div><div class="progress-bg"><div class="progress-fill" style="width:${{h.occ}}%;background:linear-gradient(90deg,${{barColor}},${{barColor}}88);"></div></div><div class="h-row"><span class="h-label">Perfiles</span><div class="h-box">${{h.perfiles}}</div></div><div class="h-row"><span class="h-label">Especialidades</span><div class="h-box">${{h.especialidades}}</div></div></div>`;
        const marker = L.marker([h.lat, h.lon], {{ icon }}).addTo(map)
            .bindTooltip(tipHTML, {{ direction:'top', className:'custom-tip' }})
            .bindPopup(popupHTML, {{ maxWidth:360 }});
        hospitalMarkers[h.nombre] = marker;
    }});
    function densificar(ruta, maxDist) {{
        const nueva = [];
        for (let i = 0; i < ruta.length - 1; i++) {{
            const p1 = ruta[i], p2 = ruta[i+1];
            const dist = Math.sqrt(Math.pow(p2[0]-p1[0],2)+Math.pow(p2[1]-p1[1],2));
            const pasos = Math.max(1, Math.ceil(dist/maxDist));
            for (let j = 0; j < pasos; j++) {{ nueva.push([p1[0]+(p2[0]-p1[0])*(j/pasos), p1[1]+(p2[1]-p1[1])*(j/pasos)]); }}
        }}
        nueva.push(ruta[ruta.length-1]); return nueva;
    }}
    function haversineM(a, b) {{
        const R=6371000, rad=x=>x*Math.PI/180;
        const dLat=rad(b[0]-a[0]), dLon=rad(b[1]-a[1]);
        const h=Math.sin(dLat/2)**2+Math.cos(rad(a[0]))*Math.cos(rad(b[0]))*Math.sin(dLon/2)**2;
        return 2*R*Math.atan2(Math.sqrt(h),Math.sqrt(1-h));
    }}
    function remainingDist(idx, ruta) {{ let d=0; for (let k=idx; k<ruta.length-1; k++) d+=haversineM(ruta[k],ruta[k+1]); return d; }}
    if (route.length >= 2) {{
        const idaSuave = densificar(route, 0.00018);
        const routeLine = L.polyline(route, {{ color:'#00c8ff', weight:3, opacity:0.55, dashArray:'6,9' }}).addTo(map);
        const doneLine  = L.polyline([idaSuave[0]], {{ color:'#00e896', weight:5, opacity:0.9 }}).addTo(map);
        const ambIcon   = L.divIcon({{ className:'amb-icon', html:'🚑', iconSize:[32,32], iconAnchor:[16,16] }});
        const amb       = L.marker(idaSuave[0], {{ icon:ambIcon }}).addTo(map);
        if (hospitalMarkers[scenario.destino.nombre]) {{
            hospitalMarkers[scenario.destino.nombre].setIcon(L.divIcon({{ className:'hosp-marker hosp-target', html:'🏥🏁', iconSize:[36,36], iconAnchor:[18,18] }}));
        }}
        const routeBounds = routeLine.getBounds();
        if (routeBounds.isValid()) {{
            const diagKm = map.distance(routeBounds.getSouthWest(), routeBounds.getNorthEast()) / 1000;
            if (diagKm <= 12) {{
                map.fitBounds(routeBounds.pad(0.08), {{ maxZoom: 15.6 }});
            }} else {{
                map.setView(ambulancePoint, ambulanceZoom);
            }}
        }}
        const etaTotal=Math.max(2,Number(scenario.eta_min||8)), pasoAnimacion=4, tickMs=50;
        let idx=0;
        function animar() {{
            if (idx >= idaSuave.length) {{ document.getElementById('kEta').textContent='0 min'; document.getElementById('kDist').textContent='0 m'; return; }}
            amb.setLatLng(idaSuave[idx]); doneLine.setLatLngs(idaSuave.slice(0,idx+1));
            const progress=idx/Math.max(1,idaSuave.length-1);
            const etaNow=Math.max(0,Math.round(etaTotal*(1-progress)));
            const rem=remainingDist(idx,idaSuave);
            document.getElementById('kEta').textContent=etaNow+' min';
            document.getElementById('kDist').textContent=rem>=1000?(rem/1000).toFixed(1)+' km':Math.round(rem)+' m';
            idx+=pasoAnimacion; setTimeout(animar,tickMs);
        }}
        setTimeout(animar,120);
    }}
    document.getElementById('kDestino').textContent = scenario.destino.nombre || '—';
    document.getElementById('kEta').textContent     = `${{scenario.eta_min}} min`;
  </script>
</body>
</html>
"""
    components.html(html, height=760)


def map_render_fallback_map(scenario: Dict[str, Any], bases: List[Dict[str, Any]]) -> None:
    points: List[Dict[str, float]] = []

    for b in bases:
        points.append({"lat": float(b["lat"]), "lon": float(b["lon"])})

    for h in scenario.get("hospitales", []):
        points.append({"lat": float(h["lat"]), "lon": float(h["lon"])})

    route = scenario.get("gps_ida", [])
    if route:
        step = max(1, len(route) // 90)
        for lat, lon in route[::step]:
            points.append({"lat": float(lat), "lon": float(lon)})

    destino = scenario.get("destino", {})
    if destino and destino.get("lat") is not None and destino.get("lon") is not None:
        points.append({"lat": float(destino["lat"]), "lon": float(destino["lon"])})

    if not points:
        st.info("Mapa de respaldo no disponible: no hay coordenadas para mostrar.")
        return

    st.caption("Mapa de respaldo (si el interactivo no carga)")
    st.map(pd.DataFrame(points), latitude="lat", longitude="lon", zoom=13)


def main() -> None:
    # ── PAGE CONFIG ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="AmbulancIA · Operador",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── GLOBAL STYLES ────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;600&display=swap');

            :root {
                --bg-base:    #080e18;
                --bg-card:    #0d1624;
                --bg-card2:   #111c2e;
                --border:     rgba(0, 200, 255, 0.10);
                --border-hot: rgba(0, 200, 255, 0.30);
                --cyan:       #00c8ff;
                --amber:      #f5a623;
                --green:      #00e896;
                --red:        #ff4757;
                --text:       #d6e8f5;
                --muted:      #4d6a85;
                --glow-cyan:  0 0 18px rgba(0,200,255,0.22);
            }

            html, body, [data-testid="stAppViewContainer"] {
                background-color: var(--bg-base) !important;
                color: var(--text) !important;
            }
            [data-testid="stAppViewContainer"]::before {
                content: '';
                position: fixed; inset: 0;
                background:
                    radial-gradient(ellipse 65% 45% at 5%   0%,  rgba(0,200,255,0.07) 0%, transparent 55%),
                    radial-gradient(ellipse 50% 40% at 95% 100%, rgba(0,232,150,0.04) 0%, transparent 50%),
                    radial-gradient(ellipse 40% 35% at 50%  50%, rgba(10,30,60,0.5)   0%, transparent 70%);
                pointer-events: none; z-index: 0;
            }
            #MainMenu, header, footer { visibility: hidden; }
            .block-container {
                padding-top: 0.8rem !important;
                padding-bottom: 1.5rem !important;
                padding-left: 1.4rem !important;
                padding-right: 1.4rem !important;
                max-width: 100% !important;
                position: relative; z-index: 1;
            }

            /* ── HERO ── */
            .u-hero {
                margin: 0 0 16px 0;
                border-radius: 16px;
                padding: 22px 28px;
                background: linear-gradient(118deg, #070f1d 0%, #0b1f3a 45%, #061a15 100%);
                border: 1px solid var(--border-hot);
                box-shadow: var(--glow-cyan), inset 0 1px 0 rgba(0,200,255,0.08);
                position: relative; overflow: hidden;
            }
            .u-hero::before {
                content: 'OPERADOR';
                position: absolute; right: 28px; top: 50%;
                transform: translateY(-50%);
                font-family: 'Rajdhani', sans-serif;
                font-size: 5.5rem; font-weight: 700;
                color: rgba(0,200,255,0.035);
                letter-spacing: 0.18em;
                pointer-events: none; user-select: none;
                white-space: nowrap;
            }
            .u-hero h2 {
                margin: 0;
                font-family: 'Rajdhani', sans-serif;
                font-size: 2rem; font-weight: 700;
                letter-spacing: 0.07em; text-transform: uppercase;
                color: #fff; line-height: 1.1;
            }
            .u-hero h2 span { color: var(--cyan); }
            .u-hero p {
                margin: 7px 0 0;
                font-family: 'DM Sans', sans-serif;
                font-size: 0.9rem; color: var(--muted);
                letter-spacing: 0.02em;
                max-width: 74ch;
            }
            .live-dot {
                display: inline-block; width: 8px; height: 8px;
                border-radius: 50%; background: var(--green);
                margin-right: 6px; vertical-align: middle;
                animation: pulseDot 1.6s ease-in-out infinite;
            }
            @keyframes pulseDot {
                0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(0,232,150,0.6); }
                50%      { opacity:0.7; box-shadow:0 0 0 6px rgba(0,232,150,0); }
            }
            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                margin-top: 12px;
                padding: 4px 10px;
                border-radius: 999px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.66rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: var(--green);
                border: 1px solid rgba(0,232,150,0.3);
                background: rgba(0,232,150,0.08);
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: rgba(8,14,24,0.7);
                border: 1px solid var(--border);
                border-radius: 11px;
                padding: 6px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px;
                color: var(--muted);
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                border: 1px solid transparent;
            }
            .stTabs [aria-selected="true"] {
                background: rgba(0,200,255,0.10) !important;
                color: var(--cyan) !important;
                border-color: rgba(0,200,255,0.28) !important;
            }

            /* ── SECTION HEADERS ── */
            .u-section {
                font-family: 'Rajdhani', sans-serif;
                font-size: 0.68rem; font-weight: 700;
                color: var(--muted);
                letter-spacing: 0.22em; text-transform: uppercase;
                margin: 1.4rem 0 0.7rem 0;
                display: flex; align-items: center; gap: 10px;
            }
            .u-section .sec-icon {
                color: var(--cyan); font-size: 0.8rem; opacity: 0.8;
            }
            .u-section::after {
                content: ''; flex: 1;
                height: 1px;
                background: linear-gradient(90deg, var(--border-hot), transparent);
            }

            .v-card {
                background: linear-gradient(165deg, rgba(13,22,36,0.95), rgba(17,28,46,0.95));
                border: 1px solid var(--border);
                border-radius: 13px;
                padding: 12px 14px;
                box-shadow: 0 10px 24px rgba(0,0,0,0.30);
                margin-bottom: 10px;
            }
            .v-card.compact { padding: 10px 12px; }
            .v-title {
                font-family: 'Rajdhani', sans-serif;
                font-size: 0.95rem;
                color: #fff;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 8px;
            }
            .v-subtitle {
                margin: 8px 0 5px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.63rem;
                color: var(--muted);
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .v-kpis {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 8px;
                margin-bottom: 8px;
            }
            .v-kpis > div {
                background: rgba(0,200,255,0.06);
                border: 1px solid rgba(0,200,255,0.16);
                border-radius: 8px;
                padding: 7px 8px;
            }
            .v-k-label {
                display: block;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.62rem;
                color: var(--muted);
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .v-k-val {
                display: block;
                font-family: 'Rajdhani', sans-serif;
                font-size: 1rem;
                color: var(--cyan);
                font-weight: 700;
                margin-top: 2px;
            }
            .v-block {
                border: 1px solid rgba(0,200,255,0.16);
                border-radius: 9px;
                background: rgba(0,200,255,0.04);
                color: rgba(214,232,245,0.9);
                padding: 10px 11px;
                min-height: 85px;
                max-height: 180px;
                overflow: auto;
                font-size: 0.82rem;
                line-height: 1.45;
                white-space: pre-wrap;
            }
            .pred-head {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                margin-bottom: 8px;
            }
            .pred-badge {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 4px 9px;
                font-size: 0.72rem;
                color: var(--cyan);
                border: 1px solid rgba(0,200,255,0.2);
                background: rgba(0,200,255,0.08);
            }
            .bar-wrap {
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 7px 10px;
                align-items: center;
                margin-top: 7px;
            }
            .bar-wrap > span {
                grid-column: 1 / -1;
                font-size: 0.74rem;
                color: var(--muted);
            }
            .bar-wrap > b {
                color: var(--cyan);
                font-size: 0.78rem;
                font-family: 'JetBrains Mono', monospace;
            }
            .bar {
                height: 8px;
                border-radius: 999px;
                background: rgba(255,255,255,0.06);
                overflow: hidden;
                border: 1px solid rgba(0,200,255,0.14);
            }
            .bar i {
                display: block;
                height: 100%;
                background: linear-gradient(90deg, rgba(0,200,255,0.65), rgba(0,232,150,0.75));
            }
            .trace-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 4px 0;
                font-size: 0.78rem;
                color: rgba(214,232,245,0.84);
            }
            .trace-row b {
                color: var(--cyan);
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.76rem;
            }
            .chip-wrap {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
            }
            .chip {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 3px 8px;
                font-size: 0.7rem;
                color: rgba(214,232,245,0.88);
                border: 1px solid rgba(0,200,255,0.2);
                background: rgba(0,200,255,0.08);
            }
            .chip.good {
                border-color: rgba(0,232,150,0.25);
                background: rgba(0,232,150,0.08);
            }
            .chip.warn {
                border-color: rgba(245,166,35,0.32);
                background: rgba(245,166,35,0.10);
            }

            /* ── METRIC CARDS (custom HTML, not st.metric) ── */
            .metric-strip {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                margin: 0 0 4px 0;
            }
            .metric-card {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 16px 18px;
                position: relative; overflow: hidden;
                animation: fadeSlideUp 0.4s ease both;
                transition: border-color 0.25s, box-shadow 0.25s;
                cursor: default;
            }
            .metric-card::before {
                content: '';
                position: absolute; top: 0; left: 0; right: 0; height: 2px;
                background: var(--accent-color, var(--cyan));
                opacity: 0.6;
            }
            .metric-card:hover {
                border-color: var(--border-hot);
                box-shadow: var(--glow-cyan);
            }
            .metric-card .mc-icon {
                font-size: 1.4rem; margin-bottom: 10px; display: block;
                filter: drop-shadow(0 0 6px rgba(0,200,255,0.5));
            }
            .metric-card .mc-label {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.62rem; color: var(--muted);
                letter-spacing: 0.14em; text-transform: uppercase;
                margin-bottom: 6px; display: block;
            }
            .metric-card .mc-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.9rem; font-weight: 700;
                color: var(--accent-color, var(--cyan));
                line-height: 1; display: block;
            }
            .metric-card .mc-sub {
                font-family: 'DM Sans', sans-serif;
                font-size: 0.72rem; color: var(--muted);
                margin-top: 5px; display: block;
            }
            @keyframes fadeSlideUp {
                from { opacity:0; transform:translateY(10px); }
                to   { opacity:1; transform:translateY(0); }
            }

            /* ── STATUS BAR ── */
            .status-bar {
                display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 10px 16px;
                margin: 0 0 14px 0;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                animation: fadeSlideUp 0.45s ease 0.1s both;
            }
            .status-bar .sb-label { color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }
            .status-bar .sb-val   { color: var(--cyan); font-weight: 700; }
            .status-bar .sb-sep   {
                width: 1px; height: 14px;
                background: var(--border); flex-shrink: 0;
            }
            .status-bar .sb-pill  {
                padding: 3px 9px; border-radius: 4px;
                border: 1px solid rgba(0,200,255,0.20);
                background: rgba(0,200,255,0.06);
                color: var(--cyan); letter-spacing: 0.08em;
            }
            .status-bar .sb-pill--green {
                border-color: rgba(0,232,150,0.25);
                background: rgba(0,232,150,0.06);
                color: var(--green);
            }
            .status-bar .sb-pill--amber {
                border-color: rgba(245,166,35,0.25);
                background: rgba(245,166,35,0.06);
                color: var(--amber);
            }

            /* ── INCIDENT CARDS ── */
            .incident-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                gap: 10px;
                margin: 0 0 4px 0;
            }
            .incident-card {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 14px 16px;
                position: relative; overflow: hidden;
                animation: fadeSlideUp 0.4s ease both;
                transition: border-color 0.2s, box-shadow 0.2s;
            }
            .incident-card::before {
                content: '';
                position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
                background: var(--inc-color, var(--amber));
            }
            .incident-card:hover {
                border-color: var(--inc-color, var(--amber));
                box-shadow: 0 0 14px rgba(245,166,35,0.12);
            }
            .incident-card .ic-type {
                font-family: 'Rajdhani', sans-serif;
                font-size: 0.95rem; font-weight: 700;
                color: #fff; letter-spacing: 0.04em;
                margin-bottom: 4px;
            }
            .incident-card .ic-detail {
                font-family: 'DM Sans', sans-serif;
                font-size: 0.78rem; color: var(--muted);
                margin-bottom: 8px;
            }
            .incident-card .ic-badge {
                display: inline-block;
                padding: 2px 8px; border-radius: 3px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase;
                background: rgba(0,0,0,0.3);
                border: 1px solid var(--inc-color, var(--amber));
                color: var(--inc-color, var(--amber));
            }
            .incident-card .ic-radio {
                float: right;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.65rem; color: var(--muted);
                margin-top: -22px;
            }

            /* ── st.metric OVERRIDE (fallback) ── */
            [data-testid="metric-container"] {
                background: var(--bg-card) !important;
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
                padding: 14px 16px !important;
            }
            [data-testid="stMetricLabel"] {
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 0.63rem !important; color: var(--muted) !important;
                letter-spacing: 0.12em !important; text-transform: uppercase !important;
            }
            [data-testid="stMetricValue"] {
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 1.6rem !important; font-weight: 700 !important;
                color: var(--cyan) !important;
            }

            /* ── DATAFRAME ── */
            [data-testid="stDataFrame"] {
                border: 1px solid var(--border) !important;
                border-radius: 10px !important; overflow: hidden;
            }
            [data-testid="stDataFrame"] thead th {
                background: var(--bg-card2) !important;
                color: var(--muted) !important;
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 0.68rem !important; letter-spacing: 0.1em !important;
                text-transform: uppercase !important;
            }
            [data-testid="stDataFrame"] tbody td {
                background: var(--bg-card) !important; color: var(--text) !important;
                font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important;
                border-color: var(--border) !important;
            }

            /* ── ALERTS / ERRORS / EXPANDERS ── */
            [data-testid="stAlert"] {
                background: rgba(255,71,87,0.07) !important;
                border: 1px solid rgba(255,71,87,0.25) !important;
                border-radius: 8px !important; color: #ff8a96 !important;
                font-family: 'DM Sans', sans-serif !important;
            }
            [data-testid="stExpander"] {
                background: var(--bg-card) !important;
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
            }
            hr { border-color: var(--border) !important; }

            /* ── MAP WRAP ── */
            .map-wrap {
                margin-top: 12px; padding: 4px;
                background: var(--bg-card);
                border: 1px solid var(--border-hot);
                border-radius: 16px;
                box-shadow: var(--glow-cyan), 0 24px 60px rgba(0,0,0,0.5);
            }
            iframe { height: 780px !important; border-radius: 13px !important; display: block; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── HERO ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="u-hero">
            <h2>Panel Operador <span>MPAA</span> Unificado</h2>
            <p>Gestión clínica · Derivación inteligente · Monitorización operativa en tiempo real · Madrid</p>
            <div class="hero-badge"><span class="live-dot"></span>Operación en vivo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_clinica, tab_operacion, tab_mapa = st.tabs([
        "Consola clínica",
        "Coordinación operativa",
        "Mapa y navegación",
    ])

    # ── CLINICAL EVALUATION ──────────────────────────────────────────────────
    with tab_clinica:
        st.markdown('<div class="u-section"><span class="sec-icon">⬡</span> Evaluación clínica</div>', unsafe_allow_html=True)
        op_module = load_operator_service_embedded()
        render_operator_service_embedded(op_module)

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    graph        = map_load_graph_with_traffic()
    hospitals_df = map_load_hospitals(graph)
    bases        = map_load_samur_bases(graph)
    incidents    = map_get_scenario_incidents()

    if hospitals_df.empty:
        st.error("No se pudieron cargar hospitales con coordenadas válidas.")
        return
    if not bases:
        st.error("No se pudieron cargar bases SAMUR válidas.")
        return

    shared_state = load_state()
    scenario     = map_build_operational_scenario(graph, hospitals_df, bases, shared_state)

    n_hospitales = int(hospitals_df["centro_id"].nunique()) if "centro_id" in hospitals_df.columns else len(hospitals_df)
    n_bases      = len(bases)
    n_incidents  = len(incidents)
    destino_nombre = scenario["destino"].get("nombre", "—")
    eta_val        = scenario.get("eta_min", "—")
    status_text    = scenario.get("status", "—")

    with tab_operacion:
        st.markdown('<div class="u-section"><span class="sec-icon">◈</span> Coordinación operativa</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="status-bar">
                <span class="sb-label">Destino activo</span>
                <span class="sb-val">{destino_nombre[:40]}</span>
                <span class="sb-sep"></span>
                <span class="sb-label">ETA estimado</span>
                <span class="sb-val">{eta_val} min</span>
                <span class="sb-sep"></span>
                <span class="sb-pill sb-pill--green"><span class="live-dot" style="width:6px;height:6px;margin-right:4px;"></span>SYNC ACTIVO</span>
                <span class="sb-pill sb-pill--amber">⚠ {n_incidents} INCIDENCIAS</span>
                <span class="sb-pill">{status_text[:45]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-strip">
                <div class="metric-card" style="--accent-color:var(--cyan); animation-delay:0.05s">
                    <span class="mc-icon">🏥</span>
                    <span class="mc-label">Hospitales disponibles</span>
                    <span class="mc-value">{n_hospitales}</span>
                    <span class="mc-sub">centros activos en red</span>
                </div>
                <div class="metric-card" style="--accent-color:var(--green); animation-delay:0.10s">
                    <span class="mc-icon">🚑</span>
                    <span class="mc-label">Bases SAMUR</span>
                    <span class="mc-value">{n_bases}</span>
                    <span class="mc-sub">bases operativas Madrid</span>
                </div>
                <div class="metric-card" style="--accent-color:var(--amber); animation-delay:0.15s">
                    <span class="mc-icon">⚠</span>
                    <span class="mc-label">Incidencias activas</span>
                    <span class="mc-value">{n_incidents}</span>
                    <span class="mc-sub">alertas de tráfico vigentes</span>
                </div>
                <div class="metric-card" style="--accent-color:var(--cyan); animation-delay:0.20s">
                    <span class="mc-icon">⏱</span>
                    <span class="mc-label">ETA al destino</span>
                    <span class="mc-value">{eta_val}</span>
                    <span class="mc-sub">minutos estimados</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="u-section"><span class="sec-icon">▲</span> Incidencias activas</div>', unsafe_allow_html=True)
        inc_color_map = {"alto": "#ff4757", "medio": "#f5a623", "bajo": "#00c8ff"}
        inc_icon_map = {"alto": "🔴", "medio": "🟡", "bajo": "🔵"}
        cards_html = '<div class="incident-grid">'
        for i, inc in enumerate(incidents):
            color = inc_color_map.get(inc["nivel"], "#4d6a85")
            icon = inc_icon_map.get(inc["nivel"], "⚪")
            delay = 0.05 + i * 0.06
            cards_html += f"""
                <div class="incident-card" style="--inc-color:{color}; animation-delay:{delay}s">
                    <div class="ic-type">{icon} {inc['tipo']}</div>
                    <div class="ic-detail">{inc['detalle']}</div>
                    <span class="ic-badge">{inc['nivel']}</span>
                    <span class="ic-radio">r={inc['radio']}m</span>
                </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<div class="u-section"><span class="sec-icon">▣</span> Top hospitales sugeridos</div>', unsafe_allow_html=True)
        ranking_df = pd.DataFrame(scenario.get("hospitales", []))
        if not ranking_df.empty:
            view_cols = [
                "nombre",
                "municipio",
                "centro_tipo",
                "medicos_disponibles",
                "occ",
                "wait",
            ]
            keep = [col for col in view_cols if col in ranking_df.columns]
            ranking_df = ranking_df.sort_values(["occ", "wait", "medicos_disponibles"], ascending=[True, True, False])
            st.dataframe(ranking_df[keep].head(12), use_container_width=True, hide_index=True)

    with tab_mapa:
        st.markdown('<div class="u-section"><span class="sec-icon">◉</span> Mapa operativo</div>', unsafe_allow_html=True)
        sync_col, toggle_col = st.columns([1, 1.6], gap="small")
        with sync_col:
            if st.button("Sync mapa operador", use_container_width=True):
                st.rerun()
        with toggle_col:
            auto_sync_map = st.toggle(
                "Auto-sync mapa (1.2s)",
                value=bool(st.session_state.get("operator_map_auto_sync", True)),
                key="operator_map_auto_sync",
                help="Sincroniza automaticamente el mapa del operador con el estado del conductor.",
            )

        if auto_sync_map:
            components.html(
                """
                <script>
                (function() {
                    try {
                        const parent = window.parent;
                        if (!parent) return;

                        if (parent.__operatorMapAutoSyncTimer) {
                            clearInterval(parent.__operatorMapAutoSyncTimer);
                            parent.__operatorMapAutoSyncTimer = null;
                        }

                        const runSync = () => {
                            try {
                                const tabs = Array.from(parent.document.querySelectorAll('[data-baseweb="tab"]'));
                                const mapTab = tabs.find((t) => /mapa y navegacion/i.test((t.innerText || '').trim()));
                                const mapTabActive = mapTab && mapTab.getAttribute('aria-selected') === 'true';
                                if (!mapTabActive) return;

                                const now = Date.now();
                                const nextAllowed = Number(parent.sessionStorage.getItem('operatorNextMapSyncAt') || '0');
                                if (now < nextAllowed) return;
                                parent.sessionStorage.setItem('operatorNextMapSyncAt', String(now + 600));

                                const btns = Array.from(parent.document.querySelectorAll('button'));
                                const syncBtn = btns.find((b) => /sync mapa operador/i.test((b.innerText || '').trim()));
                                if (syncBtn) {
                                    syncBtn.click();
                                }
                            } catch (e) {}
                        };

                        runSync();
                        parent.__operatorMapAutoSyncTimer = parent.setInterval(runSync, 180);
                    } catch (e) {}
                })();
                </script>
                """,
                height=0,
            )
        else:
            components.html(
                """
                <script>
                (function() {
                    try {
                        const parent = window.parent;
                        if (!parent) return;
                        if (parent.__operatorMapAutoSyncTimer) {
                            clearInterval(parent.__operatorMapAutoSyncTimer);
                            parent.__operatorMapAutoSyncTimer = null;
                        }
                    } catch (e) {}
                })();
                </script>
                """,
                height=0,
            )

        st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
        map_render_driver_map(
            scenario=scenario,
            bases=bases,
            incidents=incidents,
            hospitals_count=n_hospitales,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        map_render_fallback_map(scenario=scenario, bases=bases)


if __name__ == "__main__":
    main()