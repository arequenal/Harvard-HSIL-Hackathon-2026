import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
import random
import json
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.dispatch_shared import default_state, load_state, save_state

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "madrid_grafo.graphml"
HOSPITALES_PATH = BASE_DIR / "hospitales_madrid_nodos.csv"
PROCESSED_HOSPITALES_PATH = BASE_DIR.parent / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"

st.set_page_config(page_title="AmbulancIA · Conductor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;600&display=swap');

        :root {
            --bg-base:    #080e18;
            --bg-card:    #0d1624;
            --bg-card2:   #111c2e;
            --border:     rgba(0, 200, 255, 0.10);
            --border-hot: rgba(0, 200, 255, 0.32);
            --cyan:       #00c8ff;
            --amber:      #f5a623;
            --green:      #00e896;
            --red:        #ff4757;
            --text:       #d6e8f5;
            --muted:      #4d6a85;
            --glow-cyan:  0 0 18px rgba(0, 200, 255, 0.25);
            --glow-amber: 0 0 18px rgba(245, 166, 35, 0.30);
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-base) !important;
            color: var(--text) !important;
        }
        [data-testid="stAppViewContainer"]::before {
            content: '';
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse 70% 50% at 10% 0%,   rgba(0,200,255,0.07) 0%, transparent 60%),
                radial-gradient(ellipse 55% 45% at 90% 100%, rgba(0,232,150,0.05) 0%, transparent 55%);
            pointer-events: none;
            z-index: 0;
        }
        #MainMenu, header, footer { visibility: hidden; }
        .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            max-width: 100% !important;
            position: relative;
            z-index: 1;
        }

        /* ── HERO ── */
        .hero-shell {
            border-radius: 16px;
            padding: 20px 24px;
            margin: 0 0 16px 0;
            background: linear-gradient(120deg, #0a1929 0%, #0d2140 50%, #091d1a 100%);
            border: 1px solid var(--border-hot);
            box-shadow: var(--glow-cyan), inset 0 1px 0 rgba(0,200,255,0.08);
            position: relative;
            overflow: hidden;
        }
        .hero-shell::before {
            content: 'CONDUCTOR';
            position: absolute;
            right: 24px; top: 50%;
            transform: translateY(-50%);
            font-family: 'Rajdhani', sans-serif;
            font-size: 5rem;
            font-weight: 700;
            color: rgba(0, 200, 255, 0.04);
            letter-spacing: 0.15em;
            pointer-events: none;
            user-select: none;
        }
        .hero-shell h1 {
            margin: 0;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: #fff;
            text-transform: uppercase;
        }
        .hero-shell h1 span { color: var(--cyan); }
        .hero-shell p {
            margin: 6px 0 0;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.88rem;
            color: var(--muted);
            letter-spacing: 0.02em;
        }
        .live-dot {
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--green);
            margin-right: 6px;
            animation: pulseDot 1.6s ease-in-out infinite;
            vertical-align: middle;
        }
        @keyframes pulseDot {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,232,150,0.6); }
            50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0,232,150,0); }
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 4px;
            background: rgba(0, 200, 255, 0.08);
            border: 1px solid rgba(0, 200, 255, 0.22);
            color: var(--cyan);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            margin-right: 8px;
            margin-top: 12px;
            text-transform: uppercase;
        }

        /* ── STATUS CARDS ── */
        .state-strip {
            margin-top: 0;
            padding: 16px;
            border-radius: 14px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .panel-title {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--muted);
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin: 0 0 12px 0;
        }
        .status-card {
            background: var(--bg-card2);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 16px;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: border-color 0.3s, box-shadow 0.3s;
            animation: fadeSlideUp 0.4s ease both;
        }
        .status-card:hover {
            border-color: var(--border-hot);
            box-shadow: var(--glow-cyan);
        }
        @keyframes fadeSlideUp {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .status-label {
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: var(--muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .status-value {
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            font-weight: 700;
            color: var(--cyan);
            line-height: 1.2;
            word-break: break-word;
        }

        /* ── PILLS & ALERTS ── */
        .state-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }
        .state-pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 10px;
            border-radius: 4px;
            background: rgba(0,200,255,0.06);
            color: var(--cyan);
            border: 1px solid rgba(0,200,255,0.15);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.06em;
        }
        .alert-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 10px;
            border-radius: 4px;
            background: rgba(245, 166, 35, 0.08);
            border: 1px solid rgba(245, 166, 35, 0.22);
            color: var(--amber);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.04em;
        }
        .alert-chip--none {
            background: rgba(77, 106, 133, 0.10);
            border-color: rgba(77, 106, 133, 0.20);
            color: var(--muted);
        }

        /* ── CONTROLS ── */
        .control-strip {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 14px;
        }
        .stButton > button {
            background: transparent !important;
            border: 1px solid var(--border-hot) !important;
            color: var(--cyan) !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            border-radius: 6px !important;
            padding: 0.45rem 1.1rem !important;
            transition: background 0.2s, box-shadow 0.2s !important;
        }
        .stButton > button:hover {
            background: rgba(0, 200, 255, 0.10) !important;
            box-shadow: var(--glow-cyan) !important;
        }
        [data-testid="stAlert"] {
            background: rgba(245, 166, 35, 0.07) !important;
            border: 1px solid rgba(245, 166, 35, 0.25) !important;
            border-radius: 8px !important;
            color: var(--amber) !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        [data-testid="stExpander"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }
        [data-testid="stExpander"] summary {
            color: var(--muted) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.06em !important;
        }

        /* ── MAP WRAPPER ── */
        .map-wrap {
            margin-top: 16px;
            padding: 4px;
            background: var(--bg-card);
            border: 1px solid var(--border-hot);
            border-radius: 16px;
            box-shadow: var(--glow-cyan), 0 24px 60px rgba(0,0,0,0.5);
        }
        iframe {
            height: 100vh !important;
            border-radius: 13px;
            display: block;
        }

        /* ── SOFT SYNC ── */
        body.conductor-soft-sync {
            opacity: 0.88;
            filter: saturate(0.9) brightness(0.97);
            transition: opacity 180ms ease, filter 180ms ease;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_graph_with_traffic():
    G = ox.load_graphml(GRAPH_PATH)
    for u, v, key, data in G.edges(keys=True, data=True):
        traffic_factor = random.uniform(1.0, 3.0)
        data['traffic_factor'] = traffic_factor
        data['weighted_length'] = data.get('length', 1.0) * traffic_factor
    return G


@st.cache_data
def cargar_hospitales(_grafo):
    if PROCESSED_HOSPITALES_PATH.exists():
        df = pd.read_csv(PROCESSED_HOSPITALES_PATH, sep=";")

        if "nombre" not in df.columns:
            df["nombre"] = df["centro_id"].astype(str)

        for col in ["lat", "lon"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["lat"].notna() & df["lon"].notna()].copy()

        df["nodo_red"] = df.apply(
            lambda row: int(ox.distance.nearest_nodes(_grafo, X=float(row["lon"]), Y=float(row["lat"]))),
            axis=1,
        )
    else:
        df = pd.read_csv(HOSPITALES_PATH)

    for col in ["direccion_completa", "especialidades_texto", "perfiles_atencion"]:
        if col not in df.columns:
            df[col] = ""
    return df


grafo = load_graph_with_traffic()
df_hospitales = cargar_hospitales(grafo)
lista_nodos = list(grafo.nodes())

ZONAS_TRAFICO = [
    {"lat": 40.4215, "lon": -3.6590, "radio": 1500, "nivel": "Alto"},
    {"lat": 40.3920, "lon": -3.6850, "radio": 1600, "nivel": "Alto"},
    {"lat": 40.4490, "lon": -3.6450, "radio": 1300, "nivel": "Alto"},
    {"lat": 40.4650, "lon": -3.6880, "radio": 1400, "nivel": "Medio"},
    {"lat": 40.4190, "lon": -3.7020, "radio": 1100, "nivel": "Medio"},
    {"lat": 40.4080, "lon": -3.6750, "radio": 900, "nivel": "Bajo"}
]

AMBULATORIOS = [
    {"nombre": "C.S. Chamberí", "lat": 40.4338, "lon": -3.7020},
    {"nombre": "C.S. Pacífico", "lat": 40.4026, "lon": -3.6732},
    {"nombre": "C.S. Delicias", "lat": 40.3953, "lon": -3.6946},
    {"nombre": "C.S. Goya", "lat": 40.4245, "lon": -3.6762},
    {"nombre": "C.S. Salamanca", "lat": 40.4285, "lon": -3.6811},
    {"nombre": "C.S. San Fermín", "lat": 40.3705, "lon": -3.6925},
    {"nombre": "C.S. Argüelles", "lat": 40.4281, "lon": -3.7153},
    {"nombre": "C.S. Moratalaz", "lat": 40.4081, "lon": -3.6521},
    {"nombre": "C.S. Fuencarral", "lat": 40.4901, "lon": -3.6931},
    {"nombre": "C.S. Latina", "lat": 40.3855, "lon": -3.7381}
]
for amb in AMBULATORIOS:
    amb['nodo_red'] = int(ox.distance.nearest_nodes(grafo, X=amb['lon'], Y=amb['lat']))

SOS_POINTS = [
    {"nombre": "SOS Gran Via", "lat": 40.4202, "lon": -3.7058},
    {"nombre": "SOS Retiro", "lat": 40.4166, "lon": -3.6844},
    {"nombre": "SOS Ventas", "lat": 40.4312, "lon": -3.6637},
    {"nombre": "SOS Cuatro Caminos", "lat": 40.4473, "lon": -3.7033},
    {"nombre": "SOS Legazpi", "lat": 40.3918, "lon": -3.6941},
]


def get_active_sos() -> dict:
    if "active_sos" not in st.session_state:
        st.session_state["active_sos"] = random.choice(SOS_POINTS).copy()
    sos = dict(st.session_state["active_sos"])
    sos["nodo_red"] = int(ox.distance.nearest_nodes(grafo, X=float(sos["lon"]), Y=float(sos["lat"])))
    return sos


def _best_base_to_target(target_node: int):
    best = AMBULATORIOS[0]
    best_cost = float('inf')
    for amb in AMBULATORIOS:
        try:
            cost = nx.shortest_path_length(grafo, source=amb['nodo_red'], target=target_node, weight='weighted_length')
        except nx.NetworkXNoPath:
            continue
        if cost < best_cost:
            best_cost = cost
            best = amb
    return best


def reset_operativo() -> None:
    save_state(default_state())
    st.session_state.pop("active_sos", None)
    st.cache_data.clear()
    st.cache_resource.clear()

def _route_nodes(source_node: int, target_node: int):
    route_graphs = [
        (grafo, 'weighted_length'),
        (grafo, 'length'),
        (grafo.to_undirected(), 'weighted_length'),
        (grafo.to_undirected(), 'length'),
    ]
    for route_graph, weight in route_graphs:
        try:
            return nx.shortest_path(route_graph, source=source_node, target=target_node, weight=weight)
        except nx.NetworkXNoPath:
            continue
    return None


def _route_coords(route_nodes):
    coords = []
    for node in route_nodes:
        coords.append([float(grafo.nodes[node]['y']), float(grafo.nodes[node]['x'])])
    return coords


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().strip().split())


def _resolve_destination_candidate(df: pd.DataFrame, destination_id: str, destination_name: str) -> pd.DataFrame:
    candidate = pd.DataFrame()

    if destination_id:
        id_norm = destination_id.strip().upper()
        candidate = df[df["centro_id"].astype(str).str.strip().str.upper() == id_norm]
        if not candidate.empty:
            return candidate

    if destination_name:
        name_norm = _normalize_text(destination_name)
        series_norm = df["nombre"].astype(str).apply(_normalize_text)

        exact = df[series_norm.eq(name_norm)]
        if not exact.empty:
            return exact

        contains = df[series_norm.str.contains(name_norm, regex=False)]
        if not contains.empty:
            return contains

        scores = series_norm.apply(lambda n: SequenceMatcher(None, name_norm, n).ratio())
        best_idx = scores.idxmax() if not scores.empty else None
        if best_idx is not None and float(scores.loc[best_idx]) >= 0.56:
            return df.loc[[best_idx]]

    return candidate


@st.cache_resource
def startup_reset_once() -> bool:
    # Ensure shared state file exists without overwriting operator updates.
    load_state()
    return True


def render_state_debug(state: dict[str, object], destination_id: str, destination_name: str, traffic_alerts: list[str]) -> None:
    with st.container():
        alert_count = len(traffic_alerts)
        alert_color = "#f5a623" if alert_count else "#4d6a85"
        alert_label = f"{alert_count} ALERTA{'S' if alert_count != 1 else ''}" if alert_count else "SIN ALERTAS"
        dest_display = destination_name or destination_id or "ESPERANDO OPERADOR"

        st.markdown(
            f"""
            <div class="hero-shell">
                <h1>Ambulanc<span>IA</span> · Conductor</h1>
                <p>Ruta operativa en tiempo real · Sincronización con operador · Red vial de Madrid</p>
                <div style="margin-top:12px; display:flex; flex-wrap:wrap; gap:8px; align-items:center;">
                    <span class="status-chip"><span class="live-dot"></span>EN VIVO</span>
                    <span class="status-chip">RED VIAL</span>
                    <span class="status-chip" style="color:{alert_color}; border-color:{alert_color}33;">⚠ {alert_label}</span>
                    <span class="status-chip" style="color:#00e896; border-color:#00e89633;">↗ {dest_display[:32]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_sync, col_reset, col_hint = st.columns([1, 1, 4], gap="small")
        with col_sync:
            if st.button("⟳  Sincronizar", type="primary", use_container_width=True):
                st.rerun()
        with col_reset:
            if st.button("↺  Reset", type="secondary", use_container_width=True):
                reset_operativo()
                st.rerun()
        with col_hint:
            st.markdown(
                '<p style="padding:0.55rem 0.4rem; font-family:\'JetBrains Mono\',monospace; '
                'font-size:0.72rem; color:#4d6a85; letter-spacing:0.04em; margin:0;">'
                '// Estado se refresca al pulsar Sincronizar. En espera SOS, auto-sync silencioso en segundo plano.</p>',
                unsafe_allow_html=True,
            )

        updated_at = str(state.get("updated_at", "-"))[:19].replace("T", " ")
        version_val = state.get("version", 0)

        st.markdown('<div class="state-strip">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">▸ Estado compartido</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.markdown(
                f'<div class="status-card" style="animation-delay:0.05s">'
                f'<span class="status-label">// versión</span>'
                f'<span class="status-value">v{version_val}</span></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="status-card" style="animation-delay:0.10s">'
                f'<span class="status-label">// actualizado</span>'
                f'<span class="status-value" style="font-size:0.82rem;">{updated_at}</span></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="status-card" style="animation-delay:0.15s">'
                f'<span class="status-label">// id destino</span>'
                f'<span class="status-value">{destination_id or "—"}</span></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="status-card" style="animation-delay:0.20s">'
                f'<span class="status-label">// nombre destino</span>'
                f'<span class="status-value" style="font-size:0.82rem;">{destination_name or "—"}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="state-meta">', unsafe_allow_html=True)
        st.markdown('<span class="state-pill">● SYNC ACTIVO</span>', unsafe_allow_html=True)
        st.markdown('<span class="state-pill">◈ RUTA RED VIAL</span>', unsafe_allow_html=True)
        if traffic_alerts:
            for a in traffic_alerts:
                st.markdown(f'<span class="alert-chip">⚠ {a}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="alert-chip alert-chip--none">✓ sin alertas de tráfico</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        destination_payload = state.get("destination", {}) if isinstance(state.get("destination", {}), dict) else {}
        extra_cols = st.columns(3, gap="small")
        with extra_cols[0]:
            st.markdown(
                f'<div class="status-card" style="animation-delay:0.25s">'
                f'<span class="status-label">// teléfono</span>'
                f'<span class="status-value" style="font-size:0.82rem;">{str(destination_payload.get("telefono", "") or "—")}</span></div>',
                unsafe_allow_html=True,
            )
        with extra_cols[1]:
            st.markdown(
                f'<div class="status-card" style="animation-delay:0.30s">'
                f'<span class="status-label">// municipio</span>'
                f'<span class="status-value" style="font-size:0.82rem;">{str(destination_payload.get("municipio", "") or "—")}</span></div>',
                unsafe_allow_html=True,
            )
        with extra_cols[2]:
            st.markdown(
                f'<div class="status-card" style="animation-delay:0.35s">'
                f'<span class="status-label">// centro tipo</span>'
                f'<span class="status-value" style="font-size:0.82rem;">{str(destination_payload.get("centro_tipo", "") or "—")}</span></div>',
                unsafe_allow_html=True,
            )

        with st.expander("// ver estado JSON completo"):
            st.json(state)


# Reset automatico al iniciar el conductor (una vez por arranque del proceso).
startup_reset_once()
if "startup_session_init" not in st.session_state:
    st.session_state.pop("active_sos", None)
    st.session_state.pop("ambulance_waiting_at_sos", None)
    st.session_state["startup_session_init"] = True

# Leer destino real publicado por el operador
state = load_state()
destination_id = str(state.get("destination", {}).get("centro_id", "")).strip()
destination_name = str(state.get("destination", {}).get("nombre", "")).strip()
destination_payload = state.get("destination", {}) if isinstance(state.get("destination", {}), dict) else {}
traffic_alerts = [str(a) for a in state.get("traffic_alerts", []) if str(a).strip()]
active_sos = get_active_sos()
render_state_debug(state, destination_id, destination_name, traffic_alerts)

candidate = _resolve_destination_candidate(df_hospitales, destination_id, destination_name)
has_destination_signal = bool(destination_id or destination_name)
has_destination_coords = destination_payload.get("lat") is not None and destination_payload.get("lon") is not None
has_destination = (not candidate.empty or has_destination_coords) and has_destination_signal
was_waiting_at_sos = bool(st.session_state.get("ambulance_waiting_at_sos", False))
route_status = "Ruta en curso: ambulancia desplazandose a la señal SOS."
route_to_sos = []
route_to_hospital = []
selected_hospital = None
selected_base = _best_base_to_target(int(active_sos['nodo_red']))

route_nodes_to_sos = _route_nodes(int(selected_base['nodo_red']), int(active_sos['nodo_red']))
if route_nodes_to_sos:
    route_to_sos = _route_coords(route_nodes_to_sos)
else:
    route_to_sos = [[float(selected_base['lat']), float(selected_base['lon'])], [float(active_sos['lat']), float(active_sos['lon'])]]

if len(route_to_sos) < 2:
    route_to_sos = [[float(selected_base['lat']), float(selected_base['lon'])], [float(active_sos['lat']), float(active_sos['lon'])]]
else:
    final_sos_point = [float(active_sos['lat']), float(active_sos['lon'])]
    if route_to_sos[-1] != final_sos_point:
        route_to_sos.append(final_sos_point)

selected_hospital_info = {
    "nombre": "Destino pendiente por operador",
    "lat": float(active_sos["lat"]),
    "lon": float(active_sos["lon"]),
    "direccion": "",
    "especialidades": "",
    "perfiles": "",
    "occ": random.randint(30, 98),
    "wait": random.randint(10, 120),
}

if has_destination:
    if not candidate.empty:
        selected_hospital = candidate.iloc[0]
        selected_target = int(selected_hospital['nodo_red'])
    else:
        selected_hospital = None
        selected_target = int(
            ox.distance.nearest_nodes(
                grafo,
                X=float(destination_payload.get("lon")),
                Y=float(destination_payload.get("lat")),
            )
        )
    route_nodes = _route_nodes(int(active_sos['nodo_red']), selected_target)
    if was_waiting_at_sos:
        route_status = "Orden recibida: salida inmediata desde SOS hacia el hospital indicado por operador."
        route_to_sos = [[float(active_sos['lat']), float(active_sos['lon'])], [float(active_sos['lat']), float(active_sos['lon'])]]
    else:
        route_status = "Ruta en dos tramos: base → SOS y SOS → hospital indicado por operador."

    if route_nodes:
        route_to_hospital = _route_coords(route_nodes)
    else:
        route_status = "No existe una ruta conectada entre SOS y hospital. Se muestra una traza directa para el segundo tramo."
        route_to_hospital = []

    dest_lat = float(selected_hospital['lat']) if selected_hospital is not None else float(destination_payload.get("lat"))
    dest_lon = float(selected_hospital['lon']) if selected_hospital is not None else float(destination_payload.get("lon"))

    if len(route_to_hospital) < 2:
        route_to_hospital = [[float(active_sos['lat']), float(active_sos['lon'])], [dest_lat, dest_lon]]
        if route_status == "Ruta en dos tramos: base → SOS y SOS → hospital indicado por operador.":
            route_status = "El tramo SOS → hospital se dibuja en modo directo."
    else:
        final_hospital_point = [dest_lat, dest_lon]
        if route_to_hospital[-1] != final_hospital_point:
            route_to_hospital.append(final_hospital_point)

    selected_hospital_info = {
        "nombre": str(
            (selected_hospital.get("nombre", "") if selected_hospital is not None else destination_payload.get("nombre", ""))
            or destination_name
            or "Hospital destino"
        ),
        "lat": dest_lat,
        "lon": dest_lon,
        "direccion": str(
            (selected_hospital.get("direccion_completa", "") if selected_hospital is not None else destination_payload.get("direccion", ""))
            or destination_payload.get("direccion", "")
            or ""
        ),
        "especialidades": str(
            (selected_hospital.get("especialidades_texto", "") if selected_hospital is not None else destination_payload.get("especialidades", ""))
            or destination_payload.get("especialidades", "")
            or ""
        ),
        "perfiles": str(
            (selected_hospital.get("perfiles_atencion", "") if selected_hospital is not None else destination_payload.get("perfiles", ""))
            or destination_payload.get("perfiles", "")
            or ""
        ),
        "occ": random.randint(30, 98),
        "wait": random.randint(10, 120),
    }
    st.session_state["ambulance_waiting_at_sos"] = False

if destination_id and destination_name and candidate.empty:
    route_status = "Destino recibido pero no localizado en la base; la ambulancia llega al SOS y espera nueva orden del operador."

if not has_destination and not has_destination_signal:
    if was_waiting_at_sos:
        route_status = "Ambulancia en punto SOS, esperando orden del operador."
        route_to_sos = [[float(active_sos['lat']), float(active_sos['lon'])], [float(active_sos['lat']), float(active_sos['lon'])]]
    else:
        route_status = "Ruta en curso: base → SOS. Al llegar al SOS, la ambulancia quedara esperando la orden del operador."
    st.session_state["ambulance_waiting_at_sos"] = True

auto_refresh_waiting = bool(st.session_state.get("ambulance_waiting_at_sos", False)) and not has_destination_signal

if auto_refresh_waiting:
    # Discreet waiting mode: trigger Streamlit sync button in background.
    components.html(
        """
        <script>
        (function() {
            try {
                if (window.parent.__conductorAutoReloadTimer) {
                    clearInterval(window.parent.__conductorAutoReloadTimer);
                }

                const syncIfVisible = () => {
                    try {
                        if (window.parent.document.visibilityState !== 'visible') return;

                        const btns = Array.from(window.parent.document.querySelectorAll('button'));
                        const syncBtn = btns.find((b) => /sincronizar/i.test((b.innerText || '').trim()));

                        if (syncBtn) {
                            syncBtn.click();
                            return;
                        }

                        // Rare fallback if the button is temporarily unavailable.
                        const now = Date.now();
                        const lastHard = Number(window.parent.sessionStorage.getItem('conductorLastHardReloadAt') || '0');
                        if (now - lastHard > 15000) {
                            window.parent.sessionStorage.setItem('conductorLastHardReloadAt', String(now));
                            window.parent.sessionStorage.setItem('conductorScrollY', String(window.parent.scrollY || 0));
                            window.parent.location.reload();
                        }
                    } catch (e) {}
                };

                window.parent.__conductorAutoReloadTimer = window.setInterval(syncIfVisible, 250);
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
                if (window.parent.__conductorAutoReloadTimer) {
                    clearInterval(window.parent.__conductorAutoReloadTimer);
                    window.parent.__conductorAutoReloadTimer = null;
                }
            } catch (e) {}
        })();
        </script>
        """,
        height=0,
    )

st.info(route_status)

operativo = {
    "hospitales": [
        {
            "nombre": str(row.get("nombre", "Hospital")),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "nodo_red": int(row["nodo_red"]),
            "direccion": str(row.get("direccion_completa", "") or ""),
            "especialidades": str(row.get("especialidades_texto", "") or ""),
            "perfiles": str(row.get("perfiles_atencion", "") or ""),
            "occ": random.randint(35, 98),
            "wait": random.randint(8, 125),
        }
        for _, row in df_hospitales.iterrows()
    ],
    "gps_sos": route_to_sos,
    "gps_ida": route_to_hospital,
    "destino": selected_hospital_info,
    "origen": selected_base,
    "sos": active_sos,
    "was_waiting_at_sos": was_waiting_at_sos,
    "esperando_destino": not has_destination,
    "auto_refresh_waiting": auto_refresh_waiting,
    "mensaje": f"<b>Señal SOS activa</b><br>{active_sos['nombre']}<br>La ambulancia acudira primero al SOS y despues al hospital indicado",
}

ambu_json = json.dumps(AMBULATORIOS)
zonas_json = json.dumps(ZONAS_TRAFICO)
operativos_json = json.dumps([operativo])
alerts_json = json.dumps(traffic_alerts if traffic_alerts else ["Atasco moderado en la vía principal", "Obras puntuales en acceso secundario"])

html_crudo = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@600;700&display=swap');
        html, body, #map { height: 100%; width: 100%; margin: 0; padding: 0; background-color: #080e18; overflow: hidden; }

        @keyframes pulseRed  { 0%,100%{box-shadow:0 0 0 0 rgba(255,71,87,0.8)} 60%{box-shadow:0 0 0 14px rgba(255,71,87,0)} }
        @keyframes pulseCyan { 0%,100%{box-shadow:0 0 0 0 rgba(0,200,255,0.7)} 60%{box-shadow:0 0 0 12px rgba(0,200,255,0)} }
        @keyframes glowTarget{ 0%,100%{box-shadow:0 0 6px 2px rgba(0,200,255,0.4)} 50%{box-shadow:0 0 22px 8px rgba(0,200,255,0.75)} }
        @keyframes scanline  { 0%{background-position:0 0} 100%{background-position:0 4px} }

        .sos-marker {
            background: linear-gradient(135deg,#c0001a,#ff2240);
            border: 2px solid rgba(255,255,255,0.9); border-radius: 50%;
            color: white; font-family:'Rajdhani',sans-serif; font-weight:700; font-size:11px;
            text-align:center; line-height:28px; letter-spacing:0.06em;
            animation: pulseRed 1.4s infinite;
        }
        .amb-icon { font-size:30px; text-align:center; filter:drop-shadow(0 0 8px rgba(0,200,255,0.9)); z-index:1001 !important; }
        .hosp-marker {
            background: #0d1624; border: 2px solid; border-radius: 8px;
            text-align:center; line-height:26px; font-size:15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.6);
            transition: all 0.4s ease;
        }
        .hosp-green  { border-color:#00e896; box-shadow:0 0 8px rgba(0,232,150,0.25); }
        .hosp-orange { border-color:#f5a623; box-shadow:0 0 8px rgba(245,166,35,0.25); }
        .hosp-red    { border-color:#ff4757; box-shadow:0 0 8px rgba(255,71,87,0.25); }
        .hosp-target { border-color:#00c8ff !important; animation:glowTarget 1.5s infinite; z-index:900 !important; transform:scale(1.25); }
        .ambu-marker { background:#0d1624; border:2px solid #00c8ff; border-radius:6px; text-align:center; line-height:22px; font-size:15px; }

        .custom-tip {
            font-family:'JetBrains Mono',monospace; font-size:12px;
            border-radius:8px; border:1px solid rgba(0,200,255,0.25) !important;
            background:#0d1624 !important; color:#d6e8f5 !important;
            box-shadow:0 4px 16px rgba(0,0,0,0.6), 0 0 12px rgba(0,200,255,0.12);
        }
        .leaflet-tooltip.custom-tip { padding:8px 12px; }
        .traffic-tip { font-weight:700; }

        .progress-bg   { background:#1a2a3a; width:100%; height:6px; border-radius:3px; margin-top:5px; overflow:hidden; }
        .progress-fill { height:100%; border-radius:3px; transition:width 0.5s ease; }

        #hud {
            position:absolute; top:14px; right:14px; z-index:1200;
            background:rgba(8,14,24,0.94);
            border:1px solid rgba(0,200,255,0.22);
            border-radius:12px; padding:14px 16px;
            width:300px;
            font-family:'JetBrains Mono',monospace;
            box-shadow:0 0 24px rgba(0,200,255,0.12), 0 8px 32px rgba(0,0,0,0.7);
            backdrop-filter:blur(8px);
        }
        #hud::before {
            content:'';
            position:absolute; inset:0; border-radius:12px;
            background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,200,255,0.015) 3px,rgba(0,200,255,0.015) 4px);
            pointer-events:none;
        }
        .hud-title {
            font-family:'Rajdhani',sans-serif; font-weight:700;
            font-size:0.95rem; letter-spacing:0.1em; text-transform:uppercase;
            color:#fff; margin-bottom:10px;
            border-bottom:1px solid rgba(0,200,255,0.15); padding-bottom:8px;
            display:flex; justify-content:space-between; align-items:center;
        }
        .live-badge {
            font-family:'JetBrains Mono',monospace; font-size:0.6rem;
            color:#00e896; border:1px solid rgba(0,232,150,0.35);
            padding:2px 6px; border-radius:3px; letter-spacing:0.1em;
            animation:pulseCyan 2s infinite;
        }
        .hud-row {
            display:flex; justify-content:space-between; align-items:center;
            margin:6px 0; font-size:0.72rem;
        }
        .hud-label { color:rgba(100,160,200,0.7); letter-spacing:0.08em; text-transform:uppercase; }
        .hud-val   { color:#00c8ff; font-weight:700; font-size:0.8rem; }
        .hud-sep   { border:none; border-top:1px solid rgba(0,200,255,0.10); margin:8px 0; }
        .alerts-title { color:rgba(245,166,35,0.8); font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px; }
        .alerts { font-size:0.68rem; max-height:80px; overflow:auto; color:rgba(214,232,245,0.65); }
        .alerts li { margin:3px 0; list-style:none; padding-left:0; }
        .alerts li::before { content:'⚠ '; color:#f5a623; }
    </style>
</head>
<body>
    <div id="hud">
        <div class="hud-title">Conductor AmbulancIA <span class="live-badge">EN VIVO</span></div>
        <div class="hud-row"><span class="hud-label">Destino</span><b class="hud-val" id="kDestino">—</b></div>
        <div class="hud-row"><span class="hud-label">ETA</span><b class="hud-val" id="kEta">—</b></div>
        <div class="hud-row"><span class="hud-label">Distancia</span><b class="hud-val" id="kDist">—</b></div>
        <hr class="hud-sep">
        <div class="alerts-title">Alertas de tráfico</div>
        <ul class="alerts" id="alerts"></ul>
    </div>
    <div id="map"></div>

    <script>
        const ambulatorios = __AMBULATORIOS__;
        const zonasTrafico = __ZONAS_TRAFICO__;
        const operativos = __OPERATIVOS__;
        const alerts = __ALERTS__;

        const map = L.map('map', {preferCanvas: true}).setView([40.4168, -3.7038], 13);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
            attribution: '© CartoDB'
        }).addTo(map);

        // Restore parent page scroll after auto-refresh to avoid visual jump.
        try {
            const softSyncUntil = Number(window.parent.sessionStorage.getItem('conductorSyncUntil') || '0');
            if (softSyncUntil && Date.now() < softSyncUntil) {
                window.parent.document.body.classList.add('conductor-soft-sync');
                window.setTimeout(() => {
                    try {
                        window.parent.document.body.classList.remove('conductor-soft-sync');
                    } catch (e) {}
                }, Math.max(0, softSyncUntil - Date.now()) + 220);
                window.parent.sessionStorage.removeItem('conductorSyncUntil');
            }

            const savedY = window.parent.sessionStorage.getItem('conductorScrollY');
            if (savedY !== null) {
                window.parent.scrollTo(0, parseInt(savedY, 10));
                window.parent.sessionStorage.removeItem('conductorScrollY');
            }
        } catch (e) {}

        ambulatorios.forEach(function(a) {
            const icon = L.divIcon({className: 'ambu-marker', html: '🩺', iconSize: [26,26], iconAnchor: [13,13]});
            L.marker([a.lat, a.lon], {icon: icon}).bindTooltip("<b style='color:#00c8ff'>BASE SVB</b><br>" + a.nombre, {direction: 'top', className: 'custom-tip'}).addTo(map);
        });

        const alertsEl = document.getElementById('alerts');
        if (alerts.length) {
            alerts.forEach((a) => {
                const li = document.createElement('li');
                li.textContent = a;
                alertsEl.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'Sin alertas activas';
            li.style.color = 'rgba(0,232,150,0.7)';
            li.style.listStyle = 'none';
            li.textContent = '✓ Sin alertas activas';
            alertsEl.appendChild(li);
        }

        const op = operativos[0];
        document.getElementById('kDestino').textContent = op.esperando_destino ? 'Esperando operador' : op.destino.nombre;

        const sosIcon = L.divIcon({className: 'sos-marker', html: 'SOS', iconSize: [28,28], iconAnchor: [14,14]});
        const markerSOS = L.marker([op.sos.lat, op.sos.lon], {icon: sosIcon}).addTo(map);
        markerSOS.bindTooltip('<b style="color:#ff4757">SEÑAL SOS</b><br>' + op.sos.nombre, {direction: 'top', className: 'custom-tip'});

        const hospitalMarkers = {};
        op.hospitales.forEach(function(h) {
            const colorClass = (h.occ > 85 ? 'hosp-red' : (h.occ > 50 ? 'hosp-orange' : 'hosp-green'));
            const barColor = h.occ > 85 ? '#ff4757' : (h.occ > 50 ? '#f5a623' : '#00e896');
            const icon = L.divIcon({className: 'hosp-marker ' + colorClass, html: '🏥', iconSize: [30,30], iconAnchor: [15,15]});
            const marker = L.marker([h.lat, h.lon], {icon: icon}).addTo(map);
            const tipHTML = `
              <div style="text-align:left; min-width:200px;">
                <div style="font-family:'Rajdhani',sans-serif; font-size:14px; font-weight:700; color:#fff; margin-bottom:6px;">${h.nombre}</div>
                <div style="color:rgba(0,200,255,0.7); font-size:10px; margin-bottom:6px;">📍 ${h.direccion || '—'}</div>
                <div style="color:rgba(214,232,245,0.6); font-size:10px; margin-bottom:4px;">🩺 ${h.especialidades || 'No disponible'}</div>
                <div style="display:flex; justify-content:space-between; margin-top:6px;">
                  <span>🛏️ Ocupación: <b style="color:${barColor}">${h.occ}%</b></span>
                  <span>⏱ <b style="color:#f5a623">${h.wait} min</b></span>
                </div>
                <div class="progress-bg"><div class="progress-fill" style="width: ${h.occ}%; background: linear-gradient(90deg,${barColor},${barColor}88);"></div></div>
              </div>`;
            marker.bindTooltip(tipHTML, {direction: 'top', offset: [0, -15], className: 'custom-tip'});
            hospitalMarkers[h.nombre] = marker;
        });

        function densificar(ruta, maxDist) {
            const nueva = [];
            for (let i = 0; i < ruta.length - 1; i++) {
                const p1 = ruta[i], p2 = ruta[i + 1];
                const dist = Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2));
                const pasos = Math.max(1, Math.ceil(dist / maxDist));
                for (let j = 0; j < pasos; j++) {
                    nueva.push([p1[0] + (p2[0]-p1[0]) * (j/pasos), p1[1] + (p2[1]-p1[1]) * (j/pasos)]);
                }
            }
            nueva.push(ruta[ruta.length - 1]);
            return nueva;
        }

        function haversineM(a, b) {
          const R = 6371000;
          const rad = x => x * Math.PI / 180;
          const dLat = rad(b[0]-a[0]), dLon = rad(b[1]-a[1]);
          const h = Math.sin(dLat/2)**2 + Math.cos(rad(a[0]))*Math.cos(rad(b[0]))*Math.sin(dLon/2)**2;
          return 2*R*Math.atan2(Math.sqrt(h), Math.sqrt(1-h));
        }

        function remainingDist(idx, ruta) {
          let d = 0;
          for (let k = idx; k < ruta.length - 1; k++) d += haversineM(ruta[k], ruta[k+1]);
          return d;
        }

                const gpsSos = op.gps_sos || [];
                const gpsHosp = op.gps_ida || [];
                const ambStart = [op.origen.lat, op.origen.lon];
                const hasSosRoute = gpsSos.length >= 2;
                const hasHospRoute = !op.esperando_destino && gpsHosp.length >= 2;

                const sosSuave = hasSosRoute ? densificar(gpsSos, 0.00018) : [ambStart, [op.sos.lat, op.sos.lon]];
                const hospSuave = hasHospRoute ? densificar(gpsHosp, 0.00018) : [[op.sos.lat, op.sos.lon]];

                const routeLineSos = hasSosRoute ? L.polyline(gpsSos, {color: '#f5a623', weight: 3, opacity: 0.65, dashArray: '8,10'}).addTo(map) : null;
                const routeLineHosp = hasHospRoute ? L.polyline(gpsHosp, {color: '#00c8ff', weight: 3, opacity: 0.55, dashArray: '6,9'}).addTo(map) : null;
                const doneLine = L.polyline([sosSuave[0]], {color: '#00e896', weight: 5, opacity: 0.9}).addTo(map);

        const ambIcon = L.divIcon({className: 'amb-icon', html: '🚑', iconSize: [32,32], iconAnchor: [16,16]});
        const markerAmb = L.marker(sosSuave[0], {icon: ambIcon}).addTo(map);

                if (!op.esperando_destino && hospitalMarkers[op.destino.nombre]) {
          hospitalMarkers[op.destino.nombre].setIcon(
            L.divIcon({className: 'hosp-marker hosp-target', html: '🏥🏁', iconSize: [36,36], iconAnchor: [18,18]})
          );
          hospitalMarkers[op.destino.nombre].bindTooltip(op.mensaje, {direction: 'top', className: 'custom-tip'});
        }

                if (hasHospRoute && routeLineHosp) {
                    map.fitBounds(routeLineHosp.getBounds().pad(0.2));
                } else if (hasSosRoute && routeLineSos) {
                    map.fitBounds(routeLineSos.getBounds().pad(0.2));
                } else {
                    map.setView(ambStart, 14);
                    document.getElementById('kEta').textContent = '--';
                    document.getElementById('kDist').textContent = '0 m';
                }

                const pasoAnimacion = 4;
                const tickMs = 50;

                function animarRuta(path, etaBase, onFinish) {
                    let idx = 0;
                    function step() {
                        if (idx >= path.length) {
                            if (onFinish) onFinish();
                            return;
                        }
                        markerAmb.setLatLng(path[idx]);
                        doneLine.setLatLngs(path.slice(0, idx + 1));
                        const progress = idx / Math.max(1, path.length - 1);
                        const etaNow = Math.max(0, Math.round(etaBase * (1 - progress)));
                        const rem = remainingDist(idx, path);
                        document.getElementById('kEta').textContent = etaNow + ' min';
                        document.getElementById('kDist').textContent = rem >= 1000 ? (rem / 1000).toFixed(1) + ' km' : Math.round(rem) + ' m';
                        idx += pasoAnimacion;
                        setTimeout(step, tickMs);
                    }
                    step();
                }

                const etaSos = Math.max(1, Math.round((gpsSos.length || 2) / 10));
                const etaHosp = Math.max(1, Math.round((gpsHosp.length || 2) / 10));

                function triggerParentSync() {
                    try {
                        const now = Date.now();
                        const nextAllowed = Number(window.parent.sessionStorage.getItem('conductorNextSyncAt') || '0');
                        if (now < nextAllowed) return;

                        // Ultra-fast polling while waiting for operator approval.
                        window.parent.sessionStorage.setItem('conductorNextSyncAt', String(now + 800));
                        window.parent.sessionStorage.setItem('conductorScrollY', String(window.parent.scrollY || 0));

                        // Short darkening pulse to communicate incoming sync.
                        window.parent.sessionStorage.setItem('conductorSyncUntil', String(now + 320));
                        window.parent.document.body.classList.add('conductor-soft-sync');
                        window.setTimeout(() => {
                            try {
                                window.parent.document.body.classList.remove('conductor-soft-sync');
                            } catch (e) {}
                        }, 360);
                    } catch (e) {}

                    try {
                        const btns = Array.from(window.parent.document.querySelectorAll('button'));
                        const syncBtn = btns.find((b) => (b.innerText || '').trim().includes('Sincronizar'));
                        if (syncBtn) syncBtn.click();
                    } catch (e) {}
                }

                setTimeout(() => {
                    animarRuta(sosSuave, etaSos, () => {
                        markerAmb.setLatLng([op.sos.lat, op.sos.lon]);
                        doneLine.setLatLngs(sosSuave);
                        document.getElementById('kDist').textContent = '0 m';
                        if (!hasHospRoute) {
                            document.getElementById('kEta').textContent = '--';
                            if (op.auto_refresh_waiting) {
                                setTimeout(() => {
                                    triggerParentSync();
                                }, 700);
                            }
                            return;
                        }
                        setTimeout(() => {
                            animarRuta(hospSuave, etaHosp, () => {
                                document.getElementById('kEta').textContent = '0 min';
                                document.getElementById('kDist').textContent = '0 m';
                            });
                        }, 900);
                    });
                }, 120);
    </script>
</body>
</html>
"""

html_mapa = html_crudo.replace("__AMBULATORIOS__", ambu_json)\
                      .replace("__ZONAS_TRAFICO__", zonas_json)\
                      .replace("__OPERATIVOS__", operativos_json)\
                      .replace("__ALERTS__", alerts_json)

st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
components.html(html_mapa, height=1000)
st.markdown('</div>', unsafe_allow_html=True)