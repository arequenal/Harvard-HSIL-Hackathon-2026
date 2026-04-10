import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
import random
import json
import sys
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

st.set_page_config(page_title="IA Ambulancias Smart City", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        iframe {
            height: 100vh !important;
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


@st.cache_resource
def startup_reset_once() -> bool:
    # Reset shared operator-conductor state once per conductor app startup.
    save_state(default_state())
    return True


def render_state_debug(state: dict[str, object], destination_id: str, destination_name: str, traffic_alerts: list[str]) -> None:
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Estado compartido", str(state.get("version", 0)))
        col2.metric("Ultima actualización", str(state.get("updated_at", "-"))[:19].replace("T", " "))
        col3.metric("Destino recibido", destination_id or "-")
        col4.metric("Nombre recibido", destination_name or "-")

        st.caption("Si estos campos cambian al publicar desde el operador, el conductor sí está recibiendo el estado compartido.")
        if traffic_alerts:
            st.info("Alertas recibidas: " + " | ".join(traffic_alerts))
        else:
            st.info("Alertas recibidas: ninguna")

        left, right = st.columns([3, 1])
        with right:
            if st.button("Reset operativo", type="secondary", use_container_width=True):
                reset_operativo()
                st.rerun()

        with st.expander("Ver estado compartido completo"):
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
traffic_alerts = [str(a) for a in state.get("traffic_alerts", []) if str(a).strip()]
active_sos = get_active_sos()
render_state_debug(state, destination_id, destination_name, traffic_alerts)

candidate = pd.DataFrame()
if destination_id:
    candidate = df_hospitales[df_hospitales["centro_id"].astype(str).str.strip().str.upper() == destination_id.upper()]
if candidate.empty and destination_name:
    name_norm = destination_name.lower().strip()
    candidate = df_hospitales[
        df_hospitales["nombre"].astype(str).str.lower().str.strip().eq(name_norm)
        | df_hospitales["nombre"].astype(str).str.lower().str.contains(name_norm, regex=False)
    ]

has_destination = (not candidate.empty) and (destination_id or destination_name)
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
    selected_hospital = candidate.iloc[0]
    selected_target = int(selected_hospital['nodo_red'])
    route_nodes = _route_nodes(int(active_sos['nodo_red']), selected_target)
    if was_waiting_at_sos:
        route_status = "Orden recibida: salida inmediata desde SOS hacia el hospital indicado por operador."
        route_to_sos = [[float(active_sos['lat']), float(active_sos['lon'])], [float(active_sos['lat']), float(active_sos['lon'])]]
    else:
        route_status = "Ruta en dos tramos: base → SOS y SOS → hospital indicado por operador."

    if route_nodes:
        route_to_hospital = _route_coords(route_nodes)
    else:
        route_status = "No existe una ruta conectada entre SOS y hospital. Se muestra una traza directa de respaldo para el segundo tramo."
        route_to_hospital = []

    if len(route_to_hospital) < 2:
        route_to_hospital = [[float(active_sos['lat']), float(active_sos['lon'])], [float(selected_hospital['lat']), float(selected_hospital['lon'])]]
        if route_status == "Ruta en dos tramos: base → SOS y SOS → hospital indicado por operador.":
            route_status = "El tramo SOS → hospital se dibuja en modo directo de respaldo."
    else:
        final_hospital_point = [float(selected_hospital['lat']), float(selected_hospital['lon'])]
        if route_to_hospital[-1] != final_hospital_point:
            route_to_hospital.append(final_hospital_point)

    selected_hospital_info = {
        "nombre": str(selected_hospital.get("nombre", destination_name or "Hospital destino")),
        "lat": float(selected_hospital["lat"]),
        "lon": float(selected_hospital["lon"]),
        "direccion": str(selected_hospital.get("direccion_completa", state.get("destination", {}).get("direccion", "") ) or ""),
        "especialidades": str(selected_hospital.get("especialidades_texto", "") or ""),
        "perfiles": str(selected_hospital.get("perfiles_atencion", "") or ""),
        "occ": random.randint(30, 98),
        "wait": random.randint(10, 120),
    }
    st.session_state["ambulance_waiting_at_sos"] = False

if destination_id and destination_name and candidate.empty:
    route_status = "Destino recibido pero no localizado en la base; la ambulancia llega al SOS y espera nueva orden del operador."

if not has_destination and not (destination_id or destination_name):
    if was_waiting_at_sos:
        route_status = "Ambulancia en punto SOS, esperando orden del operador."
        route_to_sos = [[float(active_sos['lat']), float(active_sos['lon'])], [float(active_sos['lat']), float(active_sos['lon'])]]
    else:
        route_status = "Ruta en curso: base → SOS. Al llegar al SOS, la ambulancia quedara esperando la orden del operador."
    st.session_state["ambulance_waiting_at_sos"] = True

auto_refresh_waiting = bool(st.session_state.get("ambulance_waiting_at_sos", False)) and not has_destination

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
        html, body, #map { height: 100%; width: 100%; margin: 0; padding: 0; background-color: #f4f4f4; overflow: hidden; }
        @keyframes pulseRed { 0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(220, 53, 69, 0); } 100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); } }
        .sos-marker {
            background-color: #dc3545; border: 2px solid white; border-radius: 50%;
            color: white; font-weight: 700; font-size: 14px; text-align: center; line-height: 28px;
            box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); animation: pulseRed 1.4s infinite;
        }
        .punto-paciente { 
            background-color: #dc3545; border: 2px solid white; border-radius: 50%; 
            animation: pulseRed 1.5s infinite; color: white; font-weight: bold; 
            font-size: 9px; text-align: center; line-height: 22px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.5); z-index: 1000 !important;
        }
        .ruta-holografica { stroke-dasharray: 10, 15; animation: flowDash 1s linear infinite; }
        @keyframes flowDash { to { stroke-dashoffset: -25; } }
        .amb-icon { font-size: 32px; text-shadow: 2px 2px 5px rgba(0,0,0,0.8); text-align: center; z-index: 1001 !important; }
        .hosp-marker { background-color: white; border: 3px solid; border-radius: 50%; text-align: center; line-height: 24px; font-size: 16px; box-shadow: 0 3px 6px rgba(0,0,0,0.4); transition: all 0.5s ease; }
        .hosp-green { border-color: #2ecc71; }
        .hosp-orange { border-color: #f39c12; }
        .hosp-red { border-color: #e74c3c; }
        @keyframes glowTarget { 0% { box-shadow: 0 0 5px 2px rgba(52, 152, 219, 0.5); } 50% { box-shadow: 0 0 20px 8px rgba(52, 152, 219, 0.8); } 100% { box-shadow: 0 0 5px 2px rgba(52, 152, 219, 0.5); } }
        .hosp-target { border-color: #3498db !important; animation: glowTarget 1.5s infinite; z-index: 900 !important; transform: scale(1.2); }
        .ambu-marker { background-color: #e8f4f8; border: 2px solid #3498db; border-radius: 5px; text-align: center; line-height: 22px; font-size: 16px; }
        .custom-tip { font-family: Arial, sans-serif; font-size: 13px; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); border: none; text-align: center; }
        .traffic-tip { background-color: rgba(255,255,255,0.9); font-weight: bold; }
        .progress-bg { background: #e0e0e0; width: 100%; height: 8px; border-radius: 4px; margin-top: 4px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
        #hud { position: absolute; top: 14px; right: 14px; z-index: 1200; background: rgba(255,255,255,0.95); border-radius: 10px; padding: 10px 12px; border: 1px solid rgba(0,0,0,0.12); width: 320px; font-family: Arial, sans-serif; }
        .row { display: flex; justify-content: space-between; margin: 4px 0; font-size: 13px; }
        .alerts { margin-top: 8px; font-size: 12px; max-height: 90px; overflow: auto; }
    </style>
</head>
<body>
    <div id="hud">
      <div class="row"><b>Conductor Smart City</b><span>🚑</span></div>
      <div class="row"><span>Destino</span><b id="kDestino">-</b></div>
      <div class="row"><span>ETA</span><b id="kEta">-</b></div>
      <div class="row"><span>Distancia</span><b id="kDist">-</b></div>
      <div class="alerts"><b>Alertas</b><ul id="alerts"></ul></div>
    </div>
    <div id="map"></div>

    <script>
        const ambulatorios = __AMBULATORIOS__;
        const zonasTrafico = __ZONAS_TRAFICO__;
        const operativos = __OPERATIVOS__;
        const alerts = __ALERTS__;

        const map = L.map('map', {preferCanvas: true}).setView([40.4168, -3.7038], 13);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png').addTo(map);

        // Restore parent page scroll after auto-refresh to avoid visual jump.
        try {
            const savedY = window.parent.sessionStorage.getItem('conductorScrollY');
            if (savedY !== null) {
                window.parent.scrollTo(0, parseInt(savedY, 10));
                window.parent.sessionStorage.removeItem('conductorScrollY');
            }
        } catch (e) {}

        zonasTrafico.forEach(function(zona) {
            if (zona.nivel === "Bajo") return;
            const colorFondo = zona.nivel === "Alto" ? '#ff4d4d' : '#ffcc00';
            L.circle([zona.lat, zona.lon], { radius: zona.radio, color: colorFondo, fillColor: colorFondo, fillOpacity: 0.12, weight: 1, opacity: 0.25 })
              .bindTooltip("🚥 Tráfico: <b>" + zona.nivel + "</b>", { direction: 'center', className: 'custom-tip traffic-tip' })
              .addTo(map);
        });

        ambulatorios.forEach(function(a) {
            const icon = L.divIcon({className: 'ambu-marker', html: '🩺', iconSize: [26,26], iconAnchor: [13,13]});
            L.marker([a.lat, a.lon], {icon: icon}).bindTooltip("<b>Base SVB</b><br>" + a.nombre, {direction: 'top', className: 'custom-tip'}).addTo(map);
        });

        const alertsEl = document.getElementById('alerts');
        alerts.forEach((a) => {
          const li = document.createElement('li');
          li.textContent = a;
          alertsEl.appendChild(li);
        });

        const op = operativos[0];
        document.getElementById('kDestino').textContent = op.esperando_destino ? 'Esperando operador' : op.destino.nombre;

        const sosIcon = L.divIcon({className: 'sos-marker', html: 'SOS', iconSize: [28,28], iconAnchor: [14,14]});
        const markerSOS = L.marker([op.sos.lat, op.sos.lon], {icon: sosIcon}).addTo(map);
        markerSOS.bindTooltip('<b>Señal de socorro</b><br>' + op.sos.nombre, {direction: 'top', className: 'custom-tip'});

        const hospitalMarkers = {};
        op.hospitales.forEach(function(h) {
            const colorClass = (h.occ > 85 ? 'hosp-red' : (h.occ > 50 ? 'hosp-orange' : 'hosp-green'));
            const barColor = h.occ > 85 ? '#e74c3c' : (h.occ > 50 ? '#f39c12' : '#2ecc71');
            const icon = L.divIcon({className: 'hosp-marker ' + colorClass, html: '🏥', iconSize: [30,30], iconAnchor: [15,15]});
            const marker = L.marker([h.lat, h.lon], {icon: icon}).addTo(map);
            const tipHTML = `
              <div style="text-align: left;">
                <center><b>${h.nombre}</b></center><hr style="margin:4px 0;">
                <div>📍 <b>Dirección:</b><br>${h.direccion || 'No disponible'}</div>
                <div style="margin-top:4px;">🩺 <b>Especialidades:</b><br>${h.especialidades || 'No disponible'}</div>
                <div style="margin-top:4px;">🧠 <b>Perfiles:</b> ${h.perfiles || 'No definido'}</div>
                🛏️ Ocupación: <b>${h.occ}%</b>
                <div class="progress-bg"><div class="progress-fill" style="width: ${h.occ}%; background-color: ${barColor};"></div></div>
                <div style="margin-top:4px;">⏱️ Espera: <b>${h.wait} min</b></div>
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

                const routeLineSos = hasSosRoute ? L.polyline(gpsSos, {color: '#ff7a18', weight: 4, opacity: 0.55, dashArray: '10,10'}).addTo(map) : null;
                const routeLineHosp = hasHospRoute ? L.polyline(gpsHosp, {color: '#1a73e8', weight: 4, opacity: 0.5, dashArray: '8,10'}).addTo(map) : null;
                const doneLine = L.polyline([sosSuave[0]], {color: '#19e872', weight: 5, opacity: 0.9}).addTo(map);

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

                setTimeout(() => {
                    animarRuta(sosSuave, etaSos, () => {
                        markerAmb.setLatLng([op.sos.lat, op.sos.lon]);
                        doneLine.setLatLngs(sosSuave);
                        document.getElementById('kDist').textContent = '0 m';
                        if (!hasHospRoute) {
                            document.getElementById('kEta').textContent = '--';
                            if (op.auto_refresh_waiting) {
                                setTimeout(() => {
                                    try {
                                        window.parent.sessionStorage.setItem('conductorScrollY', String(window.parent.scrollY || 0));
                                    } catch (e) {}
                                    window.parent.location.reload();
                                }, 1500);
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

components.html(html_mapa, height=1000)
