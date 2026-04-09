from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import osmnx as ox
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.dispatch_shared import load_state, update_state

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "madrid_grafo.graphml"
PROCESSED_HOSPITALES_PATH = (
    PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"
)
SAMUR_BASES_PATH = PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "bases_samur_madrid.csv"


st.set_page_config(page_title="AmbulancIA - Conductor", layout="wide")

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
            .soft {
                border: 1px solid rgba(88,166,255,0.15);
                background: #1a1f3a;
                border-radius: 10px;
                padding: 10px 12px;
                color: #e0e0e0;
            }
            .exp-box {
                border-left: 4px solid #3fb950;
                background: #0d3f1a;
                padding: 10px 12px;
                border-radius: 8px;
                margin-bottom: 8px;
                color: #7ee787;
            }
        </style>
        """,
        unsafe_allow_html=True,
)


@st.cache_resource
def load_graph_with_traffic() -> nx.MultiDiGraph:
    graph = ox.load_graphml(GRAPH_PATH)
    for _, _, _, data in graph.edges(keys=True, data=True):
        tf = random.uniform(1.0, 3.0)
        data["traffic_factor"] = tf
        data["weighted_length"] = data.get("length", 1.0) * tf
    return graph


@st.cache_data
def load_hospitals(_graph: nx.MultiDiGraph) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_HOSPITALES_PATH, sep=";")
    if "centro_tipo" in df.columns:
        df = df[df["centro_tipo"].isin({"Hospital general", "Hospital especializado"})].copy()
    for col in ["lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    df["nodo_red"] = df.apply(
        lambda row: int(ox.distance.nearest_nodes(_graph, X=float(row["lon"]), Y=float(row["lat"]))), axis=1
    )
    if "nombre" not in df.columns:
        df["nombre"] = "Hospital"
    return df


@st.cache_data
def load_samur_bases(_graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
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


def compute_route(
    graph: nx.MultiDiGraph,
    bases: List[Dict[str, Any]],
    hospitals: pd.DataFrame,
    destination_centro_id: str,
) -> Dict[str, Any]:
    nodes = list(graph.nodes())
    emergency_node = random.choice(nodes)
    lat_em = float(graph.nodes[emergency_node]["y"])
    lon_em = float(graph.nodes[emergency_node]["x"])

    base = min(bases, key=lambda b: (b["lat"] - lat_em) ** 2 + (b["lon"] - lon_em) ** 2)
    route_to_em = nx.shortest_path(graph, source=base["nodo_red"], target=emergency_node, weight="length")

    destination = None
    if destination_centro_id:
        filtered = hospitals[hospitals["centro_id"].astype(str) == str(destination_centro_id)]
        if not filtered.empty:
            destination = filtered.iloc[0]
    if destination is None:
        destination = hospitals.iloc[0]

    route_to_hosp = nx.shortest_path(
        graph,
        source=emergency_node,
        target=int(destination["nodo_red"]),
        weight="weighted_length",
    )

    gps_ida = [[float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])] for n in route_to_em]
    gps_hosp = [[float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])] for n in route_to_hosp]

    return {
        "emergency": [lat_em, lon_em],
        "base": base,
        "destination": {
            "centro_id": str(destination.get("centro_id", "")),
            "nombre": str(destination.get("nombre", "Hospital")),
            "lat": float(destination["lat"]),
            "lon": float(destination["lon"]),
        },
        "gps_ida": gps_ida,
        "gps_hosp": gps_hosp,
    }


def get_traffic_incidents() -> List[Dict[str, Any]]:
    return [
        {
            "tipo": "Atasco severo",
            "lat": 40.4215,
            "lon": -3.6590,
            "radio": 1300,
            "nivel": "alto",
            "detalle": "Retencion por hora punta",
        },
        {
            "tipo": "Obras",
            "lat": 40.3920,
            "lon": -3.6850,
            "radio": 900,
            "nivel": "medio",
            "detalle": "Carril reducido",
        },
        {
            "tipo": "Evento masivo",
            "lat": 40.4531,
            "lon": -3.6883,
            "radio": 1000,
            "nivel": "medio",
            "detalle": "Cortes intermitentes",
        },
    ]


def render_map(
    route_data: Dict[str, Any],
    alerts: List[str],
    hospitals: pd.DataFrame,
    bases: List[Dict[str, Any]],
    incidents: List[Dict[str, Any]],
) -> None:
    hospitals_payload = [
        {
            "nombre": str(row.get("nombre", "Hospital")),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }
        for _, row in hospitals.iterrows()
    ]

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; width: 100%; margin: 0; }}
    #panel {{
      position:absolute; top:12px; right:12px; z-index:1200;
      background: rgba(255,255,255,0.97); border-radius: 12px; padding: 10px 12px;
      width: 320px; font-family: Arial, sans-serif; border:1px solid rgba(0,0,0,0.1);
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
        .title {{font-weight:700;color:#0b5cab;text-transform:uppercase;font-size:13px;margin-bottom:8px;}}
    .item {{font-size:13px; margin: 4px 0; display:flex; justify-content:space-between;}}
    .alert {{font-size:12px; background:#fff3e0; border-left:4px solid #f39c12; padding:6px 8px; margin-top:6px; border-radius:6px;}}
        .navbox {{
            position:absolute; bottom:16px; left:50%; transform:translateX(-50%); z-index:1200;
            background:rgba(11,92,171,0.95); color:#fff; font-family:Arial,sans-serif;
            border-radius:14px; padding:10px 14px; min-width:380px; text-align:center;
            box-shadow:0 5px 14px rgba(0,0,0,0.25);
        }}
        .navbox .main {{font-size:15px; font-weight:700;}}
        .navbox .sub {{font-size:12px; opacity:0.95; margin-top:2px;}}
  </style>
</head>
<body>
  <div id="panel">
    <div class="title">Interfaz Conductor</div>
    <div class="item"><span>Destino</span><b>{route_data['destination']['nombre']}</b></div>
    <div class="item"><span>Ruta</span><b>Activa</b></div>
    <div class="item"><span>Estado</span><b>En camino</b></div>
    {''.join([f'<div class="alert">⚠️ {a}</div>' for a in alerts])}
  </div>
    <div class="navbox">
        <div class="main" id="navMain">Iniciando navegacion hacia {route_data['destination']['nombre']}</div>
        <div class="sub" id="navSub">Modo demo GPS en primera persona activo</div>
    </div>
  <div id="map"></div>
  <script>
        const map = L.map('map').setView([40.4168, -3.7038], 12.8);
        // Estilo mas cercano a navegador (carreteras y etiquetas claras).
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19}}).addTo(map);

    const data = {json.dumps(route_data)};
        const hospitals = {json.dumps(hospitals_payload)};
        const bases = {json.dumps(bases)};
        const incidents = {json.dumps(incidents)};

        const baseIcon = L.divIcon({{html:'🚑', className:'', iconSize:[26,26], iconAnchor:[13,13]}});
        const emIcon = L.divIcon({{html:'🆘', className:'', iconSize:[24,24], iconAnchor:[12,12]}});
        const hosIcon = L.divIcon({{html:'🏥', className:'', iconSize:[16,16], iconAnchor:[8,8]}});
        const targetHosIcon = L.divIcon({{html:'🏥🏁', className:'', iconSize:[26,26], iconAnchor:[13,13]}});

        const incColor = {{alto:'#e74c3c', medio:'#f39c12', bajo:'#3498db'}};
        incidents.forEach((inc) => {{
            const color = incColor[inc.nivel] || '#888';
            L.circle([inc.lat, inc.lon], {{radius: inc.radio, color: color, fillColor: color, fillOpacity: 0.16, weight: 2}})
                .bindTooltip(`<b>${{inc.tipo}}</b><br>${{inc.detalle}}`)
                .addTo(map);
        }});

        hospitals.forEach((h) => {{
            const isTarget = h.nombre === data.destination.nombre;
            L.marker([h.lat, h.lon], {{icon: isTarget ? targetHosIcon : hosIcon}})
                .addTo(map)
                .bindTooltip(`<b>${{h.nombre}}</b>`);
        }});

        bases.forEach((b) => {{
            L.marker([b.lat, b.lon], {{icon: baseIcon}})
                .addTo(map)
                .bindTooltip(`<b>${{b.nombre}}</b><br>Base SAMUR`);
        }});

    L.marker([data.emergency[0], data.emergency[1]], {{icon: emIcon}}).addTo(map).bindTooltip('Emergencia');

                const ida = L.polyline(data.gps_ida, {{color:'#dc3545', weight:5, opacity:0.88}}).addTo(map);
                const hos = L.polyline(data.gps_hosp, {{color:'#1e88e5', weight:6, opacity:0.90}}).addTo(map);

        const fullPath = data.gps_ida.concat(data.gps_hosp);
        const ambIcon = L.divIcon({{html:'🚑', className:'', iconSize:[30,30], iconAnchor:[15,15]}});
        const ambMarker = L.marker(fullPath[0], {{icon: ambIcon}}).addTo(map);

        const bounds = L.latLngBounds(fullPath);
        map.fitBounds(bounds.pad(0.20));

        let idx = 0;
        const navMain = document.getElementById('navMain');
        const navSub = document.getElementById('navSub');
        const total = fullPath.length;

        function animateGps() {{
            if (!fullPath.length) return;
            const p = fullPath[idx];
            ambMarker.setLatLng(p);

            // Modo primera persona: la camara sigue a la ambulancia.
            map.setView(p, 16.8, {{animate: true, duration: 0.6}});

            const remaining = Math.max(0, total - idx - 1);
            const eta = Math.max(1, Math.round(remaining / 18));
            navMain.textContent = idx < data.gps_ida.length
                ? 'Dirigete al punto de emergencia'
                : 'Traslado del paciente a hospital destino';
            navSub.textContent = `Distancia restante estimada: ${{remaining}} segmentos · ETA aprox: ${{eta}} min`;

            idx += 1;
            if (idx >= total) idx = total - 1;
        }}

        setInterval(animateGps, 500);
        animateGps();
  </script>
</body>
</html>
"""
    components.html(html, height=760)


def main() -> None:
    st.title("AmbulancIA - Servicio Conductor")
    st.caption("Recibe destino y alertas del servicio operador. Usa actualizar para refrescar cambios.")

    graph = load_graph_with_traffic()
    hospitals = load_hospitals(graph)
    bases = load_samur_bases(graph)

    if hospitals.empty:
        st.error("No se cargaron hospitales")
        st.stop()

    state = load_state()

    current_version = int(state.get("version", 0))
    last_seen = int(st.session_state.get("driver_seen_version", -1))
    if current_version > last_seen:
        st.session_state["driver_seen_version"] = current_version
        if current_version > 0:
            st.success(f"Nueva actualizacion del operador recibida (version {current_version})")

    top_cols = st.columns([1, 1, 1, 1])
    top_cols[0].metric("Version", current_version)
    top_cols[1].metric("Urgencia", state.get("case", {}).get("urgencia", "No definida"))
    top_cols[2].metric("Especialidad", state.get("case", {}).get("especialidad", "No definida"))
    top_cols[3].metric("ETA", str(state.get("destination", {}).get("eta_min", "-") or "-") + " min")

    st.markdown('<div class="section-title">Resumen clinico recibido</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft"><b>Resumen caso:</b> ' + str(state.get("case", {}).get("summary", "")) + '</div>', unsafe_allow_html=True)

    exp_urg = str(state.get("case", {}).get("explicacion_urgencia", "")).strip()
    exp_spec = str(state.get("case", {}).get("explicacion_especialidad", "")).strip()

    st.markdown('<div class="section-title">Explicabilidad de la decision</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="exp-box"><b>Urgencia:</b> ' + (exp_urg if exp_urg else 'No disponible aun. Esperando publicacion del operador.') + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="exp-box"><b>Especialidad:</b> ' + (exp_spec if exp_spec else 'No disponible aun. Esperando publicacion del operador.') + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="soft"><b>Notas del operador:</b> ' + str(state.get("operator_notes", "")) + '</div>', unsafe_allow_html=True)

    with st.expander("Contacto hospital", expanded=True):
        dest = state.get("destination", {})
        st.write("**Hospital:**", dest.get("nombre", ""))
        st.write("**Telefono:**", dest.get("telefono", ""))
        st.write("**Email:**", dest.get("email", ""))
        st.write("**Direccion:**", dest.get("direccion", ""))

    incidents = get_traffic_incidents()

    route_data = compute_route(
        graph,
        bases,
        hospitals,
        destination_centro_id=str(state.get("destination", {}).get("centro_id", "")),
    )

    combined_alerts = list(state.get("traffic_alerts", []))
    combined_alerts.extend([f"{i['tipo']}: {i['detalle']}" for i in incidents[:2]])

    render_map(
        route_data,
        alerts=combined_alerts,
        hospitals=hospitals,
        bases=bases,
        incidents=incidents,
    )

    cols = st.columns([1, 1, 1])
    if cols[0].button("Actualizar estado", use_container_width=True):
        st.rerun()
    if cols[1].button("Marcar alertas revisadas", use_container_width=True):
        update_state({"traffic_alerts": []})
        st.success("Alertas limpiadas")
    if cols[2].button("Avisar incidencia nueva", use_container_width=True):
        existing = list(state.get("traffic_alerts", []))
        existing.append("Incidencia detectada por conductor: posible cierre parcial de via")
        update_state({"traffic_alerts": existing})
        st.success("Incidencia enviada al operador")


if __name__ == "__main__":
    main()
