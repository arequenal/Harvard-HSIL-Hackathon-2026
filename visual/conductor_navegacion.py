from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import osmnx as ox
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.dispatch_shared import load_state

GRAPH_PATH = Path(__file__).resolve().parent / "madrid_grafo.graphml"
HOSPITALES_PATH = (
    PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"
)
BASES_PATH = PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "bases_samur_madrid.csv"


st.set_page_config(page_title="AmbulancIA - Conductor Navegacion", layout="wide")


@st.cache_resource
def load_graph() -> nx.MultiDiGraph:
    graph = ox.load_graphml(GRAPH_PATH)
    return graph


@st.cache_data
def load_hospitals(_graph: nx.MultiDiGraph) -> pd.DataFrame:
    df = pd.read_csv(HOSPITALES_PATH, sep=";")
    for col in ["lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    if "nombre" not in df.columns:
        df["nombre"] = df["centro_id"].astype(str)

    df["nodo_red"] = df.apply(
        lambda row: int(ox.distance.nearest_nodes(_graph, X=float(row["lon"]), Y=float(row["lat"]))),
        axis=1,
    )
    return df


@st.cache_data
def load_bases(_graph: nx.MultiDiGraph) -> pd.DataFrame:
    if not BASES_PATH.exists():
        base = pd.DataFrame(
            [
                {
                    "nombre": "Base Central",
                    "lat": 40.4117,
                    "lon": -3.7430,
                }
            ]
        )
    else:
        base = pd.read_csv(BASES_PATH, sep=";")
        base = base.rename(columns={"latitud": "lat", "longitud": "lon"})
        for col in ["lat", "lon"]:
            base[col] = pd.to_numeric(base[col], errors="coerce")
        base = base[base["lat"].notna() & base["lon"].notna()].copy()

    base["nodo_red"] = base.apply(
        lambda row: int(ox.distance.nearest_nodes(_graph, X=float(row["lon"]), Y=float(row["lat"]))),
        axis=1,
    )
    return base


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    angle = math.degrees(math.atan2(x, y))
    return (angle + 360.0) % 360.0


def normalize_delta(delta: float) -> float:
    d = (delta + 180.0) % 360.0 - 180.0
    return d


def build_turn_message(prev_b: float, next_b: float, dist_m: float) -> str:
    delta = normalize_delta(next_b - prev_b)
    if delta > 30:
        action = "Gira a la derecha"
    elif delta < -30:
        action = "Gira a la izquierda"
    else:
        action = "Sigue recto"

    if dist_m >= 1000:
        dist_txt = f"en {dist_m / 1000:.1f} km"
    else:
        dist_txt = f"en {int(dist_m)} m"

    return f"{action} {dist_txt}"


def route_length_m(graph: nx.MultiDiGraph, route: List[int]) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        edge_data = graph.get_edge_data(route[i], route[i + 1])
        if not edge_data:
            continue
        best = min(edge_data.values(), key=lambda d: float(d.get("length", 1.0)))
        total += float(best.get("length", 1.0))
    return total


def build_route_and_instructions(
    graph: nx.MultiDiGraph,
    bases_df: pd.DataFrame,
    destination_row: pd.Series,
) -> Dict[str, Any]:
    base_row = bases_df.iloc[0]
    try:
        route_nodes = nx.shortest_path(
            graph,
            source=int(base_row["nodo_red"]),
            target=int(destination_row["nodo_red"]),
            weight="length",
        )
    except nx.NetworkXNoPath:
        return {
            "coords": [[float(base_row["lat"]), float(base_row["lon"])], [float(destination_row["lat"]), float(destination_row["lon"])]],
            "length_m": 6000.0,
            "instructions": [{"idx": 0, "text": "Inicia ruta hacia el hospital"}],
            "origin": {"lat": float(base_row["lat"]), "lon": float(base_row["lon"]), "nombre": str(base_row.get("nombre", "Base SAMUR"))},
        }

    coords = [[float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])] for n in route_nodes]
    length_m = route_length_m(graph, route_nodes)

    instructions: List[Dict[str, Any]] = [{"idx": 0, "text": "Inicia ruta"}]
    if len(coords) >= 3:
        step = max(4, len(coords) // 10)
        for idx in range(step, len(coords) - 1, step):
            prev_pt = (coords[idx - 1][0], coords[idx - 1][1])
            curr_pt = (coords[idx][0], coords[idx][1])
            next_pt = (coords[min(idx + 1, len(coords) - 1)][0], coords[min(idx + 1, len(coords) - 1)][1])

            b1 = bearing_deg(prev_pt, curr_pt)
            b2 = bearing_deg(curr_pt, next_pt)
            remaining_ratio = max(0.0, 1.0 - (idx / max(1, len(coords) - 1)))
            remaining_m = length_m * remaining_ratio
            instructions.append({"idx": idx, "text": build_turn_message(b1, b2, remaining_m)})

    instructions.append({"idx": max(0, len(coords) - 2), "text": "Mantente en el carril: llegada inminente"})

    return {
        "coords": coords,
        "length_m": length_m,
        "instructions": instructions,
        "origin": {
            "lat": float(base_row["lat"]),
            "lon": float(base_row["lon"]),
            "nombre": str(base_row.get("nombre", "Base SAMUR")),
        },
    }


def render_navigation_map(route_data: Dict[str, Any], destination: Dict[str, Any], alerts: List[str], eta_min: int) -> None:
    coords_json = json.dumps(route_data["coords"])
    instructions_json = json.dumps(route_data["instructions"])
    alerts_json = json.dumps(alerts)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <style>
    html, body, #map {{ height: 100%; width: 100%; margin: 0; }}
    #hud {{
      position: absolute; top: 14px; right: 14px; z-index: 1200;
      background: rgba(255,255,255,0.96); border-radius: 12px; padding: 12px 14px;
      border: 1px solid rgba(0,0,0,0.14); box-shadow: 0 6px 14px rgba(0,0,0,0.14);
      font-family: Arial, sans-serif; width: 340px;
    }}
    .title {{ font-weight: 700; font-size: 13px; color: #0b5cab; margin-bottom: 8px; text-transform: uppercase; }}
    .kpi {{ font-size: 13px; margin: 4px 0; display: flex; justify-content: space-between; gap: 10px; }}
    .instruction {{ margin-top: 8px; font-size: 15px; font-weight: 700; color: #111; line-height: 1.25; }}
    .alerts {{ margin-top: 8px; max-height: 100px; overflow: auto; font-size: 12px; color: #a33; }}
    .alerts li {{ margin-bottom: 4px; }}
    .amb {{ font-size: 30px; }}
  </style>
</head>
<body>
  <div id=\"hud\">
    <div class=\"title\">Conductor · Navegacion asistida</div>
    <div class=\"kpi\"><span>Destino</span><b>{destination['nombre']}</b></div>
    <div class=\"kpi\"><span>Direccion</span><b>{destination['direccion']}</b></div>
    <div class=\"kpi\"><span>ETA</span><b id=\"eta\">{eta_min} min</b></div>
    <div class=\"instruction\" id=\"instruction\">Inicia ruta</div>
    <div class=\"alerts\"><b>Alertas de via</b><ul id=\"alerts\"></ul></div>
  </div>
  <div id=\"map\"></div>

  <script>
    const coords = {coords_json};
    const instructions = {instructions_json};
    const alerts = {alerts_json};

    const map = L.map('map', {{ preferCanvas: true, zoomControl: false }}).setView(coords[0], 16);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png').addTo(map);

    const routeLine = L.polyline(coords, {{ color: '#1a73e8', weight: 7, opacity: 0.9 }}).addTo(map);
    const ambIcon = L.divIcon({{ className: 'amb', html: '🚑', iconSize:[34,34], iconAnchor:[17,17] }});
    const marker = L.marker(coords[0], {{ icon: ambIcon }}).addTo(map);
    L.marker(coords[coords.length - 1]).addTo(map).bindTooltip('Hospital destino');

    const alertsEl = document.getElementById('alerts');
    alerts.forEach((a) => {{
      const li = document.createElement('li');
      li.textContent = a;
      alertsEl.appendChild(li);
    }});

    function nextInstructionText(idx) {{
      let txt = 'Sigue recto';
      for (const it of instructions) {{
        if (idx <= it.idx) {{
          txt = it.text;
          break;
        }}
      }}
      return txt;
    }}

    let i = 0;
    let eta = {eta_min};
    map.fitBounds(routeLine.getBounds().pad(0.2));

    const tick = () => {{
      if (!coords.length) return;
      i = Math.min(i + 1, coords.length - 1);
      marker.setLatLng(coords[i]);
      map.setView(coords[i], 17, {{ animate: true, duration: 0.8 }});

      document.getElementById('instruction').textContent = nextInstructionText(i);
      const ratio = 1 - i / Math.max(1, coords.length - 1);
      const etaNow = Math.max(1, Math.round(eta * ratio));
      document.getElementById('eta').textContent = `${{etaNow}} min`;
    }};

    setInterval(tick, 1200);
  </script>
</body>
</html>
"""

    components.html(html, height=760)


def main() -> None:
    st.title("AmbulancIA - Conductor (Navegacion)")
    st.caption("Interfaz tipo navegador: mapa en movimiento, indicaciones de giro, ETA y alertas de trafico/obras.")

    graph = load_graph()
    hospitals = load_hospitals(graph)
    bases = load_bases(graph)
    state = load_state()

    destination_id = str(state.get("destination", {}).get("centro_id", "")).strip()
    destination_name = str(state.get("destination", {}).get("nombre", "")).strip()
    alerts = [str(a) for a in state.get("traffic_alerts", []) if str(a).strip()]
    if not alerts:
        alerts = [
            "Atasco moderado en la via principal",
            "Obras puntuales en acceso secundario",
        ]

    if destination_id:
        candidate = hospitals[hospitals["centro_id"].astype(str) == destination_id]
        if not candidate.empty:
            dest_row = candidate.iloc[0]
        else:
            dest_row = hospitals.iloc[0]
    else:
        dest_row = hospitals.iloc[0]

    route_data = build_route_and_instructions(graph, bases, dest_row)

    if route_data["length_m"] > 0:
        computed_eta = max(4, int(round(route_data["length_m"] / 1000 * 2.2)))
    else:
        computed_eta = 10

    eta_min = int(state.get("destination", {}).get("eta_min") or computed_eta)
    destination_info = {
        "centro_id": str(dest_row.get("centro_id", "")),
        "nombre": destination_name or str(dest_row.get("nombre", "Hospital destino")),
        "direccion": str(dest_row.get("direccion_completa", state.get("destination", {}).get("direccion", ""))),
        "telefono": str(dest_row.get("telefono", state.get("destination", {}).get("telefono", ""))),
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Caso", str(state.get("case", {}).get("summary", "Sin caso activo")))
    with c2:
        st.metric("Destino", destination_info["nombre"])
    with c3:
        st.metric("ETA", f"{eta_min} min")

    st.markdown("**Siguiente indicacion**")
    if route_data["instructions"]:
        st.info(route_data["instructions"][0]["text"])
    else:
        st.info("Inicia ruta")

    st.markdown("**Alertas activas**")
    for al in alerts[:5]:
        st.warning(al)

    render_navigation_map(route_data, destination_info, alerts, eta_min)


if __name__ == "__main__":
    main()
