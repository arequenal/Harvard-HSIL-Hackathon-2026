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

from visual import operator_service as op

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "madrid_grafo.graphml"
PROCESSED_HOSPITALES_PATH = (
    PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"
)
SAMUR_BASES_PATH = PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "bases_samur_madrid.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def map_generate_simulations(
    graph: nx.MultiDiGraph,
    hospitals_df: pd.DataFrame,
    bases: List[Dict[str, Any]],
    n_sims: int = 6,
) -> List[Dict[str, Any]]:
    nodes = list(graph.nodes())
    sims: List[Dict[str, Any]] = []

    for _ in range(n_sims):
        hosp_data = []
        for idx, row in hospitals_df.iterrows():
            hosp_data.append(
                {
                    "centro_id": str(row.get("centro_id", idx)),
                    "nombre": str(row.get("nombre", f"Hospital {row.get('centro_id', idx)}")),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "nodo_red": int(row["nodo_red"]),
                    "occ": random.randint(35, 98),
                    "wait": random.randint(8, 125),
                }
            )

        valid = False
        route_to_event: List[int] = []
        route_to_hospital: List[int] = []
        event_coords: List[float] = []
        origin_base: Dict[str, Any] | None = None
        best_hospital: Dict[str, Any] | None = None

        while not valid:
            emergency_node = random.choice(nodes)
            lat_em = float(graph.nodes[emergency_node]["y"])
            lon_em = float(graph.nodes[emergency_node]["x"])

            try:
                bases_sorted = sorted(bases, key=lambda b: (b["lat"] - lat_em) ** 2 + (b["lon"] - lon_em) ** 2)
                route_to_event = nx.shortest_path(
                    graph,
                    source=bases_sorted[0]["nodo_red"],
                    target=emergency_node,
                    weight="length",
                )

                min_cost = float("inf")
                for h in hosp_data:
                    try:
                        weighted_dist = nx.shortest_path_length(
                            graph,
                            source=emergency_node,
                            target=h["nodo_red"],
                            weight="weighted_length",
                        )
                        cost = weighted_dist + (h["occ"] * 45) + (h["wait"] * 30)
                        if cost < min_cost:
                            min_cost = cost
                            best_hospital = h
                    except nx.NetworkXNoPath:
                        continue

                if best_hospital is None:
                    continue

                route_to_hospital = nx.shortest_path(
                    graph,
                    source=emergency_node,
                    target=best_hospital["nodo_red"],
                    weight="weighted_length",
                )

                event_coords = [lat_em, lon_em]
                origin_base = bases_sorted[0]
                valid = True
            except nx.NetworkXNoPath:
                continue

        if origin_base is None or best_hospital is None:
            continue

        gps_ida = [[float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])] for n in route_to_event]
        gps_vuelta = [[float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])] for n in route_to_hospital]

        sims.append(
            {
                "hospitales": hosp_data,
                "emergencia": event_coords,
                "gps_ida": gps_ida,
                "gps_vuelta": gps_vuelta,
                "destino": best_hospital,
                "origen": origin_base,
            }
        )

    return sims


def map_render_driver_map(
    simulations: List[Dict[str, Any]],
    bases: List[Dict[str, Any]],
    incidents: List[Dict[str, Any]],
    hospitals_count: int,
) -> None:
    sim_json = json.dumps(simulations)
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
    html, body, #map {{ height: 100%; width: 100%; margin: 0; }}
    #hud {{
      position: absolute; top: 14px; right: 14px; z-index: 1200;
      background: rgba(255,255,255,0.96); border-radius: 10px; padding: 10px 12px;
      border: 1px solid rgba(0,0,0,0.1); box-shadow: 0 4px 10px rgba(0,0,0,0.12);
      font-family: Arial, sans-serif; width: 320px;
    }}
    .title {{ font-weight: 700; font-size: 13px; color: #0b5cab; margin-bottom: 8px; text-transform: uppercase; }}
    .kpi {{ font-size: 13px; margin: 4px 0; display:flex; justify-content: space-between; }}
    .kpi b {{ color: #1a1a1a; }}
    .amb {{ font-size: 28px; text-align:center; }}
    .hosp {{ font-size: 18px; }}
  </style>
</head>
<body>
  <div id=\"hud\">
    <div class=\"title\">Mapa Operativo</div>
    <div class=\"kpi\"><span>Hospitales activos</span><b>{hospitals_count}</b></div>
    <div class=\"kpi\"><span>Bases SAMUR</span><b id=\"kBases\">0</b></div>
    <div class=\"kpi\"><span>Escenario</span><b id=\"kEsc\">1</b></div>
    <div class=\"kpi\"><span>Destino</span><b id=\"kDestino\">-</b></div>
    <div class=\"kpi\"><span>ETA estimada</span><b id=\"kEta\">-</b></div>
    <div class=\"kpi\"><span>Estado vial</span><b id=\"kVial\">Con incidencias</b></div>
  </div>
  <div id=\"map\"></div>

  <script>
    const map = L.map('map', {{ preferCanvas: true }}).setView([40.4168, -3.7038], 12.8);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png').addTo(map);

    const sims = {sim_json};
    const bases = {bases_json};
    const incidents = {incidents_json};

    document.getElementById('kBases').textContent = String(bases.length);

    const incidentStyle = {{ alto: '#e74c3c', medio: '#f39c12', bajo: '#3498db' }};

    incidents.forEach((inc) => {{
      const color = incidentStyle[inc.nivel] || '#888';
      L.circle([inc.lat, inc.lon], {{ radius: inc.radio, color, fillColor: color, fillOpacity: 0.16, weight: 2 }})
        .bindTooltip(`<b>${{inc.tipo}}</b><br>${{inc.detalle}}`, {{ direction: 'top' }})
        .addTo(map);
    }});

    bases.forEach((b) => {{
      const icon = L.divIcon({{ className: 'amb', html: '🚑', iconSize:[28,28], iconAnchor:[14,14] }});
      L.marker([b.lat, b.lon], {{ icon }}).bindTooltip(`<b>${{b.nombre}}</b>`).addTo(map);
    }});

    let markers = [];
    let lines = [];

    function clearLayers() {{
      markers.forEach((m) => map.removeLayer(m));
      lines.forEach((l) => map.removeLayer(l));
      markers = [];
      lines = [];
    }}

    function renderScenario(idx) {{
      if (!sims.length) return;
      clearLayers();
      const s = sims[idx % sims.length];

      s.hospitales.forEach((h) => {{
        const icon = L.divIcon({{ className: 'hosp', html: '🏥', iconSize:[20,20], iconAnchor:[10,10] }});
        const marker = L.marker([h.lat, h.lon], {{ icon }}).addTo(map)
          .bindTooltip(`<b>${{h.nombre}}</b><br>Ocupacion: ${{h.occ}}%<br>Espera: ${{h.wait}} min`);
        markers.push(marker);
      }});

      const eventMarker = L.marker([s.emergencia[0], s.emergencia[1]])
        .addTo(map)
        .bindTooltip('<b>Emergencia activa</b>');
      markers.push(eventMarker);

      const line1 = L.polyline(s.gps_ida, {{ color: '#dc3545', weight: 4, opacity: 0.8 }}).addTo(map);
      const line2 = L.polyline(s.gps_vuelta, {{ color: '#0b5cab', weight: 4, opacity: 0.8, dashArray: '10,8' }}).addTo(map);
      lines.push(line1); lines.push(line2);

      const ambIcon = L.divIcon({{ className: 'amb', html: '🚑', iconSize:[30,30], iconAnchor:[15,15] }});
      const amb = L.marker(s.gps_ida[0], {{ icon: ambIcon }}).addTo(map).bindTooltip('Unidad en ruta').openTooltip();
      markers.push(amb);

      const routeLen = s.gps_vuelta.length;
      const eta = Math.max(6, Math.round(routeLen / 14));
      document.getElementById('kEsc').textContent = String((idx % sims.length) + 1);
      document.getElementById('kDestino').textContent = s.destino.nombre;
      document.getElementById('kEta').textContent = `${{eta}} min`;

      if (s.gps_vuelta.length) {{
        const bounds = L.latLngBounds(s.gps_vuelta.concat(s.gps_ida));
        map.fitBounds(bounds.pad(0.2));
      }}
    }}

    let current = 0;
    renderScenario(current);
    setInterval(() => {{
      current = (current + 1) % Math.max(1, sims.length);
      renderScenario(current);
    }}, 12000);
  </script>
</body>
</html>
"""
    components.html(html, height=760)


def main() -> None:
    op.main()

    st.markdown("---")
    st.markdown("### Mapa operativo (prototipo ambulancia)")
    st.write(
        "Visualizacion grande de rutas de ambulancia, incidencias urbanas y destino recomendado para seguimiento operativo."
    )

    graph = map_load_graph_with_traffic()
    hospitals_df = map_load_hospitals(graph)
    bases = map_load_samur_bases(graph)

    if hospitals_df.empty:
        st.error("No se pudieron cargar hospitales con coordenadas validas para el mapa.")
        return

    if not bases:
        st.error("No se pudieron cargar bases SAMUR validas para el mapa.")
        return

    if "op_mapa_simulations" not in st.session_state:
        st.session_state["op_mapa_simulations"] = map_generate_simulations(graph, hospitals_df, bases, n_sims=7)

    incidents = map_get_scenario_incidents()

    map_render_driver_map(
        simulations=st.session_state["op_mapa_simulations"],
        bases=bases,
        incidents=incidents,
        hospitals_count=int(hospitals_df["centro_id"].nunique()) if "centro_id" in hospitals_df.columns else len(hospitals_df),
    )

    st.markdown("### Incidencias simuladas activas")
    st.dataframe(pd.DataFrame(incidents), use_container_width=True)


if __name__ == "__main__":
    main()
