from __future__ import annotations

import json
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

from visual import operator_service as op
from visual.dispatch_shared import load_state

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "madrid_grafo.graphml"
PROCESSED_HOSPITALES_PATH = (
    PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"
)
SAMUR_BASES_PATH = PROJECT_ROOT / "analisis_datos" / "data" / "processed" / "bases_samur_madrid.csv"

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
    destination_id = str(destination.get("centro_id", "")).strip().upper()
    destination_name = str(destination.get("nombre", "")).strip().lower()
    specialty_name = str(case_info.get("especialidad", "")).strip()

    def _normalize_text(value: Any) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()

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

    if candidate.empty:
        # Funcion objetivo multicriterio para seleccionar hospital cuando no hay destino publicado.
        # Minimiza tiempo, espera y ocupacion, y favorece hospitales con mas medicos.
        best_idx = None
        best_obj = None
        origin_base = bases[0]

        search_space = specialty_candidates if not specialty_candidates.empty else hospitals_df
        for idx, row in search_space.iterrows():
            target_node_i = int(row["nodo_red"])
            base_i, travel_i = _best_base_and_travel(target_node_i)
            if travel_i == float("inf"):
                continue

            m = _metrics_for_row(row)
            travel_min = travel_i / 550.0
            if not specialty_candidates.empty:
                objective = (travel_min, m["wait"], m["occ"], -m["doctors"])
            else:
                objective = ((0.52 * travel_min) + (0.26 * m["wait"]) + (0.20 * m["occ"]) - (0.42 * m["doctors"]),)
            if best_obj is None or objective < best_obj:
                best_obj = objective
                best_idx = idx
                origin_base = base_i

        target_row = hospitals_df.loc[best_idx] if best_idx is not None else hospitals_df.iloc[0]
        if not specialty_candidates.empty:
            status = f"Destino recomendado por especialidad y cercanía: {specialty_name}"
        else:
            status = "Destino recomendado por funcion objetivo"
    else:
        target_row = candidate.iloc[0]
        status = "Destino sincronizado con conductor"

    target_node = int(target_row["nodo_red"])
    if not candidate.empty:
        origin_base, _ = _best_base_and_travel(target_node)

    route_nodes: List[int] | None = None
    route_graphs: List[nx.MultiDiGraph | nx.Graph] = [graph, graph.to_undirected()]
    for g in route_graphs:
        for weight in ("weighted_length", "length"):
            try:
                route_nodes = nx.shortest_path(g, source=origin_base["nodo_red"], target=target_node, weight=weight)
                break
            except nx.NetworkXNoPath:
                continue
        if route_nodes:
            break

    if route_nodes:
        route_coords = [[float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])] for n in route_nodes]
    else:
        route_coords = [[float(origin_base["lat"]), float(origin_base["lon"])], [float(target_row["lat"]), float(target_row["lon"])]]

    final_point = [float(target_row["lat"]), float(target_row["lon"])]
    if route_coords[-1] != final_point:
        route_coords.append(final_point)

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
        html, body, #map {{ height: 100%; width: 100%; margin: 0; padding: 0; background-color: #f4f4f4; overflow: hidden; }}
        @keyframes pulseRed {{ 0% {{ box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }} 70% {{ box-shadow: 0 0 0 15px rgba(220, 53, 69, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }} }}
        @keyframes glowTarget {{ 0% {{ box-shadow: 0 0 5px 2px rgba(52, 152, 219, 0.5); }} 50% {{ box-shadow: 0 0 20px 8px rgba(52, 152, 219, 0.8); }} 100% {{ box-shadow: 0 0 5px 2px rgba(52, 152, 219, 0.5); }} }}
        .sos-marker {{
            background-color: #dc3545; border: 2px solid white; border-radius: 50%;
            color: white; font-weight: 700; font-size: 12px; text-align: center; line-height: 28px;
            box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); animation: pulseRed 1.4s infinite;
        }}
        .ambu-marker {{ background-color: #e8f4f8; border: 2px solid #3498db; border-radius: 5px; text-align: center; line-height: 22px; font-size: 16px; }}
        .amb-icon {{ font-size: 32px; text-shadow: 2px 2px 5px rgba(0,0,0,0.8); text-align: center; z-index: 1001 !important; }}
        .hosp-marker {{ background-color: white; border: 3px solid; border-radius: 50%; text-align: center; line-height: 24px; font-size: 16px; box-shadow: 0 3px 6px rgba(0,0,0,0.4); transition: all 0.5s ease; }}
        .hosp-green {{ border-color: #2ecc71; }}
        .hosp-orange {{ border-color: #f39c12; }}
        .hosp-red {{ border-color: #e74c3c; }}
        .hosp-target {{ border-color: #3498db !important; animation: glowTarget 1.5s infinite; z-index: 900 !important; transform: scale(1.2); }}
        .custom-tip {{ font-family: Arial, sans-serif; font-size: 13px; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); border: none; text-align: center; }}
        .traffic-tip {{ background-color: rgba(255,255,255,0.9); font-weight: bold; }}
        .progress-bg {{ background: #e0e0e0; width: 100%; height: 8px; border-radius: 4px; margin-top: 4px; overflow: hidden; }}
        .progress-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
        .hospital-popup {{ min-width: 280px; max-width: 340px; font-family: Arial, sans-serif; }}
        .hospital-popup .h-title {{ font-size: 14px; font-weight: 700; margin-bottom: 6px; color: #11324d; }}
        .hospital-popup .h-meta {{ color: #4a6278; font-size: 12px; margin-bottom: 8px; }}
        .hospital-popup .h-row {{ margin: 6px 0; font-size: 12px; color: #22384a; }}
        .hospital-popup .h-label {{ font-weight: 700; color: #0f4f81; }}
        .hospital-popup .h-box {{ margin-top: 6px; padding: 8px; border-radius: 8px; background: #f5f9fd; border: 1px solid #d7e8f6; max-height: 92px; overflow: auto; }}
        #hud {{
            position: absolute; top: 14px; right: 14px; z-index: 1200;
            background: rgba(255,255,255,0.95); border-radius: 10px; padding: 10px 12px;
            border: 1px solid rgba(0,0,0,0.12); width: 320px; font-family: Arial, sans-serif;
        }}
        .row {{ display: flex; justify-content: space-between; margin: 4px 0; font-size: 13px; }}
        .alerts {{ margin-top: 8px; font-size: 12px; max-height: 90px; overflow: auto; }}
  </style>
</head>
<body>
  <div id=\"hud\">
        <div class=\"row\"><b>Operador Smart City</b><span>🚑</span></div>
        <div class=\"row\"><span>Destino</span><b id=\"kDestino\">-</b></div>
        <div class=\"row\"><span>ETA</span><b id=\"kEta\">-</b></div>
        <div class=\"row\"><span>Distancia</span><b id=\"kDist\">-</b></div>
        <div class=\"row\"><span>Bases SAMUR</span><b id=\"kBases\">0</b></div>
        <div class=\"alerts\"><b>Alertas</b><ul id=\"alerts\"></ul></div>
  </div>
  <div id=\"map\"></div>

  <script>
    const map = L.map('map', {{ preferCanvas: true }}).setView([40.4168, -3.7038], 12.8);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png').addTo(map);

    const scenario = {scenario_json};
    const bases = {bases_json};
    const incidents = {incidents_json};

    document.getElementById('kBases').textContent = String(bases.length);

    const incidentStyle = {{ alto: '#e74c3c', medio: '#f39c12', bajo: '#3498db' }};

    incidents.forEach((inc) => {{
      const color = incidentStyle[inc.nivel] || '#888';
            L.circle([inc.lat, inc.lon], {{ radius: inc.radio, color, fillColor: color, fillOpacity: 0.12, weight: 1, opacity: 0.25 }})
                .bindTooltip(`🚥 Tráfico: <b>${{inc.tipo}}</b><br>${{inc.detalle}}`, {{ direction: 'center', className: 'custom-tip traffic-tip' }})
        .addTo(map);
    }});

        const alertsEl = document.getElementById('alerts');
        (scenario.alerts || []).forEach((a) => {{
            const li = document.createElement('li');
            li.textContent = a;
            alertsEl.appendChild(li);
        }});
        if (!alertsEl.children.length) {{
            const li = document.createElement('li');
            li.textContent = 'Sin alertas activas';
            alertsEl.appendChild(li);
        }};

    bases.forEach((b) => {{
            const icon = L.divIcon({{ className: 'ambu-marker', html: '🩺', iconSize:[26,26], iconAnchor:[13,13] }});
            L.marker([b.lat, b.lon], {{ icon }}).bindTooltip(`<b>Base SVB</b><br>${{b.nombre}}`, {{ direction: 'top', className: 'custom-tip' }}).addTo(map);
    }});

        const route = scenario.gps_ida || [];
        const sosIcon = L.divIcon({{ className: 'sos-marker', html: 'SOS', iconSize:[28,28], iconAnchor:[14,14] }});
        if (route.length) {{
            L.marker(route[0], {{ icon: sosIcon }}).addTo(map).bindTooltip('<b>Punto operativo</b>', {{ direction: 'top', className: 'custom-tip' }});
        }}

        const hospitalMarkers = {{}};
        scenario.hospitales.forEach((h) => {{
            const colorClass = (h.occ > 85 ? 'hosp-red' : (h.occ > 50 ? 'hosp-orange' : 'hosp-green'));
            const barColor = h.occ > 85 ? '#e74c3c' : (h.occ > 50 ? '#f39c12' : '#2ecc71');
            const icon = L.divIcon({{ className: 'hosp-marker ' + colorClass, html: '🏥', iconSize:[30,30], iconAnchor:[15,15] }});
            const tipHTML = `<div style=\"text-align:left;\"><center><b>${{h.nombre}}</b></center><hr style=\"margin:4px 0;\">🛏️ Ocupación: <b>${{h.occ}}%</b><div class=\"progress-bg\"><div class=\"progress-fill\" style=\"width:${{h.occ}}%; background-color:${{barColor}};\"></div></div><div style=\"margin-top:4px;\">⏱️ Espera: <b>${{h.wait}} min</b></div></div>`;
            const popupHTML = `
                <div class=\"hospital-popup\">
                    <div class=\"h-title\">🏥 ${{h.nombre}}</div>
                    <div class=\"h-meta\">${{h.centro_id}} · ${{h.centro_tipo}} · ${{h.municipio}}</div>
                    <div class=\"h-row\"><span class=\"h-label\">Dirección:</span><br>${{h.direccion}}</div>
                    <div class=\"h-row\"><span class=\"h-label\">Contacto:</span> ${{h.telefono}} · ${{h.email}}</div>
                    <div class=\"h-row\"><span class=\"h-label\">Médicos disponibles:</span> <b>${{h.medicos_disponibles}}</b></div>
                    <div class=\"h-row\"><span class=\"h-label\">Estado operativo:</span> Ocupación <b>${{h.occ}}%</b> · Espera <b>${{h.wait}} min</b></div>
                    <div class=\"progress-bg\"><div class=\"progress-fill\" style=\"width:${{h.occ}}%; background-color:${{barColor}};\"></div></div>
                    <div class=\"h-row\"><span class=\"h-label\">Perfiles:</span><div class=\"h-box\">${{h.perfiles}}</div></div>
                    <div class=\"h-row\"><span class=\"h-label\">Especialidades:</span><div class=\"h-box\">${{h.especialidades}}</div></div>
                </div>`;
            const marker = L.marker([h.lat, h.lon], {{ icon }}).addTo(map)
                .bindTooltip(tipHTML, {{ direction: 'top', className: 'custom-tip' }})
                .bindPopup(popupHTML, {{ maxWidth: 360 }});
            hospitalMarkers[h.nombre] = marker;
        }});

        function densificar(ruta, maxDist) {{
            const nueva = [];
            for (let i = 0; i < ruta.length - 1; i++) {{
                const p1 = ruta[i], p2 = ruta[i + 1];
                const dist = Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2));
                const pasos = Math.max(1, Math.ceil(dist / maxDist));
                for (let j = 0; j < pasos; j++) {{
                    nueva.push([p1[0] + (p2[0]-p1[0]) * (j/pasos), p1[1] + (p2[1]-p1[1]) * (j/pasos)]);
                }}
            }}
            nueva.push(ruta[ruta.length - 1]);
            return nueva;
        }}

        function haversineM(a, b) {{
            const R = 6371000;
            const rad = x => x * Math.PI / 180;
            const dLat = rad(b[0]-a[0]), dLon = rad(b[1]-a[1]);
            const h = Math.sin(dLat/2)**2 + Math.cos(rad(a[0]))*Math.cos(rad(b[0]))*Math.sin(dLon/2)**2;
            return 2*R*Math.atan2(Math.sqrt(h), Math.sqrt(1-h));
        }}

        function remainingDist(idx, ruta) {{
            let d = 0;
            for (let k = idx; k < ruta.length - 1; k++) d += haversineM(ruta[k], ruta[k+1]);
            return d;
        }}

        if (route.length >= 2) {{
            const idaSuave = densificar(route, 0.00018);
            const routeLine = L.polyline(route, {{ color: '#1a73e8', weight: 4, opacity: 0.5, dashArray: '8,10' }}).addTo(map);
            const doneLine = L.polyline([idaSuave[0]], {{ color: '#19e872', weight: 5, opacity: 0.9 }}).addTo(map);
            const ambIcon = L.divIcon({{ className: 'amb-icon', html: '🚑', iconSize:[32,32], iconAnchor:[16,16] }});
            const amb = L.marker(idaSuave[0], {{ icon: ambIcon }}).addTo(map);

            if (hospitalMarkers[scenario.destino.nombre]) {{
                hospitalMarkers[scenario.destino.nombre].setIcon(
                    L.divIcon({{ className: 'hosp-marker hosp-target', html: '🏥🏁', iconSize:[36,36], iconAnchor:[18,18] }})
                );
            }}

            map.fitBounds(routeLine.getBounds().pad(0.2));

            const etaTotal = Math.max(2, Number(scenario.eta_min || 8));
            const pasoAnimacion = 4;
            const tickMs = 50;
            let idx = 0;
            function animar() {{
                if (idx >= idaSuave.length) {{
                    document.getElementById('kEta').textContent = '0 min';
                    document.getElementById('kDist').textContent = '0 m';
                    return;
                }}
                amb.setLatLng(idaSuave[idx]);
                doneLine.setLatLngs(idaSuave.slice(0, idx + 1));
                const progress = idx / Math.max(1, idaSuave.length - 1);
                const etaNow = Math.max(0, Math.round(etaTotal * (1 - progress)));
                const rem = remainingDist(idx, idaSuave);
                document.getElementById('kEta').textContent = etaNow + ' min';
                document.getElementById('kDist').textContent = rem >= 1000 ? (rem / 1000).toFixed(1) + ' km' : Math.round(rem) + ' m';
                idx += pasoAnimacion;
                setTimeout(animar, tickMs);
            }}
            setTimeout(animar, 120);
        }}

        document.getElementById('kDestino').textContent = scenario.destino.nombre || '-';
        document.getElementById('kEta').textContent = `${{scenario.eta_min}} min`;
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
    st.map(pd.DataFrame(points), latitude="lat", longitude="lon", zoom=11)


def main() -> None:
    st.markdown(
        """
        <style>
            .u-hero {
                margin-top: 0.2rem;
                margin-bottom: 0.9rem;
                border-radius: 16px;
                padding: 16px 18px;
                background: linear-gradient(130deg, #005b8f 0%, #0a8fb5 48%, #00a37d 100%);
                color: white;
                box-shadow: 0 10px 24px rgba(12, 86, 128, 0.24);
            }
            .u-hero h2 { margin: 0; font-size: 1.4rem; }
            .u-hero p { margin: 6px 0 0; opacity: 0.93; }
            .u-note {
                margin: 0 0 12px 0;
                padding: 12px 14px;
                border-radius: 14px;
                background: rgba(255,255,255,0.86);
                border: 1px solid rgba(0, 91, 143, 0.12);
                color: #173147;
                box-shadow: 0 10px 22px rgba(12, 86, 128, 0.06);
                line-height: 1.55;
            }
            .u-note strong {
                color: #0a6fb8;
            }
            .u-section {
                font-size: 1.02rem;
                font-weight: 700;
                color: #113c67;
                margin: 0.7rem 0 0.45rem;
            }
        </style>
        <div class="u-hero">
            <h2>Panel Operador MPAA Unificado</h2>
            <p>Gestión clínica y monitorización operativa en tiempo real.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="u-note">
            <strong>Flujo de trabajo:</strong> primero se resuelve la evaluación clínica, después se consolida la derivación y finalmente se sincroniza la ruta con el mapa operativo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    op.main()

    incidents = map_get_scenario_incidents()
    st.markdown('<div class="u-section">Incidencias activas</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(incidents), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="u-section">Mapa operativo</div>', unsafe_allow_html=True)

    graph = map_load_graph_with_traffic()
    hospitals_df = map_load_hospitals(graph)
    bases = map_load_samur_bases(graph)

    if hospitals_df.empty:
        st.error("No se pudieron cargar hospitales con coordenadas validas para el mapa.")
        return

    if not bases:
        st.error("No se pudieron cargar bases SAMUR validas para el mapa.")
        return

    shared_state = load_state()
    scenario = map_build_operational_scenario(graph, hospitals_df, bases, shared_state)

    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.metric(
            "Hospitales disponibles",
            int(hospitals_df["centro_id"].nunique()) if "centro_id" in hospitals_df.columns else len(hospitals_df),
        )
    with metric_b:
        st.metric("Bases SAMUR", len(bases))
    with metric_c:
        st.metric("Incidencias activas", len(incidents))

    map_render_driver_map(
        scenario=scenario,
        bases=bases,
        incidents=incidents,
        hospitals_count=int(hospitals_df["centro_id"].nunique()) if "centro_id" in hospitals_df.columns else len(hospitals_df),
    )
    map_render_fallback_map(scenario=scenario, bases=bases)


if __name__ == "__main__":
    main()
