import os
import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
import random
import json
from pathlib import Path
import streamlit.components.v1 as components

from clinical_llm import analyze_clinical_diagnosis

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "madrid_grafo.graphml"
HOSPITALES_PATH = BASE_DIR / "hospitales_madrid_nodos.csv"
PROCESSED_HOSPITALES_PATH = BASE_DIR.parent / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"
SAMUR_BASES_PATH = BASE_DIR.parent / "analisis_datos" / "data" / "processed" / "bases_samur_madrid.csv"

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA (Pantalla completa real)
# ==============================================================================
st.set_page_config(page_title="IA Ambulancias Smart City", layout="wide", initial_sidebar_state="collapsed")

# Ocultar todos los menús, cabeceras y márgenes de Streamlit para "Modo Kiosko"
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
            height: 100vh !important; /* Ocupa el 100% del alto de la ventana */
        }
    </style>
""", unsafe_allow_html=True)

# --- PANTALLA DE CARGA PERSONALIZADA ---
pantalla_carga = st.empty()
pantalla_carga.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background-color: #121212; z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; color: white; font-family: sans-serif;">
        <div style="font-size: 80px; animation: pulse 1.5s infinite;">🚑</div>
        <h1 style="margin-top: 20px; font-weight: 600; letter-spacing: 2px;">Cargando Ambulanc<span style="color: #dc3545;">IA</span></h1>
        <p style="color: #aaa; margin-top: 10px; font-size: 18px;">Preparando a todo el personal sanitario...</p>
        <div style="margin-top: 40px; width: 60px; height: 60px; border: 6px solid #333; border-top-color: #dc3545; border-radius: 50%; animation: spin 1s linear infinite;"></div>
    </div>
    <style>
        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.15); } 100% { transform: scale(1); } }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARGA DE DATOS Y MOTOR DE TRÁFICO GLOBAL (En Caché)
# ==============================================================================
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
    df = pd.DataFrame()

    if PROCESSED_HOSPITALES_PATH.exists():
        # Main source: curated processed hospitals dataset.
        df = pd.read_csv(PROCESSED_HOSPITALES_PATH, sep=";")
        allowed_center_types = {"Hospital general", "Hospital especializado"}
        if "centro_tipo" in df.columns:
            df = df[df["centro_tipo"].isin(allowed_center_types)].copy()

        if "centro_id" not in df.columns:
            df["centro_id"] = df.index.astype(str)
        df["centro_id"] = df["centro_id"].astype(str).str.strip()

        # Enrich names from legacy matched file when available.
        if HOSPITALES_PATH.exists():
            legacy_df = pd.read_csv(HOSPITALES_PATH)
            if {"centro_id", "nombre"}.issubset(legacy_df.columns):
                legacy_df = legacy_df[["centro_id", "nombre"]].copy()
                legacy_df["centro_id"] = legacy_df["centro_id"].astype(str).str.strip()
                legacy_df["nombre"] = legacy_df["nombre"].astype(str).str.strip()
                legacy_df = legacy_df[legacy_df["nombre"] != ""].drop_duplicates(subset=["centro_id"], keep="first")
                df = df.merge(legacy_df, on="centro_id", how="left", suffixes=("", "_legacy"))

        if "nombre" not in df.columns:
            df["nombre"] = ""
        if "nombre_legacy" in df.columns:
            df["nombre"] = df["nombre"].fillna("").astype(str).str.strip()
            df["nombre_legacy"] = df["nombre_legacy"].fillna("").astype(str).str.strip()
            df["nombre"] = df["nombre"].where(df["nombre"] != "", df["nombre_legacy"])
            df = df.drop(columns=["nombre_legacy"])

        df["nombre"] = df["nombre"].fillna("").astype(str).str.strip()
        df["nombre"] = df["nombre"].where(df["nombre"] != "", "Hospital " + df["centro_id"])
    elif HOSPITALES_PATH.exists():
        # Last fallback.
        df = pd.read_csv(HOSPITALES_PATH)
        if "centro_id" not in df.columns:
            df["centro_id"] = df.index.astype(str)
        if "nombre" not in df.columns:
            df["nombre"] = "Hospital " + df["centro_id"].astype(str)

    for col in ["lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["lat"].notna() & df["lon"].notna()].copy()

    if not df.empty:
        df["nodo_red"] = df.apply(
            lambda row: int(ox.distance.nearest_nodes(_grafo, X=float(row["lon"]), Y=float(row["lat"]))),
            axis=1,
        )

    # Ensure optional enrichment columns always exist for tooltip rendering.
    for col in ["direccion_completa", "especialidades_texto", "perfiles_atencion"]:
        if col not in df.columns:
            df[col] = ""
    return df


@st.cache_data
def cargar_bases_samur(_grafo):
    if SAMUR_BASES_PATH.exists():
        bases_df = pd.read_csv(SAMUR_BASES_PATH, sep=";")
        bases_df = bases_df.rename(columns={"latitud": "lat", "longitud": "lon"})

        for col in ["lat", "lon"]:
            bases_df[col] = pd.to_numeric(bases_df[col], errors="coerce")
        bases_df = bases_df[bases_df["lat"].notna() & bases_df["lon"].notna()].copy()

        bases = []
        for _, row in bases_df.iterrows():
            nombre = str(row.get("nombre", "Base SAMUR")).strip() or "Base SAMUR"
            bases.append(
                {
                    "base_id": str(row.get("base", "")).strip(),
                    "nombre": nombre,
                    "calle": str(row.get("calle", "")).strip(),
                    "distrito": str(row.get("distrito", "")).strip(),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "nodo_red": int(ox.distance.nearest_nodes(_grafo, X=float(row["lon"]), Y=float(row["lat"]))),
                }
            )
        if bases:
            return bases

    # Fallback mínimo por compatibilidad si falta el CSV procesado de SAMUR.
    return [
        {
            "base_id": "0",
            "nombre": "Base Central",
            "calle": "",
            "distrito": "",
            "lat": 40.4117,
            "lon": -3.7430,
            "nodo_red": int(ox.distance.nearest_nodes(_grafo, X=-3.7430, Y=40.4117)),
        }
    ]

grafo = load_graph_with_traffic()
df_hospitales = cargar_hospitales(grafo)
AMBULATORIOS = cargar_bases_samur(grafo)

if df_hospitales.empty:
    st.error("No se pudieron cargar hospitales con coordenadas válidas.")
    st.stop()

if not AMBULATORIOS:
    st.error("No se pudieron cargar bases SAMUR válidas.")
    st.stop()

lista_nodos = list(grafo.nodes())

RESUMEN_DATOS = {
    "hospitales": int(df_hospitales["centro_id"].nunique()) if "centro_id" in df_hospitales.columns else int(len(df_hospitales)),
    "bases_samur": int(len(AMBULATORIOS)),
}

# ============================================================================== 
# 3B. EXTRACTOR DE DIAGNÓSTICO A JSON / VECTOR
# ============================================================================== 
st.markdown("### Extractor de diagnóstico clínico")
st.caption("Pega una nota clínica en texto libre y el modelo la transforma en JSON estructurado o vector de características.")

with st.form("diagnostico_clinico_form"):
    diagnostico_texto = st.text_area(
        "Diagnóstico / nota clínica",
        placeholder="Ejemplo: Paciente de 68 años con dolor torácico súbito, disnea y saturación del 88%. Antecedentes de hipertensión.",
        height=180,
    )
    col_modelo, col_salida = st.columns([2, 1])
    with col_modelo:
        modelo_llm = st.text_input(
            "Modelo LLM",
            value=os.getenv("AMBULANCIA_LLM_MODEL", "llama3.1:8b-instruct"),
            help="Se usa Ollama por defecto. Si no está disponible, el sistema aplica un fallback local.",
        )
    with col_salida:
        modo_salida = st.radio("Vista", ["Ambas", "JSON", "Vector"], horizontal=False, index=0)
    ejecutar_analisis = st.form_submit_button("Analizar diagnóstico")

if ejecutar_analisis:
    if diagnostico_texto.strip():
        with st.spinner("Extrayendo información clínica..."):
            st.session_state["diagnostico_llm_resultado"] = analyze_clinical_diagnosis(diagnostico_texto, model=modelo_llm)
            st.session_state["diagnostico_llm_modo"] = modo_salida
    else:
        st.warning("Escribe un diagnóstico o una nota clínica antes de analizarla.")

if "diagnostico_llm_resultado" in st.session_state:
    resultado_llm = st.session_state["diagnostico_llm_resultado"]
    modo_activo = st.session_state.get("diagnostico_llm_modo", "Ambas")

    estado_fuente = "fallback local" if resultado_llm["fallback_used"] else f"{resultado_llm['provider']} / {resultado_llm['model']}"
    st.info(f"Salida generada con {estado_fuente}.")

    if modo_activo in {"Ambas", "JSON"}:
        st.subheader("JSON estructurado")
        st.json(resultado_llm["structured_output"])

    if modo_activo in {"Ambas", "Vector"}:
        st.subheader("Vector de características")
        st.dataframe(pd.DataFrame([resultado_llm["feature_map"]]), use_container_width=True)
        st.code(json.dumps(resultado_llm["feature_vector"], ensure_ascii=False, indent=2), language="json")

    with st.expander("Ver salida cruda del modelo", expanded=False):
        st.code(resultado_llm["raw_model_output"], language="json")

# --- ZONAS DE TRÁFICO ---
ZONAS_TRAFICO = [
    {"lat": 40.4215, "lon": -3.6590, "radio": 1500, "nivel": "Alto"},   
    {"lat": 40.3920, "lon": -3.6850, "radio": 1600, "nivel": "Alto"},   
    {"lat": 40.4490, "lon": -3.6450, "radio": 1300, "nivel": "Alto"},   
    {"lat": 40.4650, "lon": -3.6880, "radio": 1400, "nivel": "Medio"},  
    {"lat": 40.4190, "lon": -3.7020, "radio": 1100, "nivel": "Medio"},  
    {"lat": 40.4080, "lon": -3.6750, "radio": 900, "nivel": "Bajo"}     
]

# ==============================================================================
# 3. PRE-CÁLCULO DEL BUCLE
# ==============================================================================
operativos = []
NUM_SIMULACIONES = 5  

if 'simulaciones_generadas' not in st.session_state:
    for _ in range(NUM_SIMULACIONES):
        hosp_datos_sim = []
        for idx, row in df_hospitales.iterrows():
            hosp_datos_sim.append({
                "centro_id": str(row.get('centro_id', idx)),
                "nombre": str(row.get('nombre', f"Hospital {row.get('centro_id', idx)}")),
                "centro_tipo": str(row.get('centro_tipo', 'Hospital')),
                "lat": row['lat'], "lon": row['lon'],
                "nodo_red": int(row['nodo_red']), 
                "direccion": str(row.get('direccion_completa', '') or ''),
                "especialidades": str(row.get('especialidades_texto', '') or ''),
                "perfiles": str(row.get('perfiles_atencion', '') or ''),
                "occ": random.randint(30, 98), "wait": random.randint(10, 120)
            })

        acc_valido = False
        ruta_ida = []
        ruta_vuelta = []
        origen_amb = None
        destino_hosp = None
        coords_em = []

        while not acc_valido:
            nodo_emergencia = random.choice(lista_nodos)
            lat_em = grafo.nodes[nodo_emergencia]['y']
            lon_em = grafo.nodes[nodo_emergencia]['x']
            
            try:
                bases_ordenadas = sorted(AMBULATORIOS, key=lambda a: (a['lat'] - lat_em)**2 + (a['lon'] - lon_em)**2)
                ruta_ida = nx.shortest_path(grafo, source=bases_ordenadas[0]['nodo_red'], target=nodo_emergencia, weight='length')
                ruta_vuelta = nx.shortest_path(grafo, source=nodo_emergencia, target=df_hospitales.iloc[0]['nodo_red'], weight='weighted_length')
                
                origen_amb = bases_ordenadas[0]
                destino_hosp = hosp_datos_sim[0]
                coords_em = [lat_em, lon_em]
                acc_valido = True 
            except nx.NetworkXNoPath: 
                continue 
        
        min_coste = float('inf')
        for h in hosp_datos_sim:
            try:
                w_dist = nx.shortest_path_length(grafo, source=nodo_emergencia, target=h['nodo_red'], weight='weighted_length')
                coste = w_dist + (h['occ'] * 50) + (h['wait'] * 30)
                if coste < min_coste:
                    ruta_prueba = nx.shortest_path(grafo, source=nodo_emergencia, target=h['nodo_red'], weight='weighted_length')
                    min_coste = coste
                    destino_hosp = h
                    ruta_vuelta = ruta_prueba
            except nx.NetworkXNoPath: 
                continue
        
        gps_ida_coords = [[grafo.nodes[n]['y'], grafo.nodes[n]['x']] for n in ruta_ida]
        gps_vuelta_coords = [[grafo.nodes[n]['y'], grafo.nodes[n]['x']] for n in ruta_vuelta]
        msg_ia = f"<b>Destino Óptimo Definido</b><br>{destino_hosp['nombre']}<br>Elegido por: Tráfico, Ocupación ({destino_hosp['occ']}%) y Espera."

        operativos.append({
            "hospitales": hosp_datos_sim,
            "emergencia": coords_em,
            "gps_ida": gps_ida_coords,
            "gps_vuelta": gps_vuelta_coords,
            "destino": destino_hosp,
            "mensaje": msg_ia
        })
    
    st.session_state['simulaciones_generadas'] = operativos
else:
    operativos = st.session_state['simulaciones_generadas']

# ELIMINAR LA PANTALLA DE CARGA UNA VEZ TERMINADOS LOS CÁLCULOS
pantalla_carga.empty()

# ==============================================================================
# 4. RENDERIZADO DEL MAPA
# ==============================================================================
ambu_json = json.dumps(AMBULATORIOS)
zonas_json = json.dumps(ZONAS_TRAFICO)
operativos_json = json.dumps(operativos)
resumen_json = json.dumps(RESUMEN_DATOS)

html_crudo = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #map { height: 100%; width: 100%; margin: 0; padding: 0; background-color: #f4f4f4; overflow: hidden; }
        
        /* ANIMACIÓN SOS DEL PACIENTE */
        @keyframes pulseRed { 0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(220, 53, 69, 0); } 100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); } }
        .punto-paciente { 
            background-color: #dc3545; border: 2px solid white; border-radius: 50%; 
            animation: pulseRed 1.5s infinite; color: white; font-weight: bold; 
            font-size: 9px; text-align: center; line-height: 22px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.5); z-index: 1000 !important;
        }
        
        /* EFECTO FLUJO DE DATOS EN LA RUTA */
        .ruta-holografica { stroke-dasharray: 10, 15; animation: flowDash 1s linear infinite; }
        @keyframes flowDash { to { stroke-dashoffset: -25; } }
        
        .amb-icon { font-size: 32px; text-shadow: 2px 2px 5px rgba(0,0,0,0.8); text-align: center; z-index: 1001 !important; }
        
        .hosp-marker { background-color: white; border: 3px solid; border-radius: 50%; text-align: center; line-height: 24px; font-size: 16px; box-shadow: 0 3px 6px rgba(0,0,0,0.4); transition: all 0.5s ease; }
        .hosp-green { border-color: #2ecc71; }
        .hosp-orange { border-color: #f39c12; }
        .hosp-red { border-color: #e74c3c; }
        
        @keyframes glowTarget { 0% { box-shadow: 0 0 5px 2px rgba(52, 152, 219, 0.5); } 50% { box-shadow: 0 0 20px 8px rgba(52, 152, 219, 0.8); } 100% { box-shadow: 0 0 5px 2px rgba(52, 152, 219, 0.5); } }
        .hosp-target { border-color: #3498db !important; animation: glowTarget 1.5s infinite; z-index: 900 !important; transform: scale(1.2); }
        
        /* CSS DE LAS BASES */
        .ambu-marker { background-color: #e8f4f8; border: 2px solid #3498db; border-radius: 5px; text-align: center; line-height: 22px; font-size: 16px; }
        .custom-tip { font-family: Arial, sans-serif; font-size: 13px; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); border: none; text-align: center; }
        .traffic-tip { background-color: rgba(255,255,255,0.9); font-weight: bold; }
        
        /* CSS DE LAS BARRAS DE PROGRESO DE HOSPITALES */
        .progress-bg { background: #e0e0e0; width: 100%; height: 8px; border-radius: 4px; margin-top: 4px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }

        #hud {
            position: absolute;
            top: 16px;
            right: 16px;
            z-index: 1200;
            pointer-events: none;
        }
        .hud-card {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(11, 92, 171, 0.18);
            border-radius: 10px;
            padding: 10px 12px;
            width: 220px;
            box-shadow: 0 6px 14px rgba(0,0,0,0.18);
            font-family: Arial, sans-serif;
            color: #213547;
        }
        .hud-title {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.3px;
            margin-bottom: 8px;
            color: #0b5cab;
            text-transform: uppercase;
        }
        .hud-kpi {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            margin-bottom: 6px;
        }
        .hud-kpi b {
            color: #111;
            font-size: 14px;
        }

        /* PANEL LATERAL PARA DETALLES DE HOSPITAL */
        #sidebarPanel {
            position: fixed;
            top: 0;
            right: -380px;
            width: 360px;
            height: 100vh;
            background: white;
            box-shadow: -2px 0 8px rgba(0,0,0,0.2);
            z-index: 2000;
            transition: right 0.3s ease;
            overflow-y: auto;
            font-family: Arial, sans-serif;
        }
        #sidebarPanel.open {
            right: 0;
        }
        .sidebar-header {
            background: linear-gradient(135deg, #0b5cab 0%, #2196F3 100%);
            color: white;
            padding: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #ddd;
        }
        .sidebar-header h2 {
            margin: 0;
            font-size: 16px;
            flex-grow: 1;
        }
        .sidebar-close {
            background: rgba(255,255,255,0.3);
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            padding: 0 8px;
            border-radius: 4px;
        }
        .sidebar-close:hover {
            background: rgba(255,255,255,0.5);
        }
        .sidebar-content {
            padding: 16px;
            color: #333;
        }
        .info-section {
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #f0f0f0;
        }
        .info-section:last-child {
            border-bottom: none;
        }
        .info-label {
            font-weight: 700;
            color: #0b5cab;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 6px;
            letter-spacing: 0.4px;
        }
        .info-value {
            font-size: 14px;
            color: #333;
            line-height: 1.5;
            word-wrap: break-word;
        }
        .occupancy-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 4px;
        }
        .occupancy-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 11px;
            font-weight: bold;
        }
        .specialty-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }
        .specialty-tag {
            background: #f0f0f0;
            color: #333;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            border-left: 3px solid #0b5cab;
        }

        /* TOOLTIP PERSISTENTE */
        .hosp-tooltip-container {
            position: relative;
            z-index: 100;
        }
        .hosp-tooltip-content {
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif;
            font-size: 12px;
            max-width: 300px;
            z-index: 1100;
        }
    </style>
</head>
<body>
    <div id="sidebarPanel">
        <div class="sidebar-header">
            <h2>Detalles Hospital</h2>
            <button class="sidebar-close" onclick="cerrarSidebar()">&times;</button>
        </div>
        <div class="sidebar-content" id="sidebarContent"></div>
    </div>
    <div id="hud">
        <div class="hud-card">
            <div class="hud-title">Centro de Control</div>
            <div class="hud-kpi"><span>Hospitales</span><b id="kpiHosp">0</b></div>
            <div class="hud-kpi"><span>Bases SAMUR</span><b id="kpiBases">0</b></div>
            <div class="hud-kpi"><span>Estado</span><b id="kpiEstado">Inicializando</b></div>
        </div>
    </div>
    <div id="map"></div>
    <script>
        // CÁMARA MANUAL: Se centra en Madrid y no se mueve sola nunca más.
        var map = L.map('map', {preferCanvas: true}).setView([40.4168, -3.7038], 13);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png').addTo(map);

        var resumen = __RESUMEN_DATOS__;
        document.getElementById('kpiHosp').textContent = resumen.hospitales;
        document.getElementById('kpiBases').textContent = resumen.bases_samur;

        // ZONAS DE TRÁFICO
        var zonasTrafico = __ZONAS_TRAFICO__;
        zonasTrafico.forEach(function(zona) {
            if (zona.nivel === "Bajo") return; 
            var colorFondo = zona.nivel === "Alto" ? '#ff4d4d' : '#ffcc00'; 
            var circulo = L.circle([zona.lat, zona.lon], { radius: zona.radio, color: colorFondo, fillColor: colorFondo, fillOpacity: 0.15, weight: 1, opacity: 0.25 });
            circulo.bindTooltip("🚥 Tráfico: <b>" + zona.nivel + "</b>", { direction: 'center', className: 'custom-tip traffic-tip' }).addTo(map);
        });

        // BASES AMBULATORIOS
        var ambulatorios = __AMBULATORIOS__;
        ambulatorios.forEach(function(a) {
            var icon = L.divIcon({className: 'ambu-marker', html: '🩺', iconSize: [26,26], iconAnchor: [13,13]});
            var tooltipBase = "<b>Base SAMUR</b><br>" + a.nombre;
            if (a.distrito) tooltipBase += "<br><span style='color:#555'>Distrito: " + a.distrito + "</span>";
            if (a.calle) tooltipBase += "<br><span style='color:#555'>" + a.calle + "</span>";
            L.marker([a.lat, a.lon], {icon: icon}).bindTooltip(tooltipBase, {direction: 'top', className: 'custom-tip'}).addTo(map);
        });

        var operativos = __OPERATIVOS__;
        var currentIndex = 0;
        var markerAmb = null;
        var woundedMarker = null;
        var polylineVuelta = null;
        var hospitalMarkers = {};

        // Función para densificar y hacer fluido el movimiento
        function densificar(ruta, maxDist) {
            var nueva = [];
            for(var i=0; i<ruta.length-1; i++) {
                var p1 = ruta[i], p2 = ruta[i+1];
                var dist = Math.sqrt(Math.pow(p2[0]-p1[0],2) + Math.pow(p2[1]-p1[1],2));
                var pasos = Math.max(1, Math.ceil(dist / maxDist));
                for(var j=0; j<pasos; j++) nueva.push([p1[0] + (p2[0]-p1[0])*(j/pasos), p1[1] + (p2[1]-p1[1])*(j/pasos)]);
            }
            nueva.push(ruta[ruta.length-1]);
            return nueva;
        }

        // Actualizar datos de hospitales (sin ponerlos grises nunca)
        function actualizarHospitales(hospData) {
            hospData.forEach(function(h) {
                var colorClass = (h.occ > 85 ? 'hosp-red' : (h.occ > 50 ? 'hosp-orange' : 'hosp-green'));
                var barColor = h.occ > 85 ? '#e74c3c' : (h.occ > 50 ? '#f39c12' : '#2ecc71');
                
                var icon = L.divIcon({className: 'hosp-marker ' + colorClass, html: '🏥', iconSize: [30,30], iconAnchor: [15,15]});
                
                var markerKey = h.centro_id || h.nombre;
                if (!hospitalMarkers[markerKey]) {
                    hospitalMarkers[markerKey] = L.marker([h.lat, h.lon]).addTo(map);
                }
                
                var marker = hospitalMarkers[markerKey];
                marker.setIcon(icon);
                
                var tipHTML = `
                    <div style="text-align: left; cursor: pointer;" onclick="abrirSidebarHospital(" + JSON.stringify(h) + ")">
                        <center><b>${h.nombre}</b></center><hr style="margin:4px 0;">
                        <div style="margin-top: 2px; font-size: 11px;">🏷️ ${h.centro_id || 'N/D'} | ${h.centro_tipo || 'Hospital'}</div>
                        <div style="margin-top: 4px; font-size: 11px;">🛏️ Ocupación: <b>${h.occ}%</b></div>
                        <div class="progress-bg"><div class="progress-fill" style="width: ${h.occ}%; background-color: ${barColor};"></div></div>
                        <div style="margin-top: 4px; font-size: 11px; color: #0b5cab; font-weight: bold;">Haz click para más detalles →</div>
                    </div>`;
                
                var tooltipElement = L.popup({autoClose: false, closeButton: false, className: 'custom-tip'}).setContent(tipHTML);
                marker.bindPopup(tooltipElement);
                
                var tooltipShown = false;
                marker.on('mouseover', function() {
                    this.openPopup();
                    tooltipShown = true;
                });
                marker.on('mouseout', function() {
                    if (!tooltipShown) return;
                    setTimeout(function() {
                        if (!marker.getPopup()._container || !marker.getPopup()._container.closest('body')) return;
                        var popupContainer = marker.getPopup()._container;
                        if (!popupContainer.matches(':hover')) {
                            marker.closePopup();
                            tooltipShown = false;
                        }
                    }, 100);
                });
                
                marker.on('click', function() {
                    abrirSidebarHospital(h);
                });
            });
        }

        function abrirSidebarHospital(h) {
            var especialidadesArray = h.especialidades ? h.especialidades.split(' | ') : [];
            var perfilesArray = h.perfiles ? h.perfiles.split(' | ') : [];
            
            var content = `
                <div class="info-section">
                    <div class="info-label">Identificación</div>
                    <div class="info-value"><strong>${h.nombre}</strong></div>
                    <div class="info-value" style="font-size: 12px; color: #666;">ID: ${h.centro_id || 'N/D'} | Tipo: ${h.centro_tipo || 'Hospital'}</div>
                </div>

                <div class="info-section">
                    <div class="info-label">Ubicación</div>
                    <div class="info-value">${h.direccion || 'No disponible'}</div>
                    <div class="info-value" style="font-size: 11px; color: #999; margin-top: 6px;">Lat: ${h.lat.toFixed(4)} | Lon: ${h.lon.toFixed(4)}</div>
                </div>

                <div class="info-section">
                    <div class="info-label">Estado Operativo</div>
                    <div style="margin-bottom: 8px;">
                        <div class="info-value" style="margin-bottom: 4px;">Ocupación: <strong>${h.occ}%</strong></div>
                        <div class="occupancy-bar">
                            <div class="occupancy-fill" style="width: ${h.occ}%; background-color: ${h.occ > 85 ? '#e74c3c' : (h.occ > 50 ? '#f39c12' : '#2ecc71')};">
                                ${h.occ > 20 ? h.occ + '%' : ''}
                            </div>
                        </div>
                    </div>
                    <div class="info-value">Espera: <strong>${h.wait} min</strong></div>
                </div>

                ${especialidadesArray.length > 0 ? `
                <div class="info-section">
                    <div class="info-label">Especialidades</div>
                    <div class="specialty-list">
                        ${especialidadesArray.map(e => `<span class="specialty-tag">${e.trim()}</span>`).join('')}
                    </div>
                </div>
                ` : ''}

                ${perfilesArray.length > 0 ? `
                <div class="info-section">
                    <div class="info-label">Perfiles de Atención</div>
                    <div class="specialty-list">
                        ${perfilesArray.map(p => `<span class="specialty-tag" style="border-left-color: #4caf50;">${p.trim()}</span>`).join('')}
                    </div>
                </div>
                ` : ''}
            `;
            
            document.getElementById('sidebarContent').innerHTML = content;
            document.getElementById('sidebarPanel').classList.add('open');
        }

        function cerrarSidebar() {
            document.getElementById('sidebarPanel').classList.remove('open');
        }

        function limpiarMapa() {
            if(markerAmb) map.removeLayer(markerAmb);
            if(woundedMarker) map.removeLayer(woundedMarker);
            if(polylineVuelta) map.removeLayer(polylineVuelta);
        }

        function ejecutarBucle() {
            // Loop infinito: cuando acaba las precargadas, vuelve a la 0
            if (currentIndex >= operativos.length) currentIndex = 0; 
            
            var op = operativos[currentIndex];
            document.getElementById('kpiEstado').textContent = 'Emergencia en curso';
            limpiarMapa();
            actualizarHospitales(op.hospitales); 
            
            var gpsIda = op.gps_ida;
            var gpsVuelta = op.gps_vuelta;
            var coordAccidente = gpsIda[gpsIda.length - 1]; // ANCLAJE SEGURO DEL SOS
            
            // 1. DIBUJAMOS EL SOS
            var woundedIcon = L.divIcon({className: 'punto-paciente', html: 'SOS', iconSize: [26, 26], iconAnchor: [13, 13]});
            woundedMarker = L.marker(coordAccidente, {icon: woundedIcon}).addTo(map);

            // 2. CREAR AMBULANCIA
            var ambIcon = L.divIcon({className: 'amb-icon', html: '🚑', iconSize: [32, 32], iconAnchor: [16, 16]});
            markerAmb = L.marker(gpsIda[0], {icon: ambIcon}).addTo(map);

            // RUTA IDA Y VUELTA ULTRA FLUIDA
            var idaSuave = densificar(gpsIda, 0.0001);
            var vueltaSuave = densificar(gpsVuelta, 0.0001);
            var frameIndex = 0;
            
            // VELOCIDADES
            var velocidadIda = 10;
            var velocidadVuelta = 20;
            
            function animarIda() {
                if(frameIndex < idaSuave.length) { 
                    markerAmb.setLatLng(idaSuave[frameIndex]); 
                    frameIndex++;
                    setTimeout(animarIda, velocidadIda); 
                } else {
                    markerAmb.setLatLng(coordAccidente); // Seguro final al SOS
                    markerAmb.bindPopup("<b>🚨 Paciente localizado.</b><br>Evaluando constantes e IA de hospitales...").openPopup();
                    
                    setTimeout(function() {
                        markerAmb.closePopup();
                        map.removeLayer(woundedMarker); 
                        
                        // Resaltar destino (el resto se quedan con su color original, sin apagarse)
                        var markerTargetKey = op.destino.centro_id || op.destino.nombre;
                        var markerTarget = hospitalMarkers[markerTargetKey];
                        if(markerTarget) {
                            markerTarget.setIcon(L.divIcon({className: 'hosp-marker hosp-target', html: '🏥🏁', iconSize: [36,36], iconAnchor: [18,18]}));
                            // Usamos bindTooltip en lugar de openPopup para que solo se vea al pasar el ratón
                            markerTarget.bindTooltip(op.mensaje, {direction: 'top', offset: [0, -15], className: 'custom-tip'});
                        }

                        // LÍNEA DISCONTINUA DE VUELTA
                        polylineVuelta = L.polyline(gpsVuelta, {color: '#3498db', weight: 2.5, dashArray: '10, 10', className: 'ruta-holografica', opacity: 0.8}).addTo(map);
                        
                        setTimeout(function() {
                            frameIndex = 0;
                            animarVuelta(); 
                        }, 2500);

                    }, 3000); 
                }
            }

            function animarVuelta() {
                if(frameIndex < vueltaSuave.length) { 
                    markerAmb.setLatLng(vueltaSuave[frameIndex]); 
                    frameIndex++;
                    setTimeout(animarVuelta, velocidadVuelta); 
                } else {
                    document.getElementById('kpiEstado').textContent = 'Paciente entregado';
                    markerAmb.bindPopup("<h3 style='color:green; margin:0;'>✅ Llegada a Destino</h3>Paciente entregado.").openPopup();
                    
                    // ESPERAR 5 SEGUNDOS Y LANZAR LA SIGUIENTE EMERGENCIA
                    setTimeout(function() {
                        markerAmb.closePopup();
                        currentIndex++;
                        ejecutarBucle(); 
                    }, 5000);
                }
            }
            
            setTimeout(animarIda, 1000);
        }

        // CARGA INICIAL
        actualizarHospitales(operativos[0].hospitales);
        
        // Arranca el bucle a los 2 segundos exactos después de cargar el mapa
        setTimeout(function() {
            ejecutarBucle();
        }, 2000);

    </script>
</body>
</html>
"""

html_mapa = html_crudo.replace("__ZONAS_TRAFICO__", zonas_json)\
                      .replace("__AMBULATORIOS__", ambu_json)\
                      .replace("__OPERATIVOS__", operativos_json)\
                      .replace("__RESUMEN_DATOS__", resumen_json)

components.html(html_mapa, height=1000)

