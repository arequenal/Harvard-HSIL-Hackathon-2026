import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
import random
import json
from pathlib import Path
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "madrid_grafo.graphml"
HOSPITALES_PATH = BASE_DIR / "hospitales_madrid_nodos.csv"
PROCESSED_HOSPITALES_PATH = BASE_DIR.parent / "analisis_datos" / "data" / "processed" / "centros_servicios_establecimientos_sanitarios_limpio.csv"

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
    if PROCESSED_HOSPITALES_PATH.exists():
        df = pd.read_csv(PROCESSED_HOSPITALES_PATH, sep=";")
        # Keep only Madrid centers for consistency with the Madrid road graph.
        if "municipio" in df.columns:
            df = df[df["municipio"].astype(str).str.lower() == "madrid"].copy()

        # Adapt processed schema to app schema.
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
        # Backward-compatible fallback.
        df = pd.read_csv(HOSPITALES_PATH)

    # Ensure optional enrichment columns always exist for tooltip rendering.
    for col in ["direccion_completa", "especialidades_texto", "perfiles_atencion"]:
        if col not in df.columns:
            df[col] = ""
    return df

grafo = load_graph_with_traffic()
df_hospitales = cargar_hospitales(grafo)
lista_nodos = list(grafo.nodes())

# --- ZONAS DE TRÁFICO ---
ZONAS_TRAFICO = [
    {"lat": 40.4215, "lon": -3.6590, "radio": 1500, "nivel": "Alto"},   
    {"lat": 40.3920, "lon": -3.6850, "radio": 1600, "nivel": "Alto"},   
    {"lat": 40.4490, "lon": -3.6450, "radio": 1300, "nivel": "Alto"},   
    {"lat": 40.4650, "lon": -3.6880, "radio": 1400, "nivel": "Medio"},  
    {"lat": 40.4190, "lon": -3.7020, "radio": 1100, "nivel": "Medio"},  
    {"lat": 40.4080, "lon": -3.6750, "radio": 900, "nivel": "Bajo"}     
]

# --- BASES DE AMBULANCIAS ---
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
                "nombre": row['nombre'], "lat": row['lat'], "lon": row['lon'],
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
        
        raw_gps_ida = [[grafo.nodes[n]['y'], grafo.nodes[n]['x']] for n in ruta_ida]
        raw_gps_vuelta = [[grafo.nodes[n]['y'], grafo.nodes[n]['x']] for n in ruta_vuelta]
        msg_ia = f"<b>Destino Óptimo Definido</b><br>{destino_hosp['nombre']}<br>Elegido por: Tráfico, Ocupación ({destino_hosp['occ']}%) y Espera."

        operativos.append({
            "hospitales": hosp_datos_sim,
            "emergencia": coords_em,
            "gps_ida": raw_gps_ida,
            "gps_vuelta": raw_gps_vuelta,
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
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        // CÁMARA MANUAL: Se centra en Madrid y no se mueve sola nunca más.
        var map = L.map('map', {preferCanvas: true}).setView([40.4168, -3.7038], 13);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png').addTo(map);

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
            L.marker([a.lat, a.lon], {icon: icon}).bindTooltip("<b>Base SVB</b><br>" + a.nombre, {direction: 'top', className: 'custom-tip'}).addTo(map);
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
                
                if (!hospitalMarkers[h.nombre]) {
                    hospitalMarkers[h.nombre] = L.marker([h.lat, h.lon]).addTo(map);
                }
                
                var marker = hospitalMarkers[h.nombre];
                marker.setIcon(icon);
                
                var tipHTML = `
                    <div style="text-align: left;">
                        <center><b>${h.nombre}</b></center><hr style="margin:4px 0;">
                        <div style="margin-top: 2px;">📍 <b>Dirección:</b><br><span style="display:block; max-width: 320px; white-space: normal;">${h.direccion || 'No disponible'}</span></div>
                        <div style="margin-top: 6px;">🩺 <b>Especialidades:</b><br><span style="display:block; max-width: 320px; max-height: 66px; overflow-y: auto; white-space: normal;">${h.especialidades || 'No disponible'}</span></div>
                        <div style="margin-top: 4px;">🧠 <b>Perfiles:</b> ${h.perfiles || 'No definido'}</div>
                        🛏️ Ocupación: <b>${h.occ}%</b>
                        <div class="progress-bg"><div class="progress-fill" style="width: ${h.occ}%; background-color: ${barColor};"></div></div>
                        <div style="margin-top: 4px;">⏱️ Espera: <b>${h.wait} min</b></div>
                    </div>`;
                marker.bindTooltip(tipHTML, {direction: 'top', offset: [0, -15], className: 'custom-tip'});
            });
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
            var velocidadIda = 13;
            var velocidadVuelta = 25;
            
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
                        var markerTarget = hospitalMarkers[op.destino.nombre];
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
                      .replace("__OPERATIVOS__", operativos_json)

components.html(html_mapa, height=1000)

