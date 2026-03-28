import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
import random
import json
import streamlit.components.v1 as components
import numpy as np 

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="IA Ambulancias Smart City", layout="wide")
st.title("🚑 Smart City: Enrutamiento Dinámico e Inteligencia de Tráfico")

# ==============================================================================
# 2. CARGA DE DATOS Y MOTOR DE TRÁFICO GLOBAL (En Caché)
# ==============================================================================
@st.cache_resource
def load_graph_with_traffic_and_quartiles():
    G = ox.load_graphml("madrid_grafo.graphml")
    print("Simulando tráfico global...")
    traffic_factors = []
    
    for u, v, key, data in G.edges(keys=True, data=True):
        traffic_factor = random.uniform(1.0, 3.0)
        data['traffic_factor'] = traffic_factor
        data['weighted_length'] = data['length'] * traffic_factor
        traffic_factors.append(traffic_factor)
    
    p25, p50, p75 = np.percentile(traffic_factors, [25, 50, 75])
    banda1, banda2, banda3, banda4 = [], [], [], []
    
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            coords = [[y, x] for x, y in list(data['geometry'].coords)]
        else:
            coords = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
            
        factor = data.get('traffic_factor', 1.0)
        if factor <= p25: banda1.append(coords)
        elif factor <= p50: banda2.append(coords)
        elif factor <= p75: banda3.append(coords)
        else: banda4.append(coords)
            
    return G, banda1, banda2, banda3, banda4

@st.cache_data
def cargar_hospitales():
    return pd.read_csv("hospitales_madrid_nodos.csv")

with st.spinner("Cargando cerebro de la ciudad de Madrid..."):
    grafo, calles_banda1, calles_banda2, calles_banda3, calles_banda4 = load_graph_with_traffic_and_quartiles()
    df_hospitales = cargar_hospitales()
    lista_nodos = list(grafo.nodes())

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
# 3. INTERFAZ Y ALGORITMIA ESPACIAL (Automática)
# ==============================================================================
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Centro de Mando")
    st.info("La simulación es totalmente automática. La ambulancia llegará al accidente, evaluará con IA la ciudad y trasladará al paciente.")
    
    generar = st.button("🚨 Iniciar Operativo Automático", type="primary", use_container_width=True)
    
    # Variables inicializadas vacías para que no de error
    hospitales_datos = []
    raw_gps_ida = []
    raw_gps_vuelta = []
    coords_emergencia = []
    destino_hospital = None
    origen_ambulatorio = None
    mensaje_ia = ""

    if generar:
        with st.spinner("IA calculando operativa completa..."):
            # Generar datos frescos de hospitales
            for idx, row in df_hospitales.iterrows():
                hospitales_datos.append({
                    "nombre": row['nombre'], "lat": row['lat'], "lon": row['lon'],
                    "nodo_red": int(row['nodo_red']), 
                    "occ": random.randint(30, 98), "wait": random.randint(10, 120)
                })

            # Generar Accidente
            acc_valido = False
            while not acc_valido:
                nodo_emergencia = random.choice(lista_nodos)
                try:
                    nx.shortest_path(grafo, source=nodo_emergencia, target=df_hospitales.iloc[0]['nodo_red'])
                    acc_valido = True
                except nx.NetworkXNoPath: 
                    continue 
            
            lat_em = grafo.nodes[nodo_emergencia]['y']
            lon_em = grafo.nodes[nodo_emergencia]['x']
            coords_emergencia = [lat_em, lon_em]
            
            # Buscar Base más cercana
            bases_ordenadas = sorted(AMBULATORIOS, key=lambda a: (a['lat'] - lat_em)**2 + (a['lon'] - lon_em)**2)
            ruta_ida = []
            
            for amb in bases_ordenadas:
                try:
                    ruta_ida = nx.shortest_path(grafo, source=amb['nodo_red'], target=nodo_emergencia, weight='length')
                    origen_ambulatorio = amb
                    break 
                except nx.NetworkXNoPath: 
                    continue
            
            raw_gps_ida = [[grafo.nodes[n]['y'], grafo.nodes[n]['x']] for n in ruta_ida]

            # Buscar Hospital Óptimo (IA con Tráfico)
            min_coste = float('inf')
            ruta_vuelta = []
            
            for h in hospitales_datos:
                try:
                    w_dist = nx.shortest_path_length(grafo, source=nodo_emergencia, target=h['nodo_red'], weight='weighted_length')
                    coste = w_dist + (h['occ'] * 50) + (h['wait'] * 30)
                    
                    if coste < min_coste:
                        min_coste = coste
                        destino_hospital = h
                        ruta_vuelta = nx.shortest_path(grafo, source=nodo_emergencia, target=h['nodo_red'], weight='weighted_length')
                except nx.NetworkXNoPath: 
                    continue
            
            raw_gps_vuelta = [[grafo.nodes[n]['y'], grafo.nodes[n]['x']] for n in ruta_vuelta]
            mensaje_ia = f"<b>Destino Óptimo Definido</b><br>{destino_hospital['nombre']}<br>Elegido por: Tráfico, Ocupación ({destino_hospital['occ']}%) y Espera."

            st.success("✅ **Simulación en curso:**")
            st.write(f"🩺 **Base salida:** {origen_ambulatorio['nombre']}")
            st.write(f"🏥 **Destino:** *Misterio (La IA decidirá in-situ)*")

    # Si no se ha generado, mostrar datos de hospitales en 0
    if not hospitales_datos:
        for idx, row in df_hospitales.iterrows():
            hospitales_datos.append({"nombre": row['nombre'], "lat": row['lat'], "lon": row['lon'], "occ": 0, "wait": 0})

with col2:
    # --- 4. MOTOR VISUAL JAVASCRIPT ---
    hospitales_json = json.dumps(hospitales_datos)
    ambu_json = json.dumps(AMBULATORIOS)
    emergencia_json = json.dumps(coords_emergencia)
    gps_ida_json = json.dumps(raw_gps_ida)
    gps_vuelta_json = json.dumps(raw_gps_vuelta)
    destino_json = json.dumps(destino_hospital) if destino_hospital else "null"
    msg_json = json.dumps(mensaje_ia)

    b1_json = json.dumps(calles_banda1)
    b2_json = json.dumps(calles_banda2)
    b3_json = json.dumps(calles_banda3)
    b4_json = json.dumps(calles_banda4)
    
    html_crudo = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            html, body, #map { height: 100%; width: 100%; margin: 0; padding: 0; background-color: #f4f4f4; }
            
            /* RADAR DE PACIENTE CSS PURO */
            @keyframes pulseRed { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }
            .punto-paciente { background-color: #ff0000; border: 2px solid white; border-radius: 50%; animation: pulseRed 1.5s infinite; }
            
            .amb-icon { font-size: 32px; text-shadow: 2px 2px 5px rgba(0,0,0,0.8); text-align: center; z-index: 1000 !important; }
            
            .hosp-marker { background-color: white; border: 3px solid; border-radius: 50%; text-align: center; line-height: 24px; font-size: 16px; box-shadow: 0 3px 6px rgba(0,0,0,0.4); transition: all 0.5s ease; }
            .hosp-green { border-color: #2ecc71; }
            .hosp-orange { border-color: #f39c12; }
            .hosp-red { border-color: #e74c3c; }
            .hosp-gris { border-color: #bdc3c7; opacity: 0.8; } 
            
            @keyframes glowTarget { 0% { box-shadow: 0 0 5px 2px rgba(52, 152, 219, 0.5); } 50% { box-shadow: 0 0 20px 8px rgba(52, 152, 219, 0.8); } 100% { box-shadow: 0 0 5px 2px rgba(52, 152, 219, 0.5); } }
            .hosp-target { border-color: #3498db !important; animation: glowTarget 1.5s infinite; z-index: 900 !important; }
            
            .ambu-marker { background-color: #e8f4f8; border: 2px solid #3498db; border-radius: 5px; text-align: center; line-height: 22px; font-size: 16px; }
            .custom-tip { font-family: Arial, sans-serif; font-size: 13px; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); border: none; text-align: center; }
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map', {preferCanvas: true}).setView([40.4168, -3.7038], 13);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png').addTo(map);

            // DIBUJAR TRÁFICO
            var estiloB1 = { color: '#f0f0f0', weight: 1.5, opacity: 0.3 }; 
            var estiloB2 = { color: '#e0e0e0', weight: 2.0, opacity: 0.4 }; 
            var estiloB3 = { color: '#d0d0d0', weight: 2.5, opacity: 0.5 }; 
            var estiloB4 = { color: '#b0b0b0', weight: 3.0, opacity: 0.6 }; 
            
            L.polyline(__C_BANDA1__, estiloB1).addTo(map);
            L.polyline(__C_BANDA2__, estiloB2).addTo(map);
            L.polyline(__C_BANDA3__, estiloB3).addTo(map);
            L.polyline(__C_BANDA4__, estiloB4).addTo(map);

            // DIBUJAR AMBULATORIOS
            var ambulatorios = __AMBULATORIOS__;
            ambulatorios.forEach(function(a) {
                var icon = L.divIcon({className: 'ambu-marker', html: '🩺', iconSize: [26,26], iconAnchor: [13,13]});
                L.marker([a.lat, a.lon], {icon: icon}).bindTooltip("<b>Base SVB</b><br>" + a.nombre, {direction: 'top', className: 'custom-tip'}).addTo(map);
            });

            // DIBUJAR HOSPITALES 
            var hospitalMarkers = {}; 
            var hospitales = __HOSPITALES__;
            var destHosp = __DESTINO_IA__;
            var haySimulacion = (__GPS_IDA__.length > 0);
            
            hospitales.forEach(function(h) {
                // Al principio, si no hay simulación están en gris. Si la hay, tienen sus colores de ocupación normales.
                var colorClass = (!haySimulacion || h.occ === 0) ? 'hosp-gris' : (h.occ > 85 ? 'hosp-red' : (h.occ > 50 ? 'hosp-orange' : 'hosp-green'));
                
                var icon = L.divIcon({className: 'hosp-marker ' + colorClass, html: '🏥', iconSize: [30,30], iconAnchor: [15,15]});
                var markerHosp = L.marker([h.lat, h.lon], {icon: icon});
                
                if (h.occ > 0) {
                    var tipHTML = `<b>${h.nombre}</b><hr style="margin:4px 0;">🛏️ Ocupación: <b>${h.occ}%</b><br>⏱️ Espera: <b>${h.wait} min</b>`;
                    markerHosp.bindTooltip(tipHTML, {direction: 'top', offset: [0, -15], className: 'custom-tip'});
                } else {
                    markerHosp.bindTooltip(`<b>${h.nombre}</b><br><i>Esperando IA...</i>`, {direction: 'top', offset: [0, -15], className: 'custom-tip'});
                }
                
                markerHosp.addTo(map);
                hospitalMarkers[h.nombre] = markerHosp; 
            });

            // ANIMACIÓN AUTOMÁTICA
            var coordAccidente = __EMERGENCIA__;
            var gpsIda = __GPS_IDA__;
            var gpsVuelta = __GPS_VUELTA__;
            var msgFase2 = __MSG_FASE2__;

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

            var velocidadLenta = 25; // Aumentamos este número para que vaya más LENTA

            if (haySimulacion) {
                // 1. DIBUJAR HERIDO
                var woundedIcon = L.divIcon({className: 'punto-paciente', html: '', iconSize: [18, 18], iconAnchor: [9, 9]});
                var woundedMarker = L.marker(coordAccidente, {icon: woundedIcon}).addTo(map);
                
                // Encuadrar base -> accidente -> destino todo de golpe para que no pegue saltos
                map.fitBounds(L.latLngBounds([gpsIda[0], coordAccidente, [destHosp.lat, destHosp.lon]]), {padding: [50, 50]});

                // 2. CREAR AMBULANCIA
                var ambIcon = L.divIcon({className: 'amb-icon', html: '🚑', iconSize: [32, 32], iconAnchor: [16, 16]});
                var markerAmb = L.marker(gpsIda[0], {icon: ambIcon}).addTo(map);

                var idaSuave = densificar(gpsIda, 0.00015);
                var vueltaSuave = densificar(gpsVuelta, 0.00015);
                var frameIndex = 0;
                
                // FASE 1: IDA
                function animarIda() {
                    if(frameIndex < idaSuave.length) { 
                        markerAmb.setLatLng(idaSuave[frameIndex]); 
                        frameIndex++;
                        setTimeout(animarIda, velocidadLenta); 
                    } else {
                        // LLEGADA AL PACIENTE - PAUSA DE 3 SEGUNDOS
                        markerAmb.bindPopup("<b>🚨 Paciente localizado.</b><br>Evaluando constantes e IA de hospitales...").openPopup();
                        
                        setTimeout(function() {
                            markerAmb.closePopup();
                            map.removeLayer(woundedMarker); // Paciente en ambulancia
                            
                            // LA REVELACIÓN DEL DESTINO
                            if(destHosp) {
                                var markerTarget = hospitalMarkers[destHosp.nombre];
                                if(markerTarget) {
                                    var iconTarget = L.divIcon({
                                        className: 'hosp-marker hosp-target', 
                                        html: '🏥🏁', 
                                        iconSize: [36,36], iconAnchor: [18,18]
                                    });
                                    markerTarget.setIcon(iconTarget); // Se vuelve azul!
                                    markerTarget.bindPopup("<b>🏁 DESTINO ÓPTIMO ASIGNADO</b><br>" + msgFase2).openPopup();
                                }
                            }
                            
                            // PAUSA DE 2 SEGUNDOS PARA LEER EL DESTINO ANTES DE ARRANCAR
                            setTimeout(function() {
                                if(markerTarget) markerTarget.closePopup();
                                frameIndex = 0;
                                animarVuelta(); 
                            }, 2500);

                        }, 3000); // 3000ms = 3 segundos de evaluación
                    }
                }

                // FASE 2: VUELTA
                function animarVuelta() {
                    if(frameIndex < vueltaSuave.length) { 
                        markerAmb.setLatLng(vueltaSuave[frameIndex]); 
                        frameIndex++;
                        setTimeout(animarVuelta, velocidadLenta); 
                    } else {
                        markerAmb.bindPopup("<h3 style='color:green; margin:0;'>✅ Llegada a Destino</h3>Paciente entregado.").openPopup();
                    }
                }
                
                // ARRANCAR TODO A LOS 1.5 SEGUNDOS DE PULSAR EL BOTÓN
                setTimeout(animarIda, 1500);
            }
        </script>
    </body>
    </html>
    """
    
    html_mapa = html_crudo.replace("__C_BANDA1__", b1_json)\
                          .replace("__C_BANDA2__", b2_json)\
                          .replace("__C_BANDA3__", b3_json)\
                          .replace("__C_BANDA4__", b4_json)\
                          .replace("__HOSPITALES__", hospitales_json)\
                          .replace("__AMBULATORIOS__", ambu_json)\
                          .replace("__EMERGENCIA__", emergencia_json)\
                          .replace("__GPS_IDA__", gps_ida_json)\
                          .replace("__GPS_VUELTA__", gps_vuelta_json)\
                          .replace("__DESTINO_IA__", destino_json)\
                          .replace("__MSG_FASE2__", msg_json)

    components.html(html_mapa, height=650)