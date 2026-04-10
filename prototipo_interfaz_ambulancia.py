"""
AmbulancIA - Prototipo de pantalla para conductor de ambulancia
Optimización de rutas mediante IA con pesos dinámicos según urgencia (GPS TURN-BY-TURN + CSV + REVEAL SCREEN)
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import google.generativeai as genai
import json
import math
import time
import random
import re
import os
import requests
import tkintermapview

# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDsZSimVi_IyNqFLp15GzvIHJaO9lc5ps0")
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ══════════════════════════════════════════════════════════════════
#  PALETA
# ══════════════════════════════════════════════════════════════════
DARK_BG      = "#070C17"
PANEL_BG     = "#0E1624"
CARD_BG      = "#162030"
ACCENT_BLUE  = "#1565C0"
ACCENT_BLUE2 = "#1E88E5"
ACCENT_CYAN  = "#00E5FF"
ACCENT_RED   = "#F44336"
ACCENT_GREEN = "#00E676"
TEXT_PRIMARY = "#EDF2F7"
TEXT_MUTED   = "#546E7A"
BORDER       = "#1E3048"

URGENCY_COLORS = {1: "#00E676", 2: "#EEFF41", 3: "#FFA726", 4: "#F44336"}
URGENCY_LABELS = {
    1: "URGENCIA 1 · LEVE",
    2: "URGENCIA 2 · MODERADA",
    3: "URGENCIA 3 · GRAVE",
    4: "URGENCIA 4 · CRÍTICA",
}

# ══════════════════════════════════════════════════════════════════
#  POSICIÓN INICIAL Y HOSPITALES
# ══════════════════════════════════════════════════════════════════
START_LAT = 40.4168
START_LON = -3.7038

HOSPITALS = [
    {"id": "CH0023", "name": "H. 12 de Octubre", "lat": 40.3713, "lon": -3.6900, "capacity": 88, "municipio": "Madrid", "address": "AVDA de Córdoba S/N Madrid 28041", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0034", "name": "H. Gregorio Marañón", "lat": 40.4182, "lon": -3.6702, "capacity": 88, "municipio": "Madrid", "address": "CALLE del Doctor Esquerdo 46 Madrid 28007", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0041", "name": "H. La Paz", "lat": 40.4800, "lon": -3.6897, "capacity": 86, "municipio": "Madrid", "address": "PASEO de la Castellana 261 Madrid 28046", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0053", "name": "H. Clínico San Carlos", "lat": 40.4418, "lon": -3.7180, "capacity": 84, "municipio": "Madrid", "address": "CALLE del Profesor Martín Lagos s/n Madrid 28040", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0049", "name": "H. Ramón y Cajal", "lat": 40.4901, "lon": -3.6961, "capacity": 82, "municipio": "Madrid", "address": "CTRA de Colmenar Viejo KM.9,1 Madrid 28034", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0044", "name": "H. Príncipe de Asturias", "lat": 40.5130, "lon": -3.3317, "capacity": 80, "municipio": "Alcalá de Henares", "address": "CTRA de Meco S/N Alcalá de Henares 28805", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0102", "name": "H. Puerta de Hierro", "lat": 40.4523, "lon": -3.8697, "capacity": 80, "municipio": "Majadahonda", "address": "CALLE Manuel de Falla 1 Majadahonda 28222", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0105", "name": "H. de Móstoles", "lat": 40.3175, "lon": -3.8631, "capacity": 76, "municipio": "Móstoles", "address": "CALLE Gladiolo S/N Móstoles 28933", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0075", "name": "H. de Getafe", "lat": 40.3048, "lon": -3.7237, "capacity": 74, "municipio": "Getafe", "address": "CTRA de Madrid-Toledo KM.12,500 Getafe 28905", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0080", "name": "H. Fundación Alcorcón", "lat": 40.3452, "lon": -3.8273, "capacity": 71, "municipio": "Alcorcón", "address": "CALLE Budapest 1 Alcorcón 28922", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0104", "name": "H. de Torrejón", "lat": 40.4594, "lon": -3.4872, "capacity": 71, "municipio": "Torrejón de Ardoz", "address": "CALLE Mateo Inurria 1 Torrejón de Ardoz 28850", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0095", "name": "H. Infanta Elena", "lat": 40.1923, "lon": -3.6686, "capacity": 70, "municipio": "Valdemoro", "address": "AVDA Reyes Católicos 21 Valdemoro 28342", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0086", "name": "H. de Fuenlabrada", "lat": 40.2842, "lon": -3.7942, "capacity": 69, "municipio": "Fuenlabrada", "address": "CMNO del Molino 2 Fuenlabrada 28942", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0106", "name": "H. de Collado Villalba", "lat": 40.6337, "lon": -4.0078, "capacity": 68, "municipio": "Collado Villalba", "address": "CTRA de Alpedrete M-608, KM. 41 Collado Villalba 28400", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0029", "name": "H. de La Princesa", "lat": 40.4342, "lon": -3.6766, "capacity": 65, "municipio": "Madrid", "address": "CALLE de Diego de León 62 Madrid 28006", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0052", "name": "H. Severo Ochoa", "lat": 40.3282, "lon": -3.7648, "capacity": 62, "municipio": "Leganés", "address": "AVDA Orellana S/N Leganés 28911", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0097", "name": "H. Infanta Sofía", "lat": 40.5498, "lon": -3.6268, "capacity": 62, "municipio": "San Sebastián de los Reyes", "address": "PASEO Europa 34 San Sebastián de los Reyes 28702", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0032", "name": "H. Rey Juan Carlos", "lat": 40.3242, "lon": -3.8791, "capacity": 58, "municipio": "Móstoles", "address": "CALLE Doctor Luis Montes S/N Móstoles 28935", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0099", "name": "H. del Tajo", "lat": 40.0366, "lon": -3.5986, "capacity": 57, "municipio": "Aranjuez", "address": "AVDA Amazonas Central S/N Aranjuez 28300", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0096", "name": "H. del Henares", "lat": 40.4237, "lon": -3.5625, "capacity": 56, "municipio": "Coslada", "address": "AVDA de Marie Curie s/n Coslada 28822", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0037", "name": "H. Niño Jesús", "lat": 40.4143, "lon": -3.6775, "capacity": 56, "municipio": "Madrid", "address": "AVDA de Menéndez Pelayo 65 Madrid 28009", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0100", "name": "H. del Sureste", "lat": 40.3072, "lon": -3.4432, "capacity": 54, "municipio": "Arganda del Rey", "address": "RONDA del Sur 10 Arganda del Rey 28500", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0098", "name": "H. Infanta Cristina", "lat": 40.2390, "lon": -3.7671, "capacity": 54, "municipio": "Parla", "address": "AVDA 9 de Junio 2 Parla 28981", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0101", "name": "H. Infanta Leonor", "lat": 40.3857, "lon": -3.6190, "capacity": 53, "municipio": "Madrid", "address": "AVDA de la Gran Vía del Este 80 Madrid 28031", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'pediatria', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia', 'oncologia']},
    {"id": "CH0051", "name": "H. Santa Cristina", "lat": 40.4216, "lon": -3.6720, "capacity": 46, "municipio": "Madrid", "address": "CALLE del Maestro Vives 2 Madrid 28009", "specialty": ['cardiovascular', 'neurologia', 'trauma_ortopedia', 'obstetricia_neonatal', 'salud_mental', 'diagnostico', 'urgencias_criticas', 'rehabilitacion', 'cirugia']}
]

# ══════════════════════════════════════════════════════════════════
#  GPS TURN-BY-TURN Y RUTAS
# ══════════════════════════════════════════════════════════════════
def get_real_route_and_steps(start_lat, start_lon, dest_lat, dest_lon):
    url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{dest_lon},{dest_lat}?overview=full&geometries=geojson&steps=true"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("code") == "Ok":
            coords = [(lat, lon) for lon, lat in data["routes"][0]["geometry"]["coordinates"]]
            steps = data["routes"][0]["legs"][0]["steps"]
            return coords, steps
    except Exception as e:
        print(f"⚠️ Error OSRM: {e}")
    return [(start_lat, start_lon), (dest_lat, dest_lon)], []

def format_step_instruction(step, dist_meters):
    maneuver = step.get('maneuver', {})
    m_type = maneuver.get('type', '')
    modifier = maneuver.get('modifier', 'straight')
    name = step.get('name', '')
    
    mod_es = {"left": "a la izquierda", "right": "a la derecha", "sharp left": "bruscamente a la izquierda", 
              "sharp right": "bruscamente a la derecha", "slight left": "ligeramente a la izquierda", 
              "slight right": "ligeramente a la derecha", "straight": "recto", "uturn": "cambio de sentido"}.get(modifier, modifier)
    
    arrow = "↑"
    if "left" in modifier: arrow = "↖"
    elif "right" in modifier: arrow = "↗"
    elif modifier == "uturn": arrow = "↩"
    
    if m_type == "roundabout" or m_type == "rotary":
        exit_num = maneuver.get("exit", "")
        action = f"Coja la salida {exit_num} en la rotonda" if exit_num else f"En la rotonda, gire {mod_es}"
    elif m_type == "arrive":
        action, arrow = "Llegada al destino", "📍"
    elif m_type == "depart":
        action = f"Diríjase hacia {name}" if name else "Inicie la ruta"
    elif m_type == "turn":
        action = f"Gire {mod_es}" + (f" en {name}" if name else "")
    elif m_type in ["merge", "on ramp", "off ramp"]:
        action = f"Incorpórese {mod_es}" + (f" a {name}" if name else "")
    else:
        action = f"Continúe por {name}" if name else "Siga recto"
        
    dist_text = f"A {int(dist_meters)} m" if dist_meters < 1000 else f"A {dist_meters/1000:.1f} km"
    return action, dist_text, arrow

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def step_towards(lat, lon, dest_lat, dest_lon, speed_kmh=60, interval_s=1.0):
    dist = haversine(lat, lon, dest_lat, dest_lon)
    if dist < 0.005: return dest_lat, dest_lon, True
    ratio = min((speed_kmh / 3600) * interval_s / dist, 1.0)
    return lat + ratio * (dest_lat - lat), lon + ratio * (dest_lon - lon), False

# ══════════════════════════════════════════════════════════════════
#  LÓGICA DE IA
# ══════════════════════════════════════════════════════════════════
def analyze_patient_with_ai(symptoms_text, amb_lat, amb_lon, callback):
    hospitals_info = json.dumps([
        {"id": i, "nombre": h["name"], "municipio": h["municipio"],
         "especialidades": h["specialty"], "distancia_km": round(haversine(amb_lat, amb_lon, h["lat"], h["lon"]), 1)}
        for i, h in enumerate(HOSPITALS)
    ], ensure_ascii=False, indent=2)

    prompt = f"""Eres el cerebro de IA de una ambulancia en Madrid. 
Utiliza las especialidades y distancias para encontrar el mejor destino de esta base de datos real.
Hospitales: {hospitals_info}
DEVUELVE ÚNICAMENTE JSON:
{{ "urgency_level": <1-4>, "hospital_id": <0-{len(HOSPITALS)-1}>, "eta_minutes": <int>, "medical_notes": "<resumen 1 línea>" }}
SÍNTOMAS: {symptoms_text}"""

    def run():
        try:
            text = re.sub(r"```json|```", "", gemini.generate_content(prompt).text.strip()).strip()
            callback(json.loads(text), None)
        except Exception as e: callback(None, str(e))
    threading.Thread(target=run, daemon=True).start()

# ══════════════════════════════════════════════════════════════════
#  PANEL IZQUIERDO — Síntomas
# ══════════════════════════════════════════════════════════════════
class SymptomsPanel(tk.Frame):
    def __init__(self, parent, on_analyze):
        super().__init__(parent, bg=PANEL_BG, width=420)
        self.pack_propagate(False)
        self.on_analyze = on_analyze
        self._recording = False
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=DARK_BG, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text=" Ambulanc", font=("Helvetica", 20, "bold"), fg=TEXT_PRIMARY, bg=DARK_BG).pack(side="left", padx=(16,0))
        tk.Label(hdr, text="IA", font=("Helvetica", 20, "bold"), fg=ACCENT_RED, bg=DARK_BG).pack(side="left")
        tk.Label(hdr, text="⬤ SISTEMA LISTO", font=("Helvetica", 9, "bold"), fg=ACCENT_GREEN, bg=DARK_BG).pack(side="right", padx=16)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=PANEL_BG, padx=16, pady=12)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="ESTADO DEL PACIENTE", font=("Helvetica", 10, "bold"), fg=ACCENT_CYAN, bg=PANEL_BG).pack(anchor="w")
        
        ta_wrap = tk.Frame(body, bg=CARD_BG)
        ta_wrap.pack(fill="both", expand=True, pady=(10, 10))
        self.ta = scrolledtext.ScrolledText(ta_wrap, font=("Helvetica", 11), bg=CARD_BG, fg=TEXT_PRIMARY, insertbackground=ACCENT_CYAN, relief="flat", bd=8)
        self.ta.pack(fill="both", expand=True, padx=1, pady=1)
        self._ph = "Detalle aquí los síntomas..."
        self.ta.insert("1.0", self._ph)
        self.ta.bind("<FocusIn>", lambda e: self.ta.delete("1.0", "end") if self.ta.get("1.0", "end-1c") == self._ph else None)

        btn_row = tk.Frame(body, bg=PANEL_BG)
        btn_row.pack(fill="x", pady=(0, 6))
        self.rec_btn = tk.Button(btn_row, text="⏺ GRABAR VOZ", font=("Helvetica", 10, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG, relief="flat", pady=11, cursor="hand2", command=self._toggle_rec)
        self.rec_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.ai_btn = tk.Button(btn_row, text="▶ ANALIZAR IA", font=("Helvetica", 10, "bold"), fg="white", bg=ACCENT_BLUE, relief="flat", pady=11, cursor="hand2", command=self._analyze)
        self.ai_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        self.rec_lbl = tk.Label(body, text="", font=("Helvetica", 8), fg=ACCENT_RED, bg=PANEL_BG)
        self.rec_lbl.pack(anchor="w")
        self.ai_lbl  = tk.Label(body, text="", font=("Helvetica", 8), fg=ACCENT_CYAN, bg=PANEL_BG)
        self.ai_lbl.pack(anchor="w")

    def _toggle_rec(self):
        if not self._recording:
            self._recording, self._stop_typing = True, False
            self.rec_btn.config(text="⏹ DETENER", bg="#3A1515", fg=ACCENT_RED)
            threading.Thread(target=self._record_and_type, daemon=True).start()
        else:
            self._stop_typing, self._recording = True, False
            self.rec_btn.config(text="⏺ GRABAR VOZ", bg=CARD_BG, fg=TEXT_PRIMARY)

    def _record_and_type(self):
        sample = "Paciente varón, 58 años. Dolor opresivo en el pecho. PA 90/60. Inicio hace 20 minutos."
        time.sleep(1)
        self.after(0, lambda: self.ta.delete("1.0", "end"))
        self.after(0, lambda: self.rec_lbl.config(text="✍ Transcribiendo..."))
        delay = 0
        for char in sample:
            if self._stop_typing: break
            delay += random.randint(28, 60)
            self.after(delay, lambda c=char: self.ta.insert("end", c) if self.ta.winfo_exists() else None)
        self.after(delay + 200, lambda: self._toggle_rec() if self._recording else None)
        self.after(delay + 200, lambda: self.rec_lbl.config(text="✓ Transcripción completada"))

    def _analyze(self):
        text = self.ta.get("1.0", "end-1c").strip()
        if not text or text == self._ph: return
        self.ai_btn.config(state="disabled", text="Analizando...", bg="#0D3A7A")
        self.ai_lbl.config(text="⏳ Conectando con los 25 Hospitales de Madrid...")
        analyze_patient_with_ai(text, START_LAT, START_LON, lambda r, e: self.after(0, lambda: self._done(r, e)))

    def _done(self, result, error):
        self.ai_lbl.config(text="")
        self.ai_btn.config(state="normal", text="▶ ANALIZAR IA", bg=ACCENT_BLUE)
        if error: messagebox.showerror("Error IA", f"Verifica tu API Key.\n\n{error}")
        else: self.on_analyze(result)

# ══════════════════════════════════════════════════════════════════
#  PANTALLA DE INICIO (SPLASH SCREEN)
# ══════════════════════════════════════════════════════════════════
class PlaceholderPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=DARK_BG)
        tk.Label(self, text="AmbulancIA", font=("Helvetica", 42, "bold"), fg="#1E3A55", bg=DARK_BG).place(relx=0.5, rely=0.40, anchor="center")
        tk.Label(self, text=f"Sistema de Triaje y Enrutamiento Inteligente\n\nIntroduzca los síntomas del paciente y pulse ▶ ANALIZAR IA\npara activar el GPS", font=("Helvetica", 13), fg="#1E3A55", bg=DARK_BG, justify="center").place(relx=0.5, rely=0.56, anchor="center")

# ══════════════════════════════════════════════════════════════════
#  PANEL DERECHO — Navegación GPS Reveal
# ══════════════════════════════════════════════════════════════════
class NavPanel(tk.Frame):
    def __init__(self, parent, on_back):
        super().__init__(parent, bg=DARK_BG)
        self.on_back = on_back
        self._move_job = None
        self._pulse_anim = 0
        self._is_moving = False
        
        self.amb_lat = START_LAT
        self.amb_lon = START_LON
        self.route_path = None
        self.hosp_marker = None
        self._blink_job = None
        
        self._build()
        self._pulse_loop()

    def _build(self):
        # ── INDICACIONES EN GRANDE (TURN-BY-TURN) ──
        self.gps_bar = tk.Frame(self, bg=ACCENT_BLUE)
        self.gps_bar.pack(fill="x")
        
        gps_inner = tk.Frame(self.gps_bar, bg=ACCENT_BLUE, padx=14, pady=16) 
        gps_inner.pack(fill="x")

        self.arrow_lbl = tk.Label(gps_inner, text="↑", font=("Helvetica", 46, "bold"), fg="white", bg=ACCENT_BLUE)
        self.arrow_lbl.pack(side="left", padx=(0, 20))
        
        gps_txt = tk.Frame(gps_inner, bg=ACCENT_BLUE)
        gps_txt.pack(side="left", fill="x", expand=True)
        
        self.dist_lbl = tk.Label(gps_txt, text="CALCULANDO...", font=("Helvetica", 28, "bold"), fg="white", bg=ACCENT_BLUE)
        self.dist_lbl.pack(anchor="w")
        self.instr_lbl = tk.Label(gps_txt, text="Analizando ruta óptima...", font=("Helvetica", 16), fg="#BBDEFB", bg=ACCENT_BLUE)
        self.instr_lbl.pack(anchor="w", pady=(2, 0))

        # ── Destino y Totales ──
        info_bar = tk.Frame(self, bg="#070F1C", pady=8)
        info_bar.pack(fill="x")
        
        dest_col = tk.Frame(info_bar, bg="#070F1C")
        dest_col.pack(side="left", padx=14)
        tk.Label(dest_col, text="DESTINO FINAL", font=("Helvetica", 8, "bold"), fg=TEXT_MUTED, bg="#070F1C").pack(anchor="w")
        self.dest_lbl = tk.Label(dest_col, text="—", font=("Helvetica", 12, "bold"), fg=TEXT_PRIMARY, bg="#070F1C")
        self.dest_lbl.pack(anchor="w")

        eta_col = tk.Frame(info_bar, bg="#070F1C")
        eta_col.pack(side="right", padx=14)
        tk.Label(eta_col, text="TIEMPO / DISTANCIA", font=("Helvetica", 8, "bold"), fg=TEXT_MUTED, bg="#070F1C").pack(anchor="e")
        self.eta_lbl = tk.Label(eta_col, text="— min (— km)", font=("Helvetica", 13, "bold"), fg=ACCENT_CYAN, bg="#070F1C")
        self.eta_lbl.pack(anchor="e")

        # ── EL MAPA REAL INTERACTIVO ──
        self.map_widget = tkintermapview.TkinterMapView(self, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_tile_server("https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png")
        self.map_widget.set_zoom(17)

        # ── Franja de Urgencia ──
        self.urg_band = tk.Frame(self, bg=PANEL_BG, pady=7)
        self.urg_band.pack(fill="x")
        urg_row = tk.Frame(self.urg_band, bg=PANEL_BG)
        urg_row.pack(fill="x", padx=14)

        self.urg_lbl = tk.Label(urg_row, text="—", font=("Helvetica", 15, "bold"), fg=ACCENT_RED, bg=PANEL_BG)
        self.urg_lbl.pack(side="left")
        tk.Button(urg_row, text="← Nuevo paciente", font=("Helvetica", 9), fg=TEXT_MUTED, bg=PANEL_BG, relief="flat", cursor="hand2", command=self.reset_map).pack(side="right")
        self.notes_lbl = tk.Label(self.urg_band, text="", font=("Helvetica", 9), fg=TEXT_MUTED, bg=PANEL_BG, wraplength=630, justify="left")
        self.notes_lbl.pack(fill="x", padx=14, pady=(2, 0))

    def _pulse_loop(self):
        """Dibuja el punto cian de la ambulancia y su radar en el centro de la pantalla"""
        if not self.winfo_exists(): return
        self.map_widget.canvas.delete("amb_dot")
        cx = self.map_widget.winfo_width() / 2
        cy = self.map_widget.winfo_height() / 2
        if cx > 10 and cy > 10:
            self._pulse_anim = (self._pulse_anim + 1) % 15
            pr = 12 + self._pulse_anim
            self.map_widget.canvas.create_oval(cx-pr, cy-pr, cx+pr, cy+pr, outline=ACCENT_CYAN, width=1, tags="amb_dot")
            self.map_widget.canvas.create_oval(cx-8, cy-8, cx+8, cy+8, fill=ACCENT_CYAN, outline="white", width=2, tags="amb_dot")
        self.after(50, self._pulse_loop)

    def update_with_result(self, result):
        if self.route_path: self.route_path.delete()
        if self.hosp_marker: self.hosp_marker.delete()

        self._urgency = result.get("urgency_level", 1)
        hosp_id = result.get("hospital_id", 0)
        notes   = result.get("medical_notes", "")
        
        safe_id = hosp_id if 0 <= hosp_id < len(HOSPITALS) else 0
        hosp = HOSPITALS[safe_id]
        color = URGENCY_COLORS.get(self._urgency, ACCENT_BLUE2)

        # 1. ACTUALIZAR TEXTOS
        self.gps_bar.config(bg=color)
        self.arrow_lbl.config(bg=color)
        self.dist_lbl.config(bg=color)
        self.instr_lbl.config(bg=color)
        
        self.dest_lbl.config(text=f"🏥 {hosp['name']}")
        self.urg_lbl.config(text=URGENCY_LABELS.get(self._urgency, f"URG {self._urgency}"), fg=color)
        self.notes_lbl.config(text=f"📋 Nota clínica: {notes}" if notes else "")

        # 2. CALCULAR RUTA E INICIAR EL MAPA (Se dibuja todo de golpe)
        self.hosp_marker = self.map_widget.set_marker(hosp["lat"], hosp["lon"], text="H", marker_color_circle=color, marker_color_outside=DARK_BG, text_color="white")
        self.real_route_coords, self.route_steps = get_real_route_and_steps(START_LAT, START_LON, hosp["lat"], hosp["lon"])
        self.route_path = self.map_widget.set_path(self.real_route_coords, color=color, width=8)

        self.amb_lat, self.amb_lon = START_LAT, START_LON
        self.dest_lat, self.dest_lon = hosp["lat"], hosp["lon"]
        self.route_coord_idx = 0
        self.current_step_idx = 0
        self.map_widget.set_position(self.amb_lat, self.amb_lon)

        # Totales Iniciales
        dist_total = haversine(self.amb_lat, self.amb_lon, self.dest_lat, self.dest_lon)
        dist_str_total = f"{int(dist_total*1000)} m" if dist_total < 1 else f"{dist_total:.2f} km"
        eta_min = max(1, int((dist_total/55)*60))
        self.eta_lbl.config(text=f"{eta_min} min ({dist_str_total})")

        if self._blink_job:
            self.after_cancel(self._blink_job)
            self._blink_job = None
        if self._urgency >= 3:
            self._start_blink(color)

        # 3. EFECTO REVELACIÓN (Pausa de 1.5s antes de arrancar)
        self.arrow_lbl.config(text="⏱")
        self.dist_lbl.config(text="LISTO")
        self.instr_lbl.config(text="Ruta calculada. Iniciando marcha...")
        
        self._is_moving = True
        self._move_job = self.after(1500, self._tick)

    def _tick(self):
        if not self.winfo_exists() or not self._is_moving: return
        
        # MOVER LA AMBULANCIA (A 70km/h para suavidad de red)
        if self.route_coord_idx < len(self.real_route_coords) - 1:
            target_lat, target_lon = self.real_route_coords[self.route_coord_idx + 1]
            new_lat, new_lon, arrived = step_towards(self.amb_lat, self.amb_lon, target_lat, target_lon, speed_kmh=70, interval_s=0.1)
            self.amb_lat, self.amb_lon = new_lat, new_lon
            if arrived: self.route_coord_idx += 1

        self.map_widget.set_position(self.amb_lat, self.amb_lon) 

        # INSTRUCCIONES TURN-BY-TURN
        if self.route_steps and self.current_step_idx < len(self.route_steps):
            step = self.route_steps[self.current_step_idx]
            loc = step.get('maneuver', {}).get('location', [self.dest_lon, self.dest_lat])
            dist_to_turn = haversine(self.amb_lat, self.amb_lon, loc[1], loc[0]) * 1000 
            
            if dist_to_turn < 25:
                self.current_step_idx += 1
                if self.current_step_idx < len(self.route_steps):
                    step = self.route_steps[self.current_step_idx]
                    loc = step.get('maneuver', {}).get('location', [self.dest_lon, self.dest_lat])
                    dist_to_turn = haversine(self.amb_lat, self.amb_lon, loc[1], loc[0]) * 1000

            instr_text, dist_str, arrow = format_step_instruction(step, dist_to_turn)
            self.dist_lbl.config(text=dist_str)
            self.instr_lbl.config(text=instr_text)
            self.arrow_lbl.config(text=arrow)
        else:
            self.dist_lbl.config(text="LLEGADA")
            self.instr_lbl.config(text="Ha llegado a su destino")
            self.arrow_lbl.config(text="📍")

        # TOTALES ABAJO
        dist_total = haversine(self.amb_lat, self.amb_lon, self.dest_lat, self.dest_lon)
        dist_str_total = f"{int(dist_total*1000)} m" if dist_total < 1 else f"{dist_total:.2f} km"
        eta_min = max(1, int((dist_total/55)*60))
        self.eta_lbl.config(text=f"{eta_min} min ({dist_str_total})")

        self._move_job = self.after(100, self._tick)

    def reset_map(self):
        self._is_moving = False
        if self._move_job:
            self.after_cancel(self._move_job)
            self._move_job = None
        if self._blink_job:
            self.after_cancel(self._blink_job)
            self._blink_job = None
            
        # Volver al inicio
        self.on_back()
        
        # Limpiar panel izquierdo
        self.master.master.symptoms_panel.ta.delete("1.0", "end")
        self.master.master.symptoms_panel.ta.insert("1.0", "Detalle aquí los síntomas...")
        self.master.master.symptoms_panel.ai_lbl.config(text="")
        self.master.master.symptoms_panel.rec_lbl.config(text="")

    def _start_blink(self, color):
        def blink():
            if not self.winfo_exists(): return
            uc = URGENCY_COLORS.get(self._urgency, color)
            self.urg_lbl.config(fg=uc if self._blink_on else PANEL_BG)
            self._blink_on = not self._blink_on
            self._blink_job = self.after(550, blink)
        blink()

# ══════════════════════════════════════════════════════════════════
#  APP PRINCIPAL
# ══════════════════════════════════════════════════════════════════
class AmbulancIAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AmbulancIA · Navegación GPS")
        self.geometry("1100x620")
        self.configure(bg=DARK_BG)
        
        self.symptoms_panel = SymptomsPanel(self, on_analyze=self._on_result)
        self.symptoms_panel.pack(side="left", fill="y")
        
        self.right = tk.Frame(self, bg=DARK_BG)
        self.right.pack(side="left", fill="both", expand=True)
        
        # Empezamos con la pantalla de bienvenida
        self.placeholder = PlaceholderPanel(self.right)
        self.placeholder.pack(fill="both", expand=True)
        
        self.nav_panel = NavPanel(self.right, on_back=self._reset)

    def _on_result(self, result):
        # 1. Ocultar pantalla de bienvenida
        self.placeholder.pack_forget()
        # 2. Mostrar panel de navegación
        self.nav_panel.pack(fill="both", expand=True)
        # 3. Dibujar mapa, ruta e iniciar pausa de 1.5s
        self.nav_panel.update_with_result(result)

    def _reset(self):
        # Volver al estado inicial
        self.nav_panel.pack_forget()
        self.placeholder.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = AmbulancIAApp()
    app.mainloop()