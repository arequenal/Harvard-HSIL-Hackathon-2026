
# 🚑 AmbulancIA : Enrutamiento Dinámico de ambulancia

Prototipo inicial de simulador interactivo de emergencias médicas para la ciudad de Madrid. Utiliza la librería de Python NetworkX para graficar la red de calles de Madrid y el framework streamlit para mostrarlo. 
En una segunda fase, se evalúa el tráfico global de la ciudad (calculado por cuartiles) y la saturación hospitalaria en tiempo real para derivar al paciente al centro médico óptimo.

## ⚙️ Requisitos Previos
* Python 3.8 o superior.
* Visual Studio Code (o cualquier terminal).

## 🚀 Guía de Instalación y Ejecución rápida

Sigue estos pasos para arrancar el Centro de Mando en tu propio ordenador:

**1. Clona este repositorio (descarga esta carpeta)**
Descomprime la carpeta en sí y dentro el archivo madrid_grafo.rar. Asegúrate de que tu terminal está situada en la carpeta donde has descargado estos archivos. (comando cd ruta)

**2. Abre la terminal e instala las dependencias necesarias**
Ejecuta este comando para instalar el motor matemático y visual (instala las librerías y frameworks de requirements.txt):
```bash
python -m pip install -r requirements.txt
```
**3. Corre el código**
Ejecuta este comando, te abrirá una pestaña en el navegador:
```bash
python -m streamlit run app.py
```
