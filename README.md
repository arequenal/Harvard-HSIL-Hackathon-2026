# AmbulancIA - Hackathon HSIL 2026

AmbulancIA es una plataforma de apoyo a la decision para emergencias prehospitalarias en Madrid.

Combina analisis de datos, extraccion clinica, modelos de clasificacion y paneles operativos para coordinar derivacion hospitalaria en tiempo real.

## Tabla de contenidos

- Vision general
- Capacidades principales
- Arquitectura funcional
- Estructura del repositorio
- Requisitos
- Configuracion de entorno
- Ejecucion local
- Ejecucion con Docker Compose
- Uso del pipeline de triaje por CLI
- Flujo operativo recomendado
- Datos y modelos
- Solucion de problemas
- Roadmap
- Autoria

## Vision general

El sistema esta pensado para asistir dos perfiles:

- Operador: interpreta el caso clinico, ajusta decision y publica destino.
- Conductor: consume la decision publicada y navega con contexto operativo.

El proyecto incluye ademas:

- Procesamiento de datasets sanitarios en la carpeta analisis_datos.
- Pipeline de triaje texto -> urgencia + especialidad en la carpeta ml.
- Integracion LLM opcional con fallback heuristico para extraccion robusta.

## Capacidades principales

- Extraccion de variables clinicas estructuradas desde texto libre.
- Prediccion de urgencia y especialidad con probabilidades.
- Visualizacion de explicabilidad de la decision.
- Seleccion de hospital con criterios operativos.
- Estado compartido entre paneles mediante archivo persistente.
- Simulacion operativa en mapas de Madrid con activos locales.

## Arquitectura funcional

1. Entrada clinica en panel de operador o por CLI.
2. Extraccion de features con clinical_llm (LLM u heuristico).
3. Clasificacion con modelos XGBoost en ml/models.
4. Publicacion de destino y metadata en runtime/dispatch_state.json.
5. Consumo del estado por panel de conductor para navegacion y seguimiento.

## Estructura del repositorio

```text
.
├── analisis_datos/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   └── reports/
├── audio/
│   └── samples/
├── ml/
│   ├── data/
│   ├── models/
│   ├── urgency_specialty_classifier.ipynb
│   └── urgency_specialty_classifier.py
├── runtime/
│   └── dispatch_state.json
├── visual/
│   ├── data/
│   │   ├── hospitales_madrid_nodos.csv
│   │   ├── madrid_grafo.graphml
│   │   └── madrid_grafo.rar
│   ├── legacy/
│   │   ├── README.md
│   │   ├── prototipo_ambulancIA.py
│   │   ├── prototipo_ambulancIA_2.py
│   │   └── requirements.txt
│   ├── README.md
│   ├── conductor_navegacion.py
│   ├── dispatch_shared.py
│   ├── operador_mpaa_Unificado.py
│   └── operator_service.py
├── clinical_llm.py
├── docker-compose.yml
├── pyproject.toml
└── docker/
  └── Dockerfile
```

## Requisitos

- Python 3.11 o superior.
- Entorno virtual recomendado.
- Para opcion Docker: Docker Engine + Docker Compose.
- Para LLM local opcional: Ollama (host o contenedor).

Dependencias Python principales:

- streamlit
- pandas
- networkx
- osmnx
- xgboost
- scikit-learn
- faster-whisper
- matplotlib

## Configuracion de entorno

1. Duplica el archivo de ejemplo:

```bash
cp .env.example .env
```

2. Variables disponibles:

- AMBULANCIA_LLM_MODEL
- AMBULANCIA_LLM_PROVIDER
- AMBULANCIA_LLM_BASE_URL

Valores recomendados de arranque:

- provider: heuristico (sin dependencia LLM)
- provider: ollama (si ya tienes servidor disponible)

## Ejecucion local

### Opcion A: con uv (recomendada)

```bash
uv sync
```

Luego, en terminales separadas:

```bash
streamlit run visual/operator_service.py --server.port 8501
streamlit run visual/conductor_navegacion.py --server.port 8502
```

Opcional, panel unificado:

```bash
streamlit run visual/operador_mpaa_Unificado.py --server.port 8503
```

### Opcion B: con pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Ejecucion con Docker Compose

### Servicios principales

```bash
docker compose up --build
```

Accesos:

- Operador: http://localhost:8501
- Conductor: http://localhost:8502

### Incluir Ollama en Compose

```bash
docker compose --profile llm up --build
```

### Detener servicios

```bash
docker compose down
```

## Uso del pipeline de triaje por CLI

Ayuda:

```bash
python ml/urgency_specialty_classifier.py --help
```

Ejemplo rapido:

```bash
python ml/urgency_specialty_classifier.py "Paciente con dolor toracico y disnea" --pretty
```

Ejemplo con proveedor/modelo explicitos:

```bash
python ml/urgency_specialty_classifier.py \
  "Paciente de 67 anos con disnea y saturacion baja" \
  --llm-provider ollama \
  --llm-model llama3.1:8b-instruct \
  --llm-base-url http://localhost:11434 \
  --pretty
```

## Flujo operativo recomendado

1. Inicia operador y conductor.
2. Carga/transcribe texto clinico en operador.
3. Revisa urgencia y especialidad sugeridas.
4. Ajusta decision final si procede.
5. Publica destino al estado compartido.
6. Verifica recepcion en panel de conductor.

## Datos y modelos

- Datos crudos/procesados: analisis_datos/data.
- Reportes de calidad y distribucion: analisis_datos/reports.
- Modelos entrenados: ml/models.
- Datos cartograficos operativos: visual/data.

Nota: visual/data/madrid_grafo.rar se conserva como respaldo historico del grafo.

## Creditos de datos

- Datos de los hospitales de Madrid: Centros, servicios y establecimientos sanitarios - Conjunto de datos - CKAN.
- Fuente oficial: https://datos.comunidad.madrid/dataset/centros_servicios_establecimientos_sanitarios

## Solucion de problemas

- El operador no responde al LLM:
  - Verifica AMBULANCIA_LLM_PROVIDER.
  - Si usas ollama, confirma que AMBULANCIA_LLM_BASE_URL es accesible.

- Streamlit no abre puerto:
  - Comprueba que 8501/8502/8503 no estan ocupados.
  - Cambia puerto con --server.port.

- El conductor no recibe actualizaciones:
  - Revisa permisos y contenido de runtime/dispatch_state.json.
  - Verifica que ambos paneles usan el mismo workspace.

- Docker no disponible en WSL:
  - Activa integracion de Docker Desktop con WSL o usa ejecucion local.

## Roadmap

- Endurecer validacion de calidad de datos clinicos.
- Incorporar pruebas automatizadas de integracion UI-estado.
- Mejorar telemetria y auditoria de decisiones.
- Normalizar despliegue para entorno demo/produccion.

## Autoria

Equipo al'Hôpital para el hackathon HSIL de Hardvard.