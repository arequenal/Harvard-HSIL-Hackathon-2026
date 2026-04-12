# AmbulancIA · Hackathon HSIL 2026

Plataforma de apoyo a la decision para emergencias extrahospitalarias en Madrid.

El repositorio integra:
- analisis y limpieza de datos sanitarios,
- un pipeline de triaje clinico (urgencia + especialidad),
- paneles Streamlit para operador y conductor,
- sincronizacion en tiempo real mediante estado compartido.

## Objetivo

Reducir el tiempo de decision y mejorar la derivacion hospitalaria con una combinacion de:
- reglas clinicas estructuradas,
- modelos de ML entrenados,
- contexto operativo (trafico, disponibilidad, localizacion).

## Demo rapida

### Opcion A: entorno local (sin Docker)

1. Crear entorno e instalar dependencias:

	 uv sync

2. Lanzar servicio de operador:

	 streamlit run visual/operator_service.py --server.port 8501

3. Lanzar servicio de conductor (en otra terminal):

	 streamlit run visual/conductor_navegacion.py --server.port 8502

### Opcion B: Docker Compose

1. Construir y arrancar:

	 docker compose up --build

	Para incluir Ollama dentro de Compose:

	docker compose --profile llm up --build

2. Abrir en navegador:
- Operador: http://localhost:8501
- Conductor: http://localhost:8502

3. Parar servicios:

	 docker compose down

## Estructura del proyecto

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
│   ├── urgency_specialty_classifier.py
│   └── urgency_specialty_classifier.ipynb
├── runtime/
│   └── dispatch_state.json
├── visual/
│   ├── operator_service.py
│   ├── conductor_navegacion.py
│   ├── operador_mpaa_Unificado.py
│   ├── dispatch_shared.py
│   ├── data/
│   │   ├── madrid_grafo.graphml
│   │   ├── hospitales_madrid_nodos.csv
│   │   └── madrid_grafo.rar
│   └── legacy/
│       ├── README.md
│       ├── prototipo_ambulancIA.py
│       └── prototipo_ambulancIA_2.py
├── Grafos1.py
├── clinical_llm.py
├── testing_grafos_v1.py
├── pyproject.toml
├── docker-compose.yml
└── docker/Dockerfile

## Componentes clave

- Triaje clinico
	- Archivo: ml/urgency_specialty_classifier.py
	- Entrada: texto clinico
	- Salida: urgencia, especialidad, probabilidades y explicabilidad

- Extraccion de variables clinicas
	- Archivo: clinical_llm.py
	- Soporta proveedor LLM (Ollama) y fallback heuristico local

- Operador (UI)
	- Archivo: visual/operator_service.py
	- Permite transcripcion, revision de vector, seleccion de hospital y publicacion al conductor

- Conductor (UI)
	- Archivo: visual/conductor_navegacion.py
	- Consume estado compartido, representa ruta y muestra alertas operativas

- Estado compartido
	- Archivo: visual/dispatch_shared.py
	- Persistencia en runtime/dispatch_state.json

## Variables de entorno

Puedes copiar .env.example a .env para customizar comportamiento:

- AMBULANCIA_LLM_MODEL: modelo de LLM (ej. llama3.1:8b-instruct)
- AMBULANCIA_LLM_PROVIDER: proveedor de extraccion (ollama o heuristico)
- AMBULANCIA_LLM_BASE_URL: URL base de Ollama

## Comandos utiles

- Entrenar o ejecutar pipeline de triaje por CLI:

	python ml/urgency_specialty_classifier.py --help

- Ejecucion de prueba con texto:

	python ml/urgency_specialty_classifier.py "Paciente con dolor toracico y disnea" --pretty

## Estado del proyecto

Repositorio de prototipado de hackathon en evolucion. Incluye activos de exploracion y versiones historicas de interfaz para comparar enfoques.

Los prototipos historicos se concentran en visual/legacy para mantener limpia la raiz operativa de visual/.

## Contribuir

Consulta CONTRIBUTING.md para estandar de ramas, estilo y convenciones de commit.