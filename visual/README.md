# Visual module layout

This folder contains all Streamlit frontends and shared operational assets.

## Structure

- conductor_navegacion.py: Driver dashboard (route and dispatch view).
- operator_service.py: Operator dashboard (triage and destination publish).
- operador_mpaa_Unificado.py: Unified dashboard combining clinical and map views.
- dispatch_shared.py: Shared state storage helpers (runtime/dispatch_state.json).
- data/: Graph and node assets used by visual dashboards.
- legacy/: Historical prototypes kept for reference.

## Run

From project root:

- streamlit run visual/operator_service.py --server.port 8501
- streamlit run visual/conductor_navegacion.py --server.port 8502
- streamlit run visual/operador_mpaa_Unificado.py --server.port 8503
