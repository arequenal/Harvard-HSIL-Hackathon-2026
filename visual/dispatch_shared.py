"""Estado compartido entre operador y conductor.

Persistimos en runtime/dispatch_state.json para sincronizar:
- resumen del caso y decision clinica
- hospital de destino
- alertas de trafico
- versionado para refrescos en UI
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT_ROOT / "runtime" / "dispatch_state.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state() -> Dict[str, Any]:
    return {
        "version": 0,
        "updated_at": _utc_now(),
        "case": {
            "summary": "Sin caso activo",
            "source_text": "",
            "urgencia": "No definida",
            "especialidad": "No definida",
            "explicacion_urgencia": "",
            "explicacion_especialidad": "",
        },
        "destination": {
            "centro_id": "",
            "nombre": "",
            "telefono": "",
            "email": "",
            "direccion": "",
            "eta_min": None,
        },
        "traffic_alerts": [],
        "operator_notes": "",
    }


def load_state() -> Dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        state = default_state()
        save_state(state)
        return state

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        data = default_state()
        save_state(data)
        return data

    for key, value in default_state().items():
        if key not in data:
            data[key] = value
    return data


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    tmp_path.replace(STATE_PATH)


def update_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state()
    _deep_update(state, patch)
    state["version"] = int(state.get("version", 0)) + 1
    state["updated_at"] = _utc_now()
    save_state(state)
    return state


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
