"""Pipeline de triaje para ambulancias.

Este modulo conecta:
1) Extraccion de features clinicas desde texto libre usando clinical_llm.py.
2) Prediccion de nivel de urgencia y especialidad con modelos XGBoost entrenados.
"""

from __future__ import annotations

import argparse
import os
import json
import pickle
import shutil
import sys
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd
import xgboost as xgb

# Permite ejecutar este script desde la raiz del proyecto o desde la carpeta ml.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_llm import FEATURE_ORDER, analyze_clinical_diagnosis

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - optional fallback dependency
    imageio_ffmpeg = None


def _resolve_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return

    if imageio_ffmpeg is None:
        raise RuntimeError(
            "No se encontro FFmpeg en el sistema ni la dependencia imageio-ffmpeg. "
            "Instala imageio-ffmpeg o ffmpeg para habilitar la transcripcion de audio."
        )

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = str(Path(ffmpeg_exe).resolve().parent)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def _resolve_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def _resolve_compute_type(device: str, compute_type: str) -> str:
    if compute_type and compute_type != "auto":
        return compute_type

    if device == "cuda":
        return "float16"

    return "int8"


@lru_cache(maxsize=8)
def _load_whisper_model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "No se pudo importar faster-whisper. Instala la dependencia para habilitar la transcripcion de audio."
        ) from exc

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio_file(
    audio_path: str,
    *,
    model_size: str = "medium",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
) -> Dict[str, Any]:
    """Transcribe un archivo de audio local y devuelve texto y metadatos."""

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de audio: {path}")

    _resolve_ffmpeg()
    resolved_device = _resolve_device(device)
    resolved_compute_type = _resolve_compute_type(resolved_device, compute_type)
    whisper_model = _load_whisper_model(model_size, resolved_device, resolved_compute_type)

    effective_language = None if language in {None, "", "auto"} else language
    segments, info = whisper_model.transcribe(
        str(path),
        language=effective_language,
        vad_filter=True,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.45,
    )

    segment_list = []
    text_parts: List[str] = []
    for segment in segments:
        cleaned_text = segment.text.strip()
        if cleaned_text:
            text_parts.append(cleaned_text)
        segment_list.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": cleaned_text,
            }
        )

    return {
        "audio_path": str(path),
        "model_size": model_size,
        "device": resolved_device,
        "compute_type": resolved_compute_type,
        "text": " ".join(text_parts).strip(),
        "language": getattr(info, "language", effective_language or "unknown"),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "segments": segment_list,
    }


class AmbulanceTriagePipeline:
    """Pipeline completo de extraccion + prediccion para una consulta de ambulancia."""

    def __init__(self, models_dir: str = "ml/models") -> None:
        self.models_dir = self._resolve_models_path(models_dir)

        self.urgency_model = self._load_xgb_model("urgency_model.json")
        self.specialty_model = self._load_xgb_model("specialty_model.json")
        self.metadata = self._load_pickle("metadata.pkl")
        self.label_encoder = self._load_pickle("label_encoder_urgency.pkl", required=False)

        self.urgency_names = self._normalize_name_map(self.metadata.get("urgency_names", {}))
        self.specialty_names = self._normalize_name_map(self.metadata.get("specialty_names", {}))

    @staticmethod
    def _resolve_models_path(models_dir: str) -> Path:
        candidate = Path(models_dir)
        if candidate.exists():
            return candidate

        project_root = Path(__file__).resolve().parents[1]
        fallback = project_root / models_dir
        if fallback.exists():
            return fallback

        raise FileNotFoundError(f"No se encontro el directorio de modelos: {models_dir}")

    def _load_xgb_model(self, filename: str) -> xgb.XGBClassifier:
        model_path = self.models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontro el modelo: {model_path}")

        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        return model

    def _load_pickle(self, filename: str, required: bool = True) -> Any:
        file_path = self.models_dir / filename
        if not file_path.exists():
            if required:
                raise FileNotFoundError(f"No se encontro el archivo: {file_path}")
            return None

        # Algunos artefactos del proyecto fueron guardados con pickle y otros con joblib.
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file)
        except Exception:
            try:
                import joblib  # type: ignore

                return joblib.load(file_path)
            except Exception:
                if required:
                    raise
                return None

    @staticmethod
    def _normalize_name_map(raw_map: Dict[Any, Any]) -> Dict[int, str]:
        normalized: Dict[int, str] = {}
        for key, value in raw_map.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized

    def _build_feature_frame(self, feature_map: Dict[str, Any]) -> pd.DataFrame:
        ordered_row = {column: feature_map.get(column, 0) for column in FEATURE_ORDER}
        return pd.DataFrame([ordered_row], columns=FEATURE_ORDER)

    def _decode_urgency_label(self, raw_prediction: int) -> int:
        if self.label_encoder is not None:
            decoded = self.label_encoder.inverse_transform([raw_prediction])[0]
            return int(decoded)

        # Fallback si no hay label encoder: intenta usar el mapeo de metadatos.
        if raw_prediction in self.urgency_names:
            return raw_prediction

        if (raw_prediction + 1) in self.urgency_names:
            return raw_prediction + 1

        return raw_prediction

    def _decode_specialty_label(self, raw_prediction: int) -> int:
        if raw_prediction in self.specialty_names:
            return raw_prediction
        return raw_prediction

    @staticmethod
    def _build_probability_map(probabilities: List[float], name_map: Dict[int, str]) -> Dict[str, float]:
        prob_map: Dict[str, float] = {}
        for index, value in enumerate(probabilities):
            label = name_map.get(index, str(index))
            prob_map[label] = float(value)
        return prob_map

    def _build_urgency_probability_map(self, probabilities: List[float]) -> Dict[str, float]:
        prob_map: Dict[str, float] = {}
        for raw_index, value in enumerate(probabilities):
            decoded_label = self._decode_urgency_label(raw_index)
            label_name = self.urgency_names.get(decoded_label, str(decoded_label))
            prob_map[label_name] = float(value)
        return prob_map

    def triage_from_text(
        self,
        diagnosis_text: str,
        *,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        llm_base_url: str | None = None,
    ) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de triaje desde texto clinico libre."""

        kwargs: Dict[str, Any] = {}
        if llm_model is not None:
            kwargs["model"] = llm_model
        if llm_provider is not None:
            kwargs["provider"] = llm_provider
        if llm_base_url is not None:
            kwargs["base_url"] = llm_base_url

        extraction = analyze_clinical_diagnosis(diagnosis_text, **kwargs)
        feature_map = extraction["feature_map"]
        features_df = self._build_feature_frame(feature_map)

        urgency_pred_raw = int(self.urgency_model.predict(features_df)[0])
        urgency_proba = self.urgency_model.predict_proba(features_df)[0].tolist()
        urgency_label = self._decode_urgency_label(urgency_pred_raw)

        specialty_pred_raw = int(self.specialty_model.predict(features_df)[0])
        specialty_proba = self.specialty_model.predict_proba(features_df)[0].tolist()
        specialty_label = self._decode_specialty_label(specialty_pred_raw)

        result = {
            "input_text": diagnosis_text.strip(),
            "llm_extraction": {
                "provider": extraction["provider"],
                "model": extraction["model"],
                "fallback_used": extraction["fallback_used"],
                "feature_order": FEATURE_ORDER,
                "feature_map": feature_map,
            },
            "prediction": {
                "urgencia": {
                    "raw_class": urgency_pred_raw,
                    "label": urgency_label,
                    "name": self.urgency_names.get(urgency_label, str(urgency_label)),
                    "confidence": float(urgency_proba[urgency_pred_raw]),
                    "probabilities": self._build_urgency_probability_map(urgency_proba),
                },
                "especialidad": {
                    "raw_class": specialty_pred_raw,
                    "label": specialty_label,
                    "name": self.specialty_names.get(specialty_label, str(specialty_label)),
                    "confidence": float(specialty_proba[specialty_pred_raw]),
                    "probabilities": self._build_probability_map(specialty_proba, self.specialty_names),
                },
            },
        }

        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de triaje: texto clinico -> urgencia + especialidad"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Texto de la consulta. Si se omite, se lee desde stdin.",
    )
    parser.add_argument(
        "--models-dir",
        default="ml/models",
        help="Directorio que contiene urgency_model.json, specialty_model.json y metadata.pkl",
    )
    parser.add_argument("--llm-model", default=None, help="Modelo LLM para clinical_llm")
    parser.add_argument("--llm-provider", default=None, help="Proveedor LLM para clinical_llm")
    parser.add_argument("--llm-base-url", default=None, help="Base URL del proveedor LLM")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Imprimir JSON formateado con identacion.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    text = args.text if args.text is not None else input().strip()
    if not text:
        raise ValueError("Debes proporcionar un texto clinico para el triaje.")

    pipeline = AmbulanceTriagePipeline(models_dir=args.models_dir)
    result = pipeline.triage_from_text(
        text,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        llm_base_url=args.llm_base_url,
    )

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
