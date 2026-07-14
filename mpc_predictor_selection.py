import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


CANDIDATE_B = "candidate_b"
PHYSICS_P = "physics_p"
LPV_L = "lpv_l"

SUPPORTED_PREDICTORS = (CANDIDATE_B, PHYSICS_P, LPV_L)


class PredictorArtifactError(ValueError):
    pass


class OfflinePredictor(Protocol):
    def reset(self, initial_state: dict[str, float]) -> None:
        ...

    def step(self, inputs: dict[str, float]) -> dict[str, float]:
        ...


@dataclass
class GekkoPredictorVariables:
    t_batt_k: object
    t_cool_k: object
    t_plate_c: object
    t_supply_c: object
    t_return_c: object
    n_comp: object
    n_pump: object
    q_evap: object
    q_cond: object
    q_evap_cmd_w: object


def normalize_predictor_name(value: object) -> str:
    if value is None:
        return CANDIDATE_B

    normalized = str(value).strip().lower()
    if not normalized:
        return CANDIDATE_B
    if normalized not in SUPPORTED_PREDICTORS:
        raise ValueError(f"Unsupported predictor: {value!r}")
    return normalized


def load_predictor_artifact(path: str | Path, expected_type: object) -> dict:
    artifact_path = Path(path)
    try:
        with artifact_path.open("r", encoding="utf-8") as artifact_file:
            artifact = json.load(artifact_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictorArtifactError(
            f"Unable to load predictor artifact: {artifact_path}"
        ) from exc

    if not isinstance(artifact, dict):
        raise PredictorArtifactError("Predictor artifact must be a JSON object")

    normalized_expected_type = normalize_predictor_name(expected_type)
    if artifact.get("model_type") != normalized_expected_type:
        raise PredictorArtifactError(
            "Predictor artifact model_type does not match expected_type"
        )
    if artifact.get("schema_version") != 1:
        raise PredictorArtifactError("Unsupported predictor artifact schema_version")

    return artifact
