import hashlib
from collections.abc import Iterable

import pandas as pd


REQUIRED_COLUMNS = (
    "scenario_id",
    "split",
    "time_s",
    "flow_direction",
    "n_comp_cmd_rpm",
    "n_pump_cmd_rpm",
    "n_comp_eff_rpm",
    "n_pump_eff_rpm",
    "q_gen_w",
    "t_ambient_c",
    "t_batt_c",
    "t_cool_c",
    "t_plate_c",
    "t_supply_c",
    "t_return_c",
    "q_evap_ss_w",
    "q_evap_eff_w",
    "q_cond_ss_w",
    "q_cond_eff_w",
)
VALID_SPLITS = frozenset({"train", "validation", "test"})


def assign_scenario_splits(
    scenario_ids: Iterable[str], seed: int = 20260714
) -> dict[str, str]:
    unique_ids = sorted({str(value) for value in scenario_ids})
    ranked_ids = sorted(
        unique_ids,
        key=lambda value: hashlib.sha256(
            f"{seed}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    n_total = len(ranked_ids)
    n_train = int(round(0.60 * n_total))
    n_validation = int(round(0.20 * n_total))

    splits = {}
    for index, scenario_id in enumerate(ranked_ids):
        if index < n_train:
            split = "train"
        elif index < n_train + n_validation:
            split = "validation"
        else:
            split = "test"
        splits[scenario_id] = split
    return splits


def validate_identification_frame(frame: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    invalid_splits = sorted(
        {
            repr(value)
            for value in frame["split"]
            if not isinstance(value, str) or value not in VALID_SPLITS
        }
    )
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}")

    leaked_scenarios = frame.groupby("scenario_id")["split"].nunique()
    leaked_scenarios = leaked_scenarios[leaked_scenarios > 1].index.tolist()
    if leaked_scenarios:
        raise ValueError(f"Scenarios assigned to multiple splits: {leaked_scenarios}")

    if frame.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required columns contain NaN values")
