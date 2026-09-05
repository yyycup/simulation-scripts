"""Frozen 13-node cold-plate reference used only for ROM validation."""

from __future__ import annotations

import numpy as np

from cluster_plant_v2.parameters import (
    BATTERY_PLATE_AREA_M2,
    BATTERY_PLATE_NOMINAL_HTC_W_M2_K as h_bp_nominal,
    COOLANT_DENSITY_KG_M3 as rho_cool,
    COOLANT_SPECIFIC_HEAT_J_KG_K as cp_cool,
    NOMINAL_COOLANT_MASS_FLOW_KG_S as m_dot_nominal,
)


N_bp = 13
A_bp_seg = BATTERY_PLATE_AREA_M2 / N_bp


def cold_plate_fluid_exchange(
    T_plate_wall_array=None,
    T_cool_in=None,
    m_dot_cool=None,
    **kwargs,
):
    """Exact isolated copy of the legacy 13-node coolant exchange calculation."""
    if T_plate_wall_array is None:
        T_plate_wall_array = kwargs.pop("T_plate_wall", None)
    if T_cool_in is None:
        T_cool_in = kwargs.pop("T_cool_in_to_plate", None)
    if T_plate_wall_array is None or T_cool_in is None or m_dot_cool is None:
        raise TypeError(
            "cold_plate_fluid_exchange requires plate temperature, "
            "coolant inlet temperature, and coolant flow"
        )

    if np.isscalar(T_plate_wall_array):
        T_walls = np.full(N_bp, T_plate_wall_array)
    else:
        T_walls = T_plate_wall_array

    if m_dot_cool < 1e-6:
        h_dyn = 50.0
    else:
        h_dyn = max(50.0, h_bp_nominal * (m_dot_cool / m_dot_nominal) ** 0.8)

    T_local = T_cool_in
    Q_total = 0.0
    T_fluid_profile = []
    for index in range(N_bp):
        T_wall = T_walls[index]
        DT_in = T_wall - T_local
        Q_guess = h_dyn * A_bp_seg * DT_in
        T_next = T_local + Q_guess / (m_dot_cool * cp_cool + 1e-12)
        DT_out = T_wall - T_next

        if (
            abs(DT_in) < 1e-4
            or abs(DT_out) < 1e-4
            or abs(DT_in - DT_out) < 1e-4
        ):
            DT_LMTD = (DT_in + DT_out) / 2.0
        elif DT_in * DT_out <= 0:
            DT_LMTD = (DT_in + DT_out) / 2.0
        else:
            try:
                DT_LMTD = (DT_in - DT_out) / np.log(DT_in / DT_out)
            except Exception:
                DT_LMTD = (DT_in + DT_out) / 2.0

        Q_i = h_dyn * A_bp_seg * DT_LMTD
        T_next = T_local + Q_i / (m_dot_cool * cp_cool + 1e-12)
        T_fluid_profile.append((T_local + T_next) / 2.0)
        Q_total += Q_i
        T_local = T_next

    return {
        "T_out": T_local,
        "Q_total": Q_total,
        "T_fluid_profile": np.array(T_fluid_profile),
    }
