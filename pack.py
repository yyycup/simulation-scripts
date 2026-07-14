import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

import platformdirs

WORKSPACE = Path(__file__).resolve().parent
HPPC_PARAMS_PATH = WORKSPACE / "model_data" / "hppc_params.json"
platformdirs.user_config_dir = lambda *args, **kwargs: str(WORKSPACE / "outputs" / "pybamm_config")

import pybamm
from scipy.interpolate import RegularGridInterpolator

class BatteryPack:
    def __init__(self, config):
        self.config = config
        self.rows = config.get('rows', 4)
        self.cols = config.get('cols', 13)
        self.n_parallel = self.rows
        self.n_series = self.cols
        self.Ns = self.rows * self.cols
        self.total_current = config.get('total_current', config.get('current', 0.0))

        # 1. 获取 HPPC 数据
        self._hppc_data = _get_hppc_data_formatted()

        # 2. 构建 Python 插值器
        self._build_python_interpolators()

        # 3. 构建 PyBaMM 模型
        self._build_models_and_base_parameters()

        # 4. 应用热网络
        self._apply_thermal_network()

        # 5. 创建仿真
        self._create_simulation_with_dynamic_params()

        # 6. 初始化状态
        self.temps = np.full(self.Ns, self.config['initial_temp_c'] + 273.15)
        self.socs = np.full(self.Ns, self.config['initial_soc'])
        self.branch_currents_history = [[] for _ in range(self.rows)]
        self.history = [[] for _ in range(self.Ns)]

    @property
    def current(self):
        return self.total_current

    @current.setter
    def current(self, value):
        self.total_current = value

    def _build_python_interpolators(self):
        d = self._hppc_data
        soc_axis = d['soc']
        temp_axis = d['temp']

        R_tot_dis = d['r0_dis'] + d['r1_dis'] + d['r2_dis']
        R_tot_chg = d['r0_chg'] + d['r1_chg'] + d['r2_chg']

        # === 维度检查与修复 ===
        expected_shape = (len(soc_axis), len(temp_axis))
        if R_tot_dis.shape != expected_shape:
            try:
                R_tot_dis = R_tot_dis.reshape(expected_shape)
                R_tot_chg = R_tot_chg.reshape(expected_shape)
            except ValueError as e:
                raise ValueError(f"数据维度不匹配且无法修复: {e}")

        points = (soc_axis, temp_axis)
        self.interp_R_dis = RegularGridInterpolator(points, R_tot_dis, bounds_error=False, fill_value=None)
        self.interp_R_chg = RegularGridInterpolator(points, R_tot_chg, bounds_error=False, fill_value=None)

    def _build_models_and_base_parameters(self):
        self.models = [pybamm.equivalent_circuit.Thevenin(
            {"number of rc elements": 2, "thermal": "lumped"}
        ).new_copy() for _ in range(self.Ns)]

        self.base_parameter_values = pybamm.ParameterValues({
            "Cell capacity [A.h]": self.config['capacity'],
            "Nominal cell capacity [A.h]": self.config['capacity'],
            "Initial SoC": self.config['initial_soc'],
            "Ambient temperature [K]": "[input]",
            "Current function [A]": "[input]",
            "Initial temperature [K]": self.config['initial_temp_c'] + 273.15,
            "Upper voltage cut-off [V]": 4.5, "Lower voltage cut-off [V]": 2.5,
            "Cell thermal mass [J/K]": self.config['cell_thermal_mass'],
            "Entropic change [V/K]": 0.0,
            "Element-1 initial overpotential [V]": 0.0, "Element-2 initial overpotential [V]": 0.0,
            "Cell-jig heat transfer coefficient [W/K]": 0.0,
            "Jig thermal mass [J/K]": 500.0, "Jig-air heat transfer coefficient [W/K]": 0.0,
        })

    def _apply_thermal_network(self):
        T_plate_K = pybamm.InputParameter("Cooling plate temperature [K]")
        T_amb = pybamm.InputParameter("Ambient temperature [K]")
        temps_vars = [m.variables["Cell temperature [K]"] for m in self.models]

        for i in range(self.Ns):
            model = self.models[i]
            total_heat = self._calculate_heat_exchange(i, temps_vars[i], temps_vars, T_plate_K, T_amb)
            key = [v for v in model.rhs.keys() if "Cell temperature" in v.name][0]
            model.rhs[key] = model.rhs[key] + total_heat / self.base_parameter_values["Cell thermal mass [J/K]"]

    def _calculate_heat_exchange(self, i, T_cell, temps_all, T_plate, T_amb):
        r, c = self.rows, self.cols
        cfg = self.config
        row_idx, col_idx = i // c, i % c
        Q = pybamm.Scalar(0)

        # 1. 内部互连 (电池间导热)
        if i % c < c - 1 and i + 1 < self.Ns: Q += cfg['k_intercell'] * (temps_all[i + 1] - T_cell)
        if i % c > 0: Q += cfg['k_intercell'] * (temps_all[i - 1] - T_cell)
        if i // c < r - 1 and i + c < self.Ns: Q += cfg['k_intercell'] * (temps_all[i + c] - T_cell)
        if i // c > 0: Q += cfg['k_intercell'] * (temps_all[i - c] - T_cell)

        # 2. 底部冷板换热
        Q -= cfg['h_plate_convection'] * (T_cell - T_plate)

        # 3. 空气对流散热 (带开关控制)
        if cfg.get('uniform_air_convection', False):
            total_air_factor = 1.0
        else:
            row_factors = [1.5, 1.0, 0.7, 1.2] 
            row_f = row_factors[row_idx] if row_idx < len(row_factors) else 1.0
            col_f = 1.0 - 0.4 * (col_idx / (c - 1)) 
            edge_boost = 1.8 if (row_idx == 0 or row_idx == r - 1 or col_idx == 0 or col_idx == c - 1) else 1.0
            total_air_factor = row_f * col_f * edge_boost
        
        Q -= (cfg['h_air_convection_edge'] * total_air_factor) * (T_cell - T_amb)
        
        return Q

    def _create_simulation_with_dynamic_params(self):
        self.sims = []
        p = self._hppc_data
        for i, model in enumerate(self.models):
            params = self.base_parameter_values.copy()
            soc = model.variables["SoC"]
            temp = model.variables["Cell temperature [K]"]
            curr = model.variables["Current [A]"]
            s = 0.5 * (1 + pybamm.tanh(0.1 * curr))

            def get_interp(key):
                return pybamm.Interpolant((p['soc'], p['temp']), p[key], (soc, temp), name=f"{key}_{i}")

            params.update({
                "Open-circuit voltage [V]": get_interp('ocv'),
                "R0 [Ohm]": s * get_interp('r0_dis') + (1 - s) * get_interp('r0_chg'),
                "R1 [Ohm]": s * get_interp('r1_dis') + (1 - s) * get_interp('r1_chg'),
                "C1 [F]": s * get_interp('c1_dis') + (1 - s) * get_interp('c1_chg'),
                "R2 [Ohm]": s * get_interp('r2_dis') + (1 - s) * get_interp('r2_chg'),
                "C2 [F]": s * get_interp('c2_dis') + (1 - s) * get_interp('c2_chg'),
            }, check_already_exists=False)

            self.sims.append(pybamm.Simulation(model, parameter_values=params, solver=pybamm.CasadiSolver(mode="safe")))

    def calculate_current_distribution(self):
        # [核心修改] 电-热耦合开关
        if self.config.get('dynamic_resistance_update', True):
            # 实时更新：根据当前 SOC 和温度计算内阻 (电-热强耦合)
            pts = np.column_stack((self.socs, self.temps))
        else:
            # 固定参数：使用初始 SOC 和 25°C 计算内阻 (不随温度更新)
            static_soc = np.full(self.Ns, self.config['initial_soc'])
            static_temp = np.full(self.Ns, 298.15)
            pts = np.column_stack((static_soc, static_temp))

        R_cells = self.interp_R_dis(pts) if self.total_current >= 0 else self.interp_R_chg(pts)
        R_branches = np.sum(R_cells.reshape((self.rows, self.cols)), axis=1)
        G_branches = 1.0 / (R_branches + 1e-9)
        if np.sum(G_branches) == 0: return np.full(self.rows, self.total_current / self.rows)
        return self.total_current * (G_branches / np.sum(G_branches))

    def step(self, dt, T_plate, T_cabinet=298.15):
        I_branches = self.calculate_current_distribution()
        for k in range(self.rows): self.branch_currents_history[k].append(I_branches[k])

        if np.isscalar(T_plate):
            T_plate_arr = np.full(self.cols, T_plate)
        else:
            T_plate_arr = T_plate

        for i, sim in enumerate(self.sims):
            r, c = i // self.cols, i % self.cols
            T_plate_local = T_plate_arr[c]
            sol = sim.step(dt, inputs={
                "Cooling plate temperature [K]": T_plate_local,
                "Current function [A]": I_branches[r],
                "Ambient temperature [K]": T_cabinet
            })
            self.temps[i] = float(sol["Cell temperature [K]"].data[-1])
            self.socs[i] = float(sol["SoC"].data[-1])
            self.history[i].append(sol)

    def get_avg_temp(self):
        return float(np.mean(self.temps))

    def plot_cell(self, idx):
        if hasattr(self, "history") and self.history[idx]:
            combined_sol = self.history[idx][0]
            for i in range(1, len(self.history[idx])):
                combined_sol = combined_sol + self.history[idx][i]
            pybamm.dynamic_plot(combined_sol)

# --- 数据加载 ---
def _get_hppc_data_formatted():
    try:
        with HPPC_PARAMS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        soc = np.array(data['soc'])
        temp = np.array(data['temp'])
        n_s, n_t = len(soc), len(temp)

        def fix_shape(key, scale=1.0):
            arr = np.array(data[key]) * scale
            if arr.ndim == 1:
                if arr.size == n_s * n_t: return arr.reshape((n_s, n_t))
                if key == 'ocv' and arr.size == n_s: return arr.reshape(-1, 1) * np.ones((1, n_t))
            return arr

        return {
            'soc': soc, 'temp': temp,
            'ocv': fix_shape('ocv'),
            'r0_dis': fix_shape('r0_dis', 1e-3), 'r0_chg': fix_shape('r0_chg', 1e-3),
            'r1_dis': fix_shape('r1_dis', 1e-3), 'r1_chg': fix_shape('r1_chg', 1e-3),
            'c1_dis': fix_shape('c1_dis'), 'c1_chg': fix_shape('c1_chg'),
            'r2_dis': fix_shape('r2_dis', 1e-3), 'r2_chg': fix_shape('r2_chg', 1e-3),
            'c2_dis': fix_shape('c2_dis'), 'c2_chg': fix_shape('c2_chg')
        }
    except Exception as e:
        soc, temp_c = np.linspace(0, 1, 20), np.linspace(-10, 50, 7)
        temp = temp_c + 273.15
        R_map = np.zeros((20, 7))
        for i, s in enumerate(soc):
            for j, t in enumerate(temp_c):
                R_map[i, j] = 0.003 * (1.0 - 0.015 * (t - 25)) * (1.0 + 1.5 * (s - 0.5) ** 4)
        ocv = (3.0 + 0.6 * soc).reshape(-1, 1) * np.ones((1, 7))
        return {
            'soc': soc, 'temp': temp, 'ocv': ocv,
            'r0_dis': R_map, 'r0_chg': R_map, 'r1_dis': R_map * 0.2, 'r1_chg': R_map * 0.2,
            'c1_dis': np.ones((20, 7)) * 2000, 'c1_chg': np.ones((20, 7)) * 2000,
            'r2_dis': R_map * 0.3, 'r2_chg': R_map * 0.3, 'c2_dis': np.ones((20, 7)) * 1e4, 'c2_chg': np.ones((20, 7)) * 1e4
        }
