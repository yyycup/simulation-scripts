import numpy as np


class AgingModel280Ah:
    """
    280Ah LFP电池老化计算模型 (微分增量版)

    基于文献: Lv, Z., et al. Materials 2025, 18, 1342
    用途: 在单次仿真中，根据实时电流和温度，计算累积的老化损伤(1-SOH)。
    """

    def __init__(self, initial_cycle_age=0.0, capacity_ah=280.0):
        """
        初始化模型
        :param initial_cycle_age: 电池在仿真开始前的历史循环次数 (默认为 0.0，即全新电池)
        :param capacity_ah: 额定容量 (默认 280Ah)
        """
        self.capacity = capacity_ah

        # 状态变量
        self.total_cycles = initial_cycle_age
        self.accumulated_loss = 0.0  # 累积容量损失 (%)，即 1-SOH

    def _interp_params(self, temp_c):
        """内部工具: 根据温度插值获取当前时刻的 a, b, G"""
        # 核心机理参数 (源自文献图11/12)
        ref_temps = np.array([25.0, 35.0, 45.0])
        lli_params = {
            'a': np.array([0.002434, 0.004025, 0.003642]),
            'b': np.array([0.4513, 0.4500, 0.4870])
        }
        lam_params = {
            'a': np.array([2.661e-05, 3.789e-05, 5.241e-05]),
            'b': np.array([0.7800, 0.8060, 0.8527]),
            'G': np.array([7000.0, 5000.0, 3500.0])
        }

        t = np.clip(temp_c, 20.0, 55.0)
        p_lli = {k: np.interp(t, ref_temps, v) for k, v in lli_params.items()}
        p_lam = {k: np.interp(t, ref_temps, v) for k, v in lam_params.items()}
        return p_lli, p_lam

    def step(self, current_amps, temp_c, dt_seconds):
        """
        执行单步计算
        :return: 当前的 SOH (%)
        """
        dx = (np.abs(current_amps) * dt_seconds) / (2.0 * self.capacity * 3600.0)
        p_lli, p_lam = self._interp_params(temp_c)

        x_safe = np.maximum(self.total_cycles, 1e-6)
        d_lli = (p_lli['a'] * p_lli['b'] * (x_safe ** (p_lli['b'] - 1.0))) * dx

        d_lam = np.zeros_like(d_lli)
        is_gassing = self.total_cycles > p_lam['G']

        if np.any(is_gassing):
            if np.isscalar(self.total_cycles):
                if is_gassing:
                    x_eff = max(self.total_cycles - p_lam['G'], 1e-6)
                    d_lam = (p_lam['a'] * p_lam['b'] * (x_eff ** (p_lam['b'] - 1.0))) * dx
            else:
                x_eff = np.maximum(self.total_cycles[is_gassing] - p_lam['G'][is_gassing], 1e-6)
                d_lam[is_gassing] = (p_lam['a'][is_gassing] * p_lam['b'][is_gassing] * (
                            x_eff ** (p_lam['b'][is_gassing] - 1.0))) * dx[is_gassing]

        self.accumulated_loss += d_lli + d_lam
        self.total_cycles += dx

        # 返回 SOH (%) = 100% - 损失百分比
        return 100.0 - self.accumulated_loss
