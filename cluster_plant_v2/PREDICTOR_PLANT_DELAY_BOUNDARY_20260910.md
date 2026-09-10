# Plant 与 Physics-P 预测器：延迟结构不匹配的边界声明

**日期**：2026-09-10
**触发**：2026-09-08/09 的输运升级把 Plant 延迟由 15/20 s 重新定版为 5/5 s
（`HEAT_CURRENT_MASS_TRANSPORT_20260908.md` §2026-09-09）。
**性质**：只读审计 + 边界声明。**本文件不修改任何代码。**

当前 Plant 的定版依据和验证结果见
`validation/results/heat_current_mass_transport_20260909_delay5s_final/STAGE5_FINAL_VALIDATION_REPORT.md`。

---

## 1. 事实：两条延迟链现在不一致

| 组件 | 供液 | 回液 | 5 s 滞后状态 | 是否跟随新定版 |
|---|---|---|---|---|
| `ClusterPlant` / `HeatCurrentPlant` | **5 s** | **5 s** | **1 + 1** | ✅ 本次定版 |
| `control/cluster_domp_model.py`（簇级 do-mpc） | 5 s | 5 s | 1 + 1 | ✅ **自动跟随**（`SUPPLY_LAG_COUNT = round(delay/5)`） |
| **Physics-P 14 状态预测器** | **15 s** | **20 s** | **3 + 4** | ❌ **冻结，未跟随** |
| `control/physics_p_nmpc_model.py`（簇级 16 状态） | 15 s | 20 s | 3 + 4 | ❌ 基于上述预测器 |

来源：`single_pack_plant/predictor/physics_p.py:38-39`

```python
"supply_delay_s": 15.0,
"return_delay_s": 20.0,
```

---

## 2. 为什么预测器**不能简单改数字**

`single_pack_plant/predictor/state_space.py:211-222` 存在**硬结构校验**：

```python
expected = {
    "evap_input_delay_s": 0,
    "pump_flow_delay_s": 1,
    "supply_delay_s": 3,      # 必须 3 步
    "return_delay_s": 4,      # 必须 4 步
}
if actual_steps != expected_steps:
    raise PhysicsArtifactError(...)
```

且返回状态切片与之绑定：

```python
Z_RETURN = slice(10, 14)      # 状态 10–13 = 4 个回液滞后
```

⇒ **3 + 4 是"14 状态 Physics-P"的结构性身份**。把它改成 1 + 1，
状态数、切片、识别产物（`physics_p_operational_v1.json`）全部失配，
得到的是**另一个模型**，必须走一遍识别/重标定流程，不能靠改常数完成。

---

## 3. 影响范围：以下闭环结论**作废**（需重标定后重验）

| 位置 | 说明 |
|---|---|
| `single_pack_plant/controllers/dompc/nmpc.py` | 单 Pack do-mpc，14 状态 |
| `single_pack_plant/controllers/fixed_qp/mpc.py` | 单 Pack LTV QP-MPC |
| `single_pack_plant/predictor/physics_p_casadi.py` | CasADi 版同源 |
| `cluster_plant_v2/control/physics_p_nmpc_model.py` | 簇级 16 状态（14 + 2 增广） |
| `cluster_plant_v2/validation/validate_physics_p_nmpc.py` | 该预测器的验证入口 |

**作废的含义**：这些控制器/预测器与**当前 Plant**（5/5 s）配对时，
"控制器脑内世界"的输运滞后比真实 Plant **长 10–15 s**。
任何**闭环性能结论**（跟踪误差、能耗、温控指标）都建立在这个失配之上，
**不能作为对当前 Plant 的最终结论引用**。

> ⚠️ 这条边界由 `HEAT_CURRENT_MASS_TRANSPORT_20260908.md:45` 主动声明：
> "冻结的 Physics-P 14 状态预测器仍保留原 3+4 延迟状态，不属于本次
> Plant/HC 定版范围，相关闭环结论需在预测器重标定后重新验证。"
> 本文件把该声明**独立成篇**并补上技术根因与影响清单。

---

## 4. **仍然有效**的结论（不受影响）

| 对象 | 状态 |
|---|---|
| `ClusterPlant` / `HeatCurrentPlant` 开环动态 | ✅ 有效 |
| Stage 5 九工况 V1–V9（`heat_current_mass_transport_20260909_delay5s_final/`） | ✅ 有效，9/9 通过 |
| 能量账本（量纲修复后 R ≤ 2.7e-8 W） | ✅ 有效 |
| `CoolantMassTransport` 质量守恒 | ✅ 有效（管内质量变化 0.0 kg） |
| 冷板 / 蒸发器 / 泵 / 水箱 / 循环等部件方程 | ✅ 未改动 |

**一句话**：**Plant 侧全部有效；控制侧（Physics-P 预测器）闭环结论待重标定。**

---

## 5. 后续工作（未执行，需另行立项）

1. **重标定预测器**：目标延迟结构 1 + 1（或与 5/5 s 等价的结构）。
   需要重新识别，并产出新的 `physics_p_operational_v2.json` 类产物。
2. **同步状态切片**：`Z_RETURN` 等依赖状态布局的量需随新结构重定义。
3. **更新 CasADi 版**：`physics_p_casadi.py` 与 Python 版须保持一致。
4. **重跑闭环验证**：`validate_physics_p_nmpc.py` 及单 Pack 侧全部闭环用例。

参考：仓库已有 `codex/dual-mpc-predictors` 工作树
（`.worktrees/dual-mpc-predictors`，最新 `672721c`）在推进预测器相关工作，
重标定可能与该项合并处理。

---

## 6. 风险提示：**没有测试守护这条一致性**

全量回归 **283/283 通过**，但其中**没有任何用例**校验
"Plant 延迟 == 预测器延迟"。因此：

> 即使两条延迟链相差 10–15 s，测试仍会全绿。

建议在重标定完成后，补一条**跨模块一致性测试**：

```text
assert round(PLANT_SUPPLY_DELAY_S / dt) == predictor.expected_supply_steps
assert round(PLANT_RETURN_DELAY_S / dt) == predictor.expected_return_steps
```

否则同类静默失配可以再次发生。
