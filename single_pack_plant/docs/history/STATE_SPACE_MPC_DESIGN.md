# 独立 State-Space QP-MPC 设计审计

## 1. 任务边界与结论

本设计只替换 MPC 的数学求解层，不修改以下内容：

- `BatteryPack` 与制冷/冷却回路 Plant；
- 已验证的 Physics-P artifact；
- Physics-P 的热物理方程、参数和输入域；
- 现有 GEKKO MPC；
- 流向切换逻辑；
- 工况、初始条件和数据记录口径。

审计结论：首版应采用**在线离散 LTV + 直接 OSQP**，不采用单一固定工作点 LTI。原因是 Physics-P 同时包含三次制冷量曲面、300–1000 rpm 启动分段、泵流量非线性以及离散时滞；一个覆盖全部工况的固定线性模型风险过高。

审计比较过三种路线：固定工作点 LTI 最简单但适用域不足；多工作点增益调度 LTI 需要新增分区和切换逻辑；在线离散 LTV 直接复用当前 P 的一步映射，新增假设最少，因此选 LTV。

推荐的首版最小状态为 14 维。完整 Physics-P 紧凑状态为 15 维；其中 `q_cond` 在当前 artifact 的 `evap_response_model="direct"` 下不影响其他状态和受控输出，因此可从控制模型中删除。若以后更换为 `cascaded` artifact，必须恢复 `q_cond`，不得继续使用 14 维模型。

本轮仅生成设计，不实现控制器。

## 2. 当前真实架构边界

### 2.1 Plant

真实 Plant 调用链为：

```text
thermal_case_simulator.py::simulate_case()
  -> controller.command()
  -> thermal_loop.py::simulate_thermal_loop_step()
  -> pack.py::BatteryPack.step()
  -> controller.update_after_step()
```

主要职责：

- `pack.py::BatteryPack`：52 个电芯的 PyBaMM Thevenin 2RC 电热状态和空间温度场；
- `thermal_system.py`：泵、风机、压缩机、制冷循环和冷板流体换热；
- `thermal_loop.py::simulate_thermal_loop_step()`：执行器惯性、制冷量惯性、管路时滞、冷板、冷却液箱和正反向流动；
- `thermal_case_simulator.py::simulate_case()`：每 5 s 依次调用控制器、Plant、记录器。

Physics-P 不是 Plant。新 QP-MPC 只能读取 Plant 的公开测量/状态，不得把 Physics-P 的状态更新写回 Plant。

### 2.2 Physics-P

权威离散预测接口：

```text
mpc_physics_predictor.py::PhysicsPredictorState
mpc_physics_predictor.py::initialize_physics_state()
mpc_physics_predictor.py::step_physics_predictor()
```

闭环 GEKKO MPC 在 `mpc_flow_direction_strategies.py::MPCControllerDual` 中重新表达了相同 artifact 的动态和热平衡，但使用 GEKKO 动态方程、延迟算子和 NLP 求解。因此：

- 新 State-Space MPC 的非线性基准应是 `step_physics_predictor()`；
- 现有 GEKKO Physics-P MPC 是闭环 benchmark；
- 两者不要求离散轨迹逐点完全相同，但紧凑状态更新必须先与 `step_physics_predictor()` 做数值等价验证。

### 2.3 当前 artifact

文件：`model_data/physics_p_operational_v1.json`

```text
model_type              physics_p
schema_version          1
fit_status              validated
capacity model          single_cubic_v1
evap_response_model     direct
dt                      5 s（闭环控制周期）

tau_comp                3.86066832962019 s
tau_pump                4.23880013788006 s
evap input delay        0 s
pump flow delay         5 s
tau_cond                28.1077993776323 s
tau_evap                45 s
supply delay            15 s
return delay            20 s

active compressor range 1000–6000 rpm
startup extension       300–1000 rpm
pump domain             1600–4800 rpm
coolant domain          15–35 degC
ambient domain          20–40 degC
```

重要热参数仍从 artifact 读取，不在 State-Space MPC 内复制成第二套默认值。

## 3. Physics-P 实际状态审计

`PhysicsPredictorState` 当前包含：

| 字段 | 含义 | 在当前 artifact 中是否影响控制链 |
|---|---|---|
| `n_comp_eff_rpm` | 压缩机实际/有效转速 | 是 |
| `n_pump_eff_rpm` | 水泵实际/有效转速 | 是 |
| `evap_speed_history_rpm` | 蒸发器输入时滞队列 | 当前长度 0 |
| `pump_speed_history_rpm` | 泵流量时滞队列 | 是，长度 1 |
| `q_cond_w` | 冷凝器热量动态 | 否；`direct` 模式下不反馈 |
| `q_evap_w` | 蒸发器有效制冷量 | 是 |
| `supply_history_c` | 供水管路时滞队列 | 是，长度 3 |
| `t_supply_c` | 供水温度输出 | 是，但可由供水队列首元素得到 |
| `t_plate_c` | 平均冷板温度 | 是 |
| `return_history_c` | 回水管路时滞队列 | 是，长度 4 |
| `t_return_c` | 回水温度输出 | 是，但可由回水队列首元素得到 |
| `t_batt_c` | 电池平均温度 | 是 |
| `t_cool_c` | 冷却液箱温度 | 是 |

若把 tuple 历史展开，并删除可由队列直接读出的重复 `t_supply_c/t_return_c`，完整紧凑实现为 15 个标量状态：

```text
7 个热/执行器核心状态（t_bat、t_tank、t_plate、n_comp、n_pump、q_cond、q_evap）
+ 1 个泵时滞状态
+ 3 个供水时滞状态
+ 4 个回水时滞状态
= 15
```

更明确地写为：

```text
[t_bat, t_tank, t_plate, n_comp, n_pump, q_cond, q_evap,
 z_pump_1,
 z_supply_1, z_supply_2, z_supply_3,
 z_return_1, z_return_2, z_return_3, z_return_4]
```

当前 artifact 采用 `direct` 蒸发器响应：

```text
q_evap(k+1) <- q_steady
```

而不是：

```text
q_evap(k+1) <- q_cond(k)
```

因此 `q_cond` 是一个不影响受控温度、其他保留状态和控制量的旁路诊断状态。首版最小 QP 状态删除它。

## 4. 建议的 x、u、d、y

### 4.1 最小状态 x（14 维）

```text
x = [
  t_bat,
  t_tank,
  t_plate,
  n_comp,
  n_pump,
  q_evap,
  z_pump_1,
  z_supply_1, z_supply_2, z_supply_3,
  z_return_1, z_return_2, z_return_3, z_return_4
]^T
```

其中：

```text
t_supply = z_supply_1
t_return = z_return_1
n_pump_delayed = z_pump_1
```

单位：温度使用 degC，转速使用 rpm，热量率使用 W。

命名映射：`t_tank` 就是 Physics-P 字段 `t_cool_c`，`t_plate` 是 Plant 13 个冷板节点的平均温度，不是新增物理节点。

| 公式符号 | 代码名 | 含义 |
|---|---|---|
| `T_b` | `t_bat` | 电池平均温度 |
| `T_t` | `t_tank` | 冷却液箱温度 |
| `T_p` | `t_plate` | 冷板平均温度 |
| `T_s` | `t_supply` | 供水温度 |
| `T_r` | `t_return` | 回水温度 |
| `n_c` | `n_comp` | 压缩机实际转速 |
| `n_p` | `n_pump` | 水泵实际转速 |
| `q_e` | `q_evap` | 蒸发器制冷量 |
| `q_c` | `q_cond` | 冷凝器热量率；当前 `direct` artifact 下仅保留为完整模型诊断量，不进入 14 维最小状态 |

首版只兼容当前已验证 artifact 的以下结构：

```text
evap_response_model = direct
evap_input_delay_s  = 0
pump_flow_delay_s   = 5
supply_delay_s      = 15
return_delay_s      = 20
dt                  = 5 s
```

结构不匹配时应明确拒绝初始化，不能静默改变状态维数。

### 4.2 控制输入 u（2 维）

```text
u = [n_comp_cmd, n_pump_cmd]^T
```

QP 决策量仍然是压缩机和水泵转速指令，不直接优化实际转速、制冷量或流向。

### 4.3 可测扰动 d（2 维）

外部接口和 QP 数学模型统一使用：

```text
d = [I, t_ambient]^T
```

Physics-P 的现有一步函数接收发热率而不是电流。因此离散非线性映射 `f_d` 在内部先使用当前已有定义完成确定性转换：

```text
q_gen = 52 * 0.001 * (I / 4)^2
```

artifact 内部再使用固定的 `battery_heat_generation_scale=0.74`。`q_gen` 不是新增状态，也不是第三个扰动；它只是 `f_d(x,u,d)` 内部由 `I` 计算的确定性中间量。在线线性化时 `E_k=partial f_d/partial d` 直接对 `[I,t_ambient]` 求导，其中电流列通过链式法则包含 `partial q_gen/partial I`。这样既保留真实平方发热关系，又保证运行时接口始终是 `d=[I,T_a]^T`。

### 4.4 受控输出 y

首版受控输出只使用 Physics-P 实际能够预测并且当前 MPC 真正控制的电池平均温度：

```text
y = [t_bat]
y = C_d x
C_d = [1, 0, ..., 0]
```

诊断输出可由状态选择矩阵直接取得：

```text
y_diag = [
  t_bat, t_tank, t_plate, t_supply, t_return,
  n_comp, n_pump, q_evap
]
```

`Tmax` 和 `DeltaT` 不在 Physics-P 标量状态中。首版不得凭空增加相应 predictor 状态；它们继续从 Plant 温度场读取，只用于闭环验证和流向监督。

## 5. 采用离散 LTV，不构造虚假的连续模型

Physics-P 已经有包含离散时滞的 5 s 一步函数。因此首版直接对离散映射线性化：

```text
x(k+1) = f_d(x(k), u(k), d(k))
```

在名义轨迹 `(x_bar_i, u_bar_i, d_bar_i)` 上得到：

```text
A_k = df_d/dx
B_k = df_d/du
E_k = df_d/dd
c_k = f_d(x_bar_k,u_bar_k,d_bar_k)
      - A_k x_bar_k - B_k u_bar_k - E_k d_bar_k
```

于是：

```text
x(k+1) = A_k x(k) + B_k u(k) + E_k d(k) + c_k
y(k)   = C_d x(k)
```

首版不需要 `A_c/B_c/E_c/C_c`。强行先恢复连续模型再离散化会重复处理已存在的离散延迟并引入新误差。如果以后确实建立连续模型，再统一使用 `A_c/B_c/E_c/C_c` 命名。

输出矩阵不随工作点变化，因此可以写成 `C_k=C_d`；在线变化的是 `A_k/B_k/E_k/c_k`。

### 5.1 线性化方法

推荐流程：

1. 用上一周期最优控制序列左移并保持末值，形成名义输入轨迹；
2. 用当前测量状态和已知负载预览滚动 Physics-P，形成名义状态轨迹；
3. 每个预测步对离散一步映射做数值 Jacobian；
4. 在普通平滑区使用中心差分；
5. 在输入边界和 300/1000 rpm 分段点附近使用单边差分；
6. 若扰动或预测状态离开 artifact 输入域，拒绝该 QP 结果并进入现有安全回退逻辑。

不引入 JAX、CasADi 或新的符号模型。Physics-P 很小，NumPy 有限差分足够，也避免维护第三套方程。

### 5.2 数值缩放

QP 内部必须使用无量纲偏差变量：

```text
x_tilde = S_x^-1 (x - x_ref)
u_tilde = S_u^-1 (u - u_min)
d_tilde = S_d^-1 (d - d_ref)
```

这样可避免温度约 25、转速数千、制冷量数千同时进入 Hessian。以后 TD3 调节权重时，倍率才具有稳定物理含义。

缩放只影响 QP 数值，不改变 Physics-P 物理单位和对外接口。

## 6. 时滞增广

当前闭环 `dt=5 s`，因此：

```text
pump flow delay  5 s  -> 1 个移位状态
supply delay    15 s  -> 3 个移位状态
return delay    20 s  -> 4 个移位状态
evap input delay 0 s  -> 不增广
```

移位队列采用精确线性更新，不需要有限差分：

```text
z_1(k+1) = z_2(k)
...
z_m(k+1) = new_input(k)
delayed_output(k) = z_1(k)
```

当前 GEKKO Physics-P 每个控制周期把供/回水延迟偏差重新锚定到 Plant 测量，Shadow Predictor 也从当前测量重新初始化历史。为了公平比较，首版 State-Space MPC保持这一语义：

- `t_supply/t_return` 队列由当前实测供/回水温度填充；
- 泵延迟状态由当前实测 `n_pump` 填充；
- 不额外设计状态观测器；
- 不把上一周期预测队列当作真实 Plant 状态。

当前正式参数中 `physics_p_temp_bias_gain=0`，所以不需要增加 bias 状态。若以后启用预测偏差补偿，应把它作为输出仿射偏置处理，而不是加入热状态向量。

以后若要使用 Plant 的完整供/回水历史，必须作为单独的 predictor-initialization 改进验证，不能混入首版求解器对比。

## 7. 标准 QP-MPC

### 7.1 预测展开

对 LTV 模型递推构造：

```text
X = Phi x_k + Gamma U + Gamma_d D + Gamma_c
Y = C_bar X
```

其中：

```text
U = [u_0^T, ..., u_(N_p-1)^T]^T
D = [d_0^T, ..., d_(N_p-1)^T]^T
```

当前运行参数保持：

```text
dt            = 5 s
N_p peak      = 60
N_p frequency = 45
```

为公平对比，调频工况保留当前 Physics-P 的 3 步 move blocking；调峰工况每步可变。move blocking 通过控制量相等约束表达，不修改 `dt` 或预测时域。

### 7.2 目标函数

使用用户指定的标准形式：

```text
J = sum(i=0...N_p-1) [
      e_i^T Q_x e_i
    + u_tilde_i^T R_u u_tilde_i
    + Delta_u_tilde_i^T R_Delta_u Delta_u_tilde_i
    ]
    + e_Np^T P_f e_Np

e_i = y_i - y_ref_i
```

说明：

- `Q_x`：电池平均温度误差权重；
- `R_u`：归一化压缩机/水泵输入强度权重；
- `R_Delta_u`：归一化输入变化权重；
- `P_f`：终端温度误差权重；
- 不使用 GEKKO `WSPLO/WSPHI`；
- 不在首版加入 TD3 或在线学习；
- Plant 的真实压缩机/水泵能耗用于验证；首版 `R_u` 是标准 QP 的输入强度代理，不声称等于非线性真实功率积分。

首版中 `Q_x/P_f` 是 `1x1`，`R_u/R_Delta_u` 是 `2x2` 对角正定矩阵；不开放任意稠密矩阵给 TD3。

对输入使用以下零点和尺度：

```text
n_comp effort zero = 300 rpm
n_pump effort zero = 1600 rpm
```

这样 `R_u` 会倾向于降低设备转速，而不是把转速拉向一个任意工作点。

### 7.3 Hessian 与线性项

将终端权重并入块对角输出权重后：

```text
min_U  1/2 U^T H U + g^T U
```

其中：

```text
H = 2 * (
      Gamma_y^T Q_bar Gamma_y
    + R_u_bar
    + D_u^T R_Delta_u_bar D_u
    )
```

`g` 由当前状态、扰动预览、仿射项、参考温度和上一控制量共同得到。实现必须显式对称化 Hessian，并只加入很小的数值正则项保证 OSQP 数值稳定，不能用大正则掩盖模型错误。

### 7.4 约束

保留当前真实执行器约束：

```text
300  <= n_comp_cmd <= 6000 rpm
1600 <= n_pump_cmd <= 4800 rpm
```

每 5 s 控制步的变化限制：

```text
peak:
  |Delta n_comp| <= 6000 rpm/step
  |Delta n_pump| <= 300 rpm/step

frequency:
  |Delta n_comp| <= 6000 rpm/step
  |Delta n_pump| <= 600 rpm/step
```

QP 约束写成：

```text
G U <= h
```

包括：

- 输入上下界；
- 首步相对上一实际下发指令的变化限制；
- 后续相邻输入变化限制；
- 调频 move-blocking 等式约束；
- Physics-P 冷却液预测域 `15–35 degC`；
- 实际转速状态的物理边界。

环境温度 `20–40 degC` 是扰动有效性检查，不是决策约束。

首版不增加电池温度硬约束。当前 GEKKO MPC 也没有正在生效的电池硬约束；新增硬约束会改变比较问题并可能造成不可行。温度通过 `Q_x/P_f` 调节，`Tmax/DeltaT` 在 Plant 侧验收。

## 8. 求解器与恢复

推荐直接使用 `osqp`，而不是经 CVXPY 建模：

- 当前 BTMS Python 环境已安装 `osqp 1.1.1`、`scipy 1.17.1`；
- 直接 OSQP 可复用稀疏结构并 warm start；
- 当前主 `environment.yml` 只显式声明 SciPy，正式实现时需要把 OSQP 加入可复现依赖；
- CVXPY 可用于离线交叉验证，但不应成为 5 s 闭环的默认执行路径。

每周期流程：

1. 更新线性化矩阵和 QP 数值；
2. 复用 OSQP 问题稀疏结构；
3. 用上一周期左移后的 `U` warm start；
4. 接受 `solved`，并单独记录 `solved inaccurate`；
5. 检查原始约束残差、非有限值和 Physics-P 输入域；
6. 失败时不反复切换求解器，执行明确回退。

回退策略保持当前 Physics-P 语义：

- 一般失败：保持上一周期压缩机/水泵指令并裁剪到边界；
- 预测冷却液低于有效域、实测冷却液到达下界或电池不高于目标时：压缩机 300 rpm、水泵 1600 rpm；
- 记录 `solved/status/iterations/primal residual/dual residual/solve time/fallback reason`。

## 9. Runtime 权重接口

首版只支持运行时更新权重，不改变模型维数、时域、约束或线性化方式。

建议公开接口：

```python
set_runtime_weight_multipliers(
    alpha_q_x: float = 1.0,
    alpha_r_comp: float = 1.0,
    alpha_r_pump: float = 1.0,
)
```

内部映射：

```text
Q_x_runtime       = alpha_q_x    * Q_x_base
R_u_runtime[0,0]  = alpha_r_comp * R_u_base[0,0]
R_u_runtime[1,1]  = alpha_r_pump * R_u_base[1,1]
```

`R_Delta_u` 和 `P_f` 首版固定。接口必须验证倍率为有限正数，并能在不重建状态模型的情况下更新 OSQP Hessian 数值。

后续 TD3 可以只输出这些倍率，但本轮不创建 Actor、Critic、环境、replay buffer 或训练脚本。

## 10. 建议文件结构

新增生产文件控制在两个：

```text
single_pack_plant/
  physics_p_state_space.py
  state_space_qp_mpc.py
```

职责：

### `physics_p_state_space.py`

- artifact 结构检查；
- 14 维状态 pack/unpack；
- 离散非线性一步映射适配；
- 延迟移位矩阵；
- 数值 Jacobian；
- `A_k/B_k/E_k/c_k/C_d`；
- 状态和输入缩放。

### `state_space_qp_mpc.py`

- LTV 预测矩阵；
- QP Hessian、梯度和约束；
- OSQP 初始化、更新、warm start；
- runtime 权重接口；
- `command()`、回退和诊断；
- 与现有控制器相同的压缩机/水泵命令接口。

新增验证入口：

```text
run_state_space_mpc_comparison.py
```

以后实现时只需小范围修改：

```text
thermal_control_strategies.py
thermal_case_simulator.py
```

用于注册 `state_space_mpc` 控制器和记录新诊断列。以下文件必须保持不动：

```text
pack.py
thermal_loop.py
thermal_system.py
mpc_physics_predictor.py
model_data/physics_p_operational_v1.json
```

现有 `mpc_flow_direction_strategies.py` 和 GEKKO MPC 也保留，不删除、不改成 QP 的内部实现。

## 11. 闭环调用链

首版标准流向 benchmark：

```text
simulate_case()
  -> 读取 d(k)=[I(k),T_ambient(k)] 和 Plant 当前测量
  -> StateSpaceQPMPC.command()
       -> 构造 x(k)
       -> 构造 d preview；在 f_d 内部由 I 计算 q_gen
       -> 滚动名义 Physics-P 轨迹
       -> 计算 A_k/B_k/E_k/c_k
       -> 构造 H/g/G/h
       -> OSQP.solve()
       -> 返回 n_comp_cmd、n_pump_cmd
  -> simulate_thermal_loop_step()
  -> BatteryPack.step()
  -> logger
  -> 下一周期
```

流向逻辑不进入 QP 决策变量。首版先使用现有 `standard` 工况验证；通过后再让现有外部流向 supervisor 读取 QP 的温度预测序列，不能在同一阶段同时重写 QP 和流向算法。

## 12. 验证方案

### 12.1 模型级

1. **状态 pack/unpack**：往返严格一致；
2. **紧凑一步等价**：14 维映射与当前 `step_physics_predictor()` 的保留状态逐项比较；
3. **删除 q_cond 证明**：在 `direct` artifact 下改变初始 `q_cond` 不得改变其他下一步状态；
4. **延迟验证**：1/3/4 步脉冲到达时刻准确；
5. **线性化验证**：扰动幅度减半时，局部线性预测误差应呈二阶下降；
6. **artifact guard**：`cascaded` 或不同时延结构必须明确拒绝 14 维模型。

### 12.2 QP 级

1. Hessian 对称、半正定；
2. 输入和变化约束在边界工况下满足；
3. 首步变化约束使用上一实际下发指令；
4. peak/frequency 时域和 move blocking 正确；
5. runtime 权重更新不改变约束、状态或时域；
6. OSQP 失败路径返回可执行的安全指令；
7. 相同输入重复求解结果可重复。

### 12.3 闭环 benchmark

在完全相同的 Plant、Physics-P artifact、初始状态、负载、环境、`dt`、时域和执行器限制下比较：

```text
Fixed GEKKO Physics-P MPC
State-Space LTV QP-MPC
```

工况至少包括：

```text
Constant      600 s
Current step  600 s
RegD          600 s
```

通过后再运行完整 12 工况。输出指标：

```text
Tavg max/mean
Tmax max
DeltaT max/mean
compressor energy
pump energy
total energy
n_comp command/actual
n_pump command/actual
solve time median/p95/max
solved rate
fallback count
constraint maximum violation
Physics-P domain-valid rate
```

首轮建议验收门槛：

- 所有 Plant 状态和控制输出有限；
- solved rate 不低于 99%；
- 转速/变化约束最大违反不超过 `1e-6 rpm`（记录浮点容差后应为零）；
- `Tavg_max/Tmax_max` 相比 GEKKO 不恶化超过 0.1 degC；
- `DeltaT_max` 不恶化超过 0.05 degC；
- 在温控门槛通过的前提下，总能耗不增加超过 5%；
- QP 中位求解时间至少比 GEKKO 快 10 倍；
- QP p99 求解时间小于 5 s 控制周期。

这些是首轮工程验收门槛，不代表最终论文参数。若不通过，先检查线性化和缩放，再调 `Q_x/R_u/R_Delta_u/P_f`；不得通过修改 Plant 或 Physics-P 掩盖问题。

## 13. 主要风险与处理

### 风险 1：启动分段附近 Jacobian 不稳定

处理：分段点附近使用单边差分，限制单周期信赖域，并每 5 s 重新线性化。首版不引入 MIQP。

### 风险 2：输入权重不等于真实能耗

处理：明确把 `R_u` 定义为归一化输入强度代理，真实能耗始终从 Plant 积分。只有在标准 QP 基线通过后，才考虑实际功率的局部二次近似。

### 风险 3：P 状态与 Plant 状态并不完全一致

处理：沿用当前闭环做法，每周期从 Plant 同步电池平均温度、冷却液箱、冷板平均温度、供/回水、实际转速和有效制冷量；不设计新的状态估计器。

### 风险 4：同时改 QP 和流向逻辑导致无法归因

处理：首版只验证 `standard` 流向，外部流向逻辑保持冻结。QP 基线通过后再接现有 supervisor。

### 风险 5：OSQP 当前未写入主环境声明

处理：正式实现时只补充 OSQP 依赖声明，不安装新的建模框架；本轮不改环境文件。

## 14. 本轮停止点

本设计已经明确：

- Physics-P 实际状态和 14 维最小状态；
- `x/u/d/y`；
- 离散 LTV 线性化；
- 1/3/4 步时滞增广；
- 标准 QP 目标、约束和 OSQP；
- runtime 权重接口；
- 文件结构和 GEKKO benchmark；
- 验证门槛。

下一步若获批准，才编写实施计划。当前不实现 State-Space MPC，不修改 Plant、Physics-P、GEKKO MPC 或任何默认参数。
