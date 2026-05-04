# SAC Path Tracking Bicycle Controller

本工程在原有自行车模型路径跟踪项目上扩展了一个基于 Soft Actor-Critic 的强化学习路径跟踪控制器。控制器输入车辆状态和参考路径误差，输出前轮转向角，使智能车跟踪双移线、蛇形线和圆形路径。

## 1. 安装与运行

```bash
python3 -m pip install -r requirements.txt
```

快速检查训练链路：

```bash
python3 train_sac.py --smoke-test
```

推荐中等训练预算：

```bash
python3 train_sac.py --timesteps 300000 --path mixed --model-path models/sac_path_tracking.zip
```

评估 SAC 与 PID/Stanley 对比控制器：

```bash
python3 evaluate.py --model-path models/sac_path_tracking.zip --scenario all
```

如果还没有训练模型，`evaluate.py` 会先生成 PID/Stanley 基线结果。

动态仿真，而不是静态评估：

```bash
python3 simulate.py --controller sac --model-path models/sac_path_tracking.zip --path snake --scenario nominal
```

保存动态 GIF：

```bash
python3 simulate.py --controller sac --model-path models/sac_path_tracking.zip --path snake --scenario nominal --save results/sac_snake.gif --no-display
```

PID/Stanley 动态对比：

```bash
python3 simulate.py --controller pid --path double_lane_change --scenario side_wind
```

同时展示 SAC 与 PID/Stanley，并同时展示三种路径：

```bash
python3 simulate.py --compare --all-paths --model-path models/sac_path_tracking.zip --scenario nominal
```

保存组合动态 GIF：

```bash
python3 simulate.py --compare --all-paths --model-path models/sac_path_tracking.zip --scenario nominal --save results/sac_vs_pid_all_paths.gif --no-display
```

## 2. 工程结构

- `vehicle_dynamics.py`：二自由度动态自行车模型，支持侧风、低附着系数、车速变化。
- `paths.py`：双移线、蛇形线、圆形路径生成，以及参考航向、曲率、横向误差计算。
- `path_tracking_env.py`：Gymnasium 自定义环境，动作为前轮转角。
- `train_sac.py`：Stable-Baselines3 SAC 训练入口，保存模型和训练曲线。
- `evaluate.py`：测试 SAC 与 PID/Stanley，输出轨迹图、误差图、转角图、指标表。
- `simulate.py`：动态仿真入口，实时显示车辆运动、历史轨迹、横向误差、转向角和速度。
- `baseline_pid.py`：Stanley 横向控制器和速度一阶跟踪基线。
- `plot_results.py`：训练曲线、轨迹、误差、转角和性能表绘制工具。
- `bicyclemodel.py`、`controller2d.py`、`main.py`：保留原项目代码，便于对照。

## 3. 二自由度自行车模型

状态量：

```text
X = [x, y, psi, vx, vy, r]
```

其中 `x, y` 是质心位置，`psi` 是航向角，`vx` 是纵向速度，`vy` 是侧向速度，`r` 是横摆角速度。

控制量：

```text
u = delta
```

其中 `delta` 是前轮转向角，约束为 `[-30 deg, 30 deg]`。

轮胎侧偏角：

```text
alpha_f = atan2(vy + lf * r, vx) - delta
alpha_r = atan2(vy - lr * r, vx)
```

线性侧偏刚度轮胎力：

```text
Fyf = -Cf * alpha_f
Fyr = -Cr * alpha_r
```

低附着路面下会根据 `mu * m * g` 对前后轮侧向力做轻量饱和。

车辆微分方程：

```text
x_dot   = vx * cos(psi) - vy * sin(psi)
y_dot   = vx * sin(psi) + vy * cos(psi)
psi_dot = r
vx_dot  = (v_ref - vx) / tau
vy_dot  = (Fyf * cos(delta) + Fyr + Fwind) / m - vx * r
r_dot   = (lf * Fyf * cos(delta) - lr * Fyr) / Iz
```

`Fwind` 用于侧风扰动，`v_ref` 可固定也可随路径缓慢变化。

## 4. 强化学习环境

观测量为 8 维连续向量：

```text
[横向误差, 航向误差, 侧向速度, 横摆角速度, 车速, 参考曲率, 上一时刻转角, 路径进度]
```

奖励函数主要包含：

- 前进进度奖励。
- 横向误差惩罚。
- 航向误差惩罚。
- 横摆角速度惩罚。
- 转向角和转向变化率惩罚。
- 冲出道路或航向误差过大时提前终止并扣分。

目标是横向误差小、转向动作平滑，并能抵抗轻微侧风、低附着系数和车速变化。

## 5. SAC 训练参数

默认参数位于 `train_sac.py`：

```text
learning_rate = 3e-4
gamma = 0.99
buffer_size = 300000
batch_size = 256
tau = 0.005
learning_starts = 5000
train_freq = 1
gradient_steps = 1
ent_coef = auto
policy net_arch = [256, 256]
```

调参建议：

- 如果奖励曲线震荡大，优先降低 `learning_rate` 到 `1e-4`。
- 如果早期探索不足，增加 `learning_starts` 或保持 `ent_coef=auto`。
- 如果跟踪误差小但转向抖动，增加奖励中的转角变化率权重，或降低最大转向速率。
- 如果圆形路径收敛差，使用 `--path mixed` 训练，并提高总步数。

训练过程通常表现为：初期动作随机且误差较大，中期策略学会沿路径前进，后期累计奖励趋稳、平均横向误差下降。

## 6. 实验与输出文件

评估场景：

- 基准路径：双移线、蛇形线、圆形路径。
- 扰动测试：低附着路面、侧风、车速变化。

评价指标：

- `rmse_m`：横向误差均方根。
- `max_abs_error_m`：最大横向误差。
- `steer_rate_std_rad_s`：转向角变化率标准差。
- `avg_decision_time_ms`：单步平均决策时间。

结果文件输出到 `results/`：

- `training_reward_curve.png`
- `training_error_curve.png`
- `trajectory_SAC_*.png`
- `trajectory_PID_Stanley_*.png`
- `lateral_error_comparison_*.png`
- `steering_SAC_*.png`
- `steering_PID_Stanley_*.png`
- `performance_metrics.csv`
- `performance_metrics.md`

训练好的 SAC 模型默认保存为：

```text
models/sac_path_tracking.zip
```
