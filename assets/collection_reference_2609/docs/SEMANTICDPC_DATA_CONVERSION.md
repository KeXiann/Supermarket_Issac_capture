# 身体采集协议 v2 与 SemanticDPC 转换

本次新增字段不改变 BrainCo 配置的 `observation.state[29]` 或 `action.wbc[29]`。
目标模型契约固定为 `semanticdpc_0826` 提交 `04eead325db1ae67f6a74b288fc0bbfbc13ff7fc`。

## 启用新采集

先按现有流程保存 episode、停止身体部署和 PICO manager，再构建并启动新进程：

```bash
export TensorRT_ROOT=/absolute/path/to/TensorRT
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j 4
bash scripts/collect_brainco_zed.sh \
  --task-prompt 'walk to the basket' --dataset-name walk_v2_session_001
```

必须使用新数据集名称；旧 C++ 协议或旧 PICO manager 会被拒绝。采集器、C++、PICO manager
必须在同一 Linux 工作站运行，身体/手命令使用该主机的单调时钟。机器人源 tick 不是 Unix 时间。
现有身体目标与控制方式不变，新增逻辑只记录和转换数据。

## 新增字段

| 字段 | 维度 | 语义 |
|---|---:|---|
| `observation.joint_velocity` | 29 | LowState 电机实测 dq，rad/s，顺序见该字段 names |
| `observation.base_angular_velocity` | 3 | LowState 基座 IMU gyroscope，rad/s；不是 torso IMU |
| `observation.base_acceleration` | 3 | 同一个基座 IMU 的原始 accelerometer，m/s²；不是去重力后的世界系加速度 |
| `observation.lowstate_receive_monotonic_ns` | 1 | C++ 收到该 LowState 快照时的工作站单调时间 |
| `observation.lowstate_tick` | 1 | 原始设备 tick，用于识别重复和重置 |
| `observation.body_source_age_s` | 1 | C++ 状态快照时间减 LowState 接收时间 |
| `observation.body_capture_age_s` | 1 | Python 采样时间减 C++ 状态快照时间 |
| `observation.body_valid` | 1 | 源年龄和采样年龄均不超过 100 ms，时间方向正确 |
| `observation.previous_body_action` | 29 | 上一个连续 logger 周期实际入队的绝对关节目标，rad |
| `observation.previous_body_action_index` | 1 | 上述目标的周期编号；缺失时 -1 |
| `observation.previous_body_action_monotonic_ns` | 1 | 上述目标入队时间；缺失时 0 |
| `observation.previous_body_action_valid` | 1 | 首周期或上一周期没有绑定目标时为 0 |
| `observation.previous_hand_action` | 2 | 在身体状态快照时刻已成功下发的最新完整双手命令，0=开、1=合 |
| `action.aligned_hand` | 2 | 在身体当前动作入队时刻已成功下发的最新完整双手命令 |

两个手命令字段各自带 `_monotonic_ns`、`_timestamp_ns`、`_sequence`、`_session_id` 和 `_valid`。
PICO manager 保留最近 128 条成功的双手命令事件。查询时刻必须被遥测覆盖；事件历史不足时明确标记无效，
不使用未来命令或从手部实测位置反推动作。采集器缓存最近 32 个身体快照，优先选取已被手部遥测覆盖的最新快照。
原有手部实测状态和 `action.hand_binary` 继续保存，仍各有独立时间戳，不冒充身体同周期反馈。

`body_action` / `previous_body_action` 是写入发送缓冲区的目标，手事件表示双侧 DDS Write 成功，
均不代表执行完成或硬件回执。格式错误、NaN/Inf、非法四元数或非因果身体配对会拒绝；
源陈旧或手部历史不完整时使用有效性标记，转换时不作为有效训练样本。

配置中新增 `body_recording_default_q_hardware_json` 和 `body_recording_action_scale_hardware_json`，
保存实际 C++ 动作常量。字符串内容是 JSON 数组，延续现有标量配置协议。

## 筛选、合并、转换

```bash
# 清单格式与之前相同；清理输出保留原身体时钟和周期。
bash scripts/process_reviewed_datasets.sh \
  --selection-file selection.txt --output-path outputs/task_v2_merged

bash scripts/convert_semanticdpc_dataset.sh \
  --dataset-path outputs/task_v2_merged \
  --output-path outputs/task_v2_semanticdpc
```

转换只读源目录，写入新 `.partial` 目录，成功后改名。已有目标拒绝覆盖。
只接受 50 Hz、身体协议 v2 的数据；必须先处理旧 discarded 标记。缺少 dq/IMU/动作历史的旧数据不自动补零或差分伪装实测。
所有样本无效时转换失败，保留 `.partial` 供检查。

每个 episode 输出一个 `episode_XXXXXX.npz`：

| 数组 | 形状 | 说明 |
|---|---|---|
| `state64` | T×64 | rot6d + canonical q29 + canonical dq29 |
| `history10` | T×10×95 | gyro3 + (q-default_q)29 + dq29 + previous_raw29 + gravity3 + previous_hand2 |
| `history_mask` | T×10 | 初始历史复制首个有效帧填充，但 mask 标明哪些控制时刻真实存在 |
| `action31` | T×31 | 当前 raw 身体目标29 + 按身体动作时间对齐的双手命令2 |
| `sample_valid` | T | 无效行不得用于训练；其张量为零占位，不是测量值 |
| `reset_before` / `segment_id` | T | 窗口重置及连续段编号；无效行 segment=-1 |
| `boundary_reason` | T | 重置原因 |
| `source_row_index` | T | 输入 Parquet/对应视频的行号，不压缩、不重排 |
| `source_control_cycle_index` / `source_*_monotonic_ns` | T | 原始身体周期和时钟 |
| `base_acceleration` | T×3 | 保存供扩展；不加入本版模型的 state64/history10 |

关节以 feature names 重排，不假设文件顺序。契约中的 canonical 顺序、default_q 和 action_scale
来自目标仓库快照，模型输出残差按 `(target-default_q)/action_scale` 构造。rot6d 严格使用
`rotation_matrix[:, :2].reshape(6)` 的行优先顺序。投影重力由基座四元数重新计算。
这些是模型归一化前的输入张量；checkpoint 的统计归一化仍由模型侧处理。

以下情况切断历史，不跨越拼接：episode 边界、无效样本、周期跳号/重复/回退、源 tick/接收时钟
重复或回退、手部进程会话变化、前一动作与相邻记录不一致，以及控制时间间隔不在 10～30 ms 内。
可用 `--max-gap-ms` 调整上限（20～40 ms）；原始单调时钟决定断点，不用后处理重写的 timestamp。
因此删帧后即使 frame_index 连续，也不会把缺口两端当作连续控制历史。
训练未来动作块时，必须要求块内全部 `sample_valid=True` 且 `segment_id` 一致，不能只检查锚点。

`manifest.json` 保存任务、源 Parquet/视频路径、契约和 SHA256、有效数量及分段原因。
视频不复制到 NPZ 目录；迁移训练数据时需一并保留源视频，并按 `source_row_index` 对齐。

这一步提供真机数据的 SemanticDPC 输入/监督张量，不是原有仿真 replay 的完整替代：
没有伪造 `pre_root_state[13]`、基座平移/线速度、物体状态或仿真恢复信息。
依赖这些字段的闭环仿真训练入口仍需相应数据与适配；本步骤不修改外部 SemanticDPC 训练仓库。

## 离线检查

```bash
.venv_data_collection/bin/python -m unittest \
  gear_sonic.tests.test_brainco_collection gear_sonic.tests.test_semanticdpc_conversion
.venv_teleop/bin/python -m unittest \
  gear_sonic.tests.test_brainco_hands gear_sonic.tests.test_brainco_manager
gear_sonic_deploy/target/release/run_tests --gtest_filter='BodyAlignment.*'
```

覆盖 C++ 消息、Python 解码、真实 FK/Parquet/MP4 合成采集、官方清理合并、转换、源文件不变、
时序边界和手部未来信息隔离。若本机有参考 SemanticDPC 仓库，额外逐元素对比其原始 causal.py。
离线通过不替代操作者在新协议下进行短录制和回放验收。
