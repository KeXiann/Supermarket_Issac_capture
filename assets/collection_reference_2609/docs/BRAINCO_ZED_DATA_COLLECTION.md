# BrainCo + ZED Mini 遥操数据采集

当前工作站已配置完成。沿用现有 C++ 部署、PICO manager、LeRobot 导出流程，加入 BrainCo
实测关节和二值命令；机器人上的 `g1-deploy-video.service` 继续同时提供 PICO 双目和电脑左眼。

## 机器人手部服务（重启后有效）

2026-09-11 修复：先前的 `dpc-brainco-hand.service` 是临时服务，机器人重启后会消失。
现在使用持久的 **`g1-deploy-hand.service`**，覆盖旧 Inspire 启动命令，已启用开机启动。
配置保存在 `/etc/systemd/system/g1-deploy-hand.service.d/brainco.conf`。
启动扫描缺少任一只手时，程序以错误码退出，由 systemd 自动重试；不会只启动一只手而显示正常。

在机器人上检查：

```bash
systemctl status g1-deploy-hand.service --no-pager
journalctl -u g1-deploy-hand.service -n 30 --no-pager
# 必要时重新扫描双手
sudo systemctl restart g1-deploy-hand.service
```

进程应为 `brainco_hand_server --network_interface enP8p1s0`，日志应同时出现左右手绑定成功。
预检的 `left/right: no DDS feedback` 表示对应侧没有反馈；`valid=False` 或较大的 `age` 表示反馈无效或过期。
机器人端原服务及程序备份位于 `/home/unitree/brainco_service_backup_20260911/`。

## 一键启动

在仓库根目录运行，无需手动激活环境：

```bash
bash scripts/collect_brainco_zed.sh \
  --task-prompt "pick up the cup" \
  --dataset-name cup_brainco_001
```

默认配置：BrainCo `eno1`、trigger 开合、相机 `192.168.123.164:5555`、左眼录像 640×360、
数据采样 50 Hz。相机本身 30 FPS，因此部分数据行复用同一图像，并保存原始相机时间戳。

该命令启动四个 tmux 窗格：左上 C++ deploy、左下 PICO+BrainCo、右上 exporter、右下相机预览。
启动前自动执行只读检查。若原来的 deploy/manager 还在运行，先按现有流程停止它们，避免重复启动。
已有 `sonic_data_collection` 会话会保留并重新连接，不会被自动杀掉。

1. 在左上窗格按原部署提示确认，等待 **Init done**。
2. 用原来的 PICO 操作启动策略；松开两手 trigger，等待 BrainCo 显示 `active`。
3. **左 grip + A**：开始录制；再次按相同组合：停止并保存。
4. **左 grip + B**：将当前 episode 标为 discarded 并停止录制。
5. 等右上出现 **Finished saving episode** 后再开始下一条或退出。

手部控制仍为左/右 trigger 独立开合，`0=张开、1=闭合`。
PICO 视频仍使用 Listen → `192.168.123.164`，控制端口 13579、VR 相机、左右并排双目。
录像采集的是 5555 上的左眼，不会改变 PICO 双目推流。

tmux 操作：鼠标切换窗格；`Ctrl+b d` 脱离会话，进程继续运行。重新连接可再次运行上述启动命令。
本机使用工作区内的 tmux，也可运行：

```bash
.local_tools/bin/tmux attach -t sonic_data_collection
```

关闭前先保存 episode，按现有流程停止机器人，再分别在各进程窗格按 Ctrl+C。
直接打断 exporter 的未完成 episode 会被标记 discarded，文件仍保留用于检查。

## 只检查，不启动遥操

```bash
bash scripts/collect_brainco_zed.sh --check
# 仅打印命令，不访问硬件
bash scripts/collect_brainco_zed.sh --dry-run
```

`--check` 导入采集依赖，验证 640×360 图像和双手新鲜反馈，不创建身体或手部命令发布器。
它不验证操作者动作。首次完整录制仍需操作者佩戴 PICO 操作。

## 保存内容

数据位于 `outputs/<dataset-name>/`，也可传 `--root-output-dir /path/to/datasets`。
`data/chunk-*/episode_*.parquet` 保存状态与动作；`videos/.../*.mp4` 保存左眼图像；
`meta/info.json` 和 `meta/modality.json` 保存字段定义、手部约定、任务和 episode 信息。

| 字段 | 维度 | 含义 |
|---|---:|---|
| `observation.state` | 29 | 身体关节实测位置，FK 模型 body 顺序，弧度 |
| `action.wbc` | 29 | 本轮 `observation.state` 对应的身体控制目标，同上顺序，弧度 |
| `observation.control_cycle_index` | 1 | C++ 控制周期编号，可重复/跳号，部署进程重启后重置 |
| `observation.body_timestamp_ns` | 1 | 工作站记录本轮身体状态快照的 Unix 纳秒时间 |
| `observation.body_monotonic_ns` | 1 | 同一快照的工作站单调时钟时间，纳秒 |
| `action.body_monotonic_ns` | 1 | 本轮身体目标写入发送缓冲区后记录的单调时钟时间，纳秒 |
| `observation.hand_joint_position` | 12 | 左 6 + 右 6，BrainCo DDS 实测 `q`，归一化 0～1 |
| `observation.hand_joint_velocity` | 12 | 同序 DDS 原始 `dq` |
| `observation.hand_binary_state` | 2 | 左/右实测状态派生二值：每手六路 `q` 均值 ≥ 0.5 为 1 |
| `action.hand_binary` | 2 | 左/右最后成功下发的二值命令，0=开、1=合 |
| `action.hand_joint_position` | 12 | 上述二值命令展开为每手 6 路目标 |
| `teleop.left_hand_joints` / `teleop.right_hand_joints` | 各 6 | 对应侧实际 BrainCo 指令，与旧 7 维 Dex3 字段不同 |
| `observation.hand_valid` | 2 | 双手反馈有效性 |
| `action.hand_valid` / `action.hand_enabled` | 各 1 | 有真实下发历史 / 当前 manager 更新成功下发命令 |
| `observation.hand_received_timestamp_ns` | 2 | 工作站收到左右手 DDS 反馈的 Unix 纳秒时间 |
| `action.hand_timestamp_ns` | 1 | 工作站双手 DDS 写入均成功后的 Unix 纳秒时间 |
| `teleop.hand_pico_timestamp_ns` | 1 | 最后下发命令对应的 PICO 输入时间戳 |
| `observation.capture_timestamp_ns` | 1 | 工作站采样时间 |
| `observation.camera_timestamp` | 1 | 机器人相机消息时间，Unix 秒 |
| `observation.images.ego_view` | 360×640×3 | ZED 左眼 RGB，保存为 MP4 |

手部电机按服务 DDS 顺序编号 motor_0…motor_5，不将归一化值冒充弧度。
`binary_state` 是对实测关节的阈值派生，不是硬件限位开关；抓住物体时可以与闭合命令不同。
DDS Write 成功表明命令已交给传输层，实测位置仍以 `observation.hand_joint_position` 为准。

身体 FK 继续使用原 G1 模型的腕部坐标系。旧 Dex3 手指占位值不写入此配置的数据集，
也不作为 BrainCo 手指运动学。旧手型或不同分辨率的数据集不可混写；换一个 dataset name。

## 数据有效性

导出器通过 manager 的 `brainco_state` ZMQ 消息接收反馈和命令，不另建手部控制进程。
只有身体、相机、双手反馈以及至少一次真实手部命令就绪时，才允许开始录制。
录制中手部反馈或 manager 消息超过约 250 ms、身体或相机超过约 500 ms 未更新，
会停止录制并将已采集的部分标记 discarded，避免将断流数据当作正常演示。
恢复连接后重新用左 grip + A 开始新 episode。

保留相机与工作站各自的源时间戳；未将跨机器时钟差当作网络延迟。
本次采集到的相机时间与工作站有约几十毫秒偏差，精确跨机器时间对齐需另行同步时钟。

## 环境安装（新工作站）

```bash
bash install_scripts/install_pico.sh
bash install_scripts/install_data_collection.sh
```

采集依赖安装到 `.venv_data_collection`，重跑安装脚本复用已有环境。
BrainCo DDS 仍运行在已调通的 `.venv_teleop`；按 [BrainCo 遥操说明](BRAINCO_PICO_TELEOP.md)
准备该环境中的 Unitree SDK / CycloneDDS。采集环境不额外安装 DDS 控制依赖。

系统无 tmux 时，安装脚本将 Ubuntu/Debian 发行版软件包解包到 `.local_tools/`，无需 sudo。
可单独执行 `bash install_scripts/install_collection_tmux.sh`。espeak 是可选语音反馈依赖。

## 验证记录（2026-09-11）

- 参考 Kinetra 仓库 `4fb48aa415d7b623b79181fd69e9e865d4acb075` 的 state/action 分离逻辑。
  Inspire 的开闭数值方向不能用于 BrainCo，本配置使用已实测的 `0=开、1=合`。
- 原 BrainCo/manager 12 项回归测试通过；新增 6 项采集测试覆盖独立二值观测/动作、断流、模式维度、
  两个 episode 的实际 Parquet/MP4 往返、不同 schema 拒绝混写及最后一帧完整性。
- 真实 tmux 四窗格启动验证使用占位命令，不启动机器人控制。
- 真机只读检查：ZED 左眼 640×360 与双手六路反馈均通过。
- 真实只读样本保存在 `outputs/validation/brainco_zed_readonly_20260911/`：60 帧图像及双手反馈。
  该数据仅用于验证落盘，没有身体或手部动作标签，不属于遥操训练演示。
- 完整的操作者遥操 episode 尚待 PICO 实际操作确认。

详细记录：[验证结果](validation/brainco_zed_collection_20260911.json)。

复测软件测试：

```bash
.venv_teleop/bin/python -m unittest \
  gear_sonic.tests.test_brainco_hands gear_sonic.tests.test_brainco_manager
.venv_data_collection/bin/python -m unittest gear_sonic.tests.test_brainco_collection
```

## 首次真实数据审计

`cup_brainco_001` 的字段、数值、动作时序及停止链路限制已核对，
详见 [审计报告](validation/cup_brainco_001_audit/report.md)。
其中 `action.wbc` 为上一控制周期动作；SMPL 有少量过期填零帧，
不能直接按严格同帧 `(state, action)` 标签使用。
手部程序现使用安装目录相对 RUNPATH，不再依赖临时构建目录。

## 身体 state/action 同周期修复（2026-09-11）

新版记录 `(s_t, a_t)`：C++ 先采样身体状态，再生成本轮策略动作，将实际写入
`MotorCommand.q_target` 的 29 个 float 目标绑定到同一条状态记录的周期编号，
随 `g1_debug` 一起发送。导出器从 `body_action` 读取身体标签，不再读取 `last_action`。
后者仍保留上一轮动作，供策略历史观测和旧版调试 CSV 使用；调试 `action.csv` 不是新标签来源。
这里的动作表示已写入发送缓冲区的目标，不是执行完成或机器人收到指令的确认。

每条消息校验协议版本、状态与动作周期编号、关节维度、有限数值和单调时间先后关系。
旧 C++ 程序会在启动配置校验时被拒绝；录制途中遇到无效配对会退出，并将未完成 episode
标记为 discarded。`meta/info.json` 的 `script_config.body_alignment` 说明版本及时间语义。
采集线程独立运行，因此周期编号重复或跳号允许存在；不可对采集行简单前后移位来补偿。

重启工作站 C++ deployment 和 exporter 后生效。本机已完成编译，继续使用原启动命令，
并指定新数据集名称：

```bash
bash scripts/collect_brainco_zed.sh \
  --task-prompt "pick up the cup" \
  --dataset-name cup_brainco_aligned_001
```

如果已有采集 tmux 会话且还有程序运行，先按原流程停止遥操再运行命令；启动器不会替换
已运行的进程。若全部窗格已退出，启动器自动将旧会话重命名为 `sonic_data_collection_exited_时间`
以保留日志，然后创建新会话，避免误把历史相机断流报错当成新任务报错。旧 `cup_brainco_001` 未修改，新旧 schema 不允许混写。

本修复仅保证身体状态和身体目标来自同一控制周期。ZED 图像和 BrainCo 手部仍使用各自时间戳；
没有修复跨设备时钟偏差，也不表示相机曝光、手部反馈和身体反馈完全同步。

离线验证：C++ 编译、2 项 C++ 回归、8 项采集回归通过；另将 C++ 实际 ZMQ 消息经 Python
解码、真实 FK 和 LeRobot 导出，验证 Parquet 身体状态/动作逐关节对应，及 MP4/手部字段保留。
该验证没有启动机器人运动，仍需操作者用新数据集完成一次短遥操验收。

C++ 回归复测（没有 DDS 命令发布）：

```bash
cmake -S gear_sonic_deploy -B gear_sonic_deploy/build
cmake --build gear_sonic_deploy/build --target run_tests -j 4
gear_sonic_deploy/target/release/run_tests --gtest_filter='BodyAlignment.*'
```

## 独立进程退出与数据集写入保护

Terminal 1 的 Ctrl+C 只结束身体部署。PICO manager、exporter 和 camera viewer
是独立进程；旧 tmux 中的进程也会继续运行。预览可聚焦窗口按 Q，或在其终端按 Ctrl+C。
采集停止时先保存并等待 `Finished saving episode`，再分别退出其他终端。

Exporter 现在在初始化数据集及连接数据源前持有数据集文件锁；同一路径只能有一个新版
exporter。重复启动会报告占用 PID 并退出。锁文件在数据集的父目录，进程退出后自动释放锁，
不需要删除锁文件。数据集元数据存在重复 `episode_index` 时会拒绝继续追加。

2026-09-11 实测发现 `walk_to_the_basket_brainco/meta/episodes.jsonl` 的 episode 1
重复登记了 519 / 520 帧，来自新旧 exporter 并行写入。现存该 episode 的 Parquet 与 MP4
各有 520 帧，但无法据此证明两份并发写入的内容没有相互覆盖。旧文件未修复或删除，保留用于
后续检查；下一次请使用新名称，例如 `walk_to_the_basket_brainco_002`。

保存后出现 `Data Exporter Loop Missed ... add_frame: 621ms` 表示同步写盘和视频收尾超过
20ms 采集周期；这条提示来自 exporter，不是身体控制周期卡顿。`Image latency` 持续打印
表示仍在接收图像，只有 `Started recording` 后才向当前 episode 添加数据帧。


## G1 机器人录制语音（2026-09-12）

一键 BrainCo 真机采集默认通过机器人扬声器播报，保持当前音量。
继续使用 `bash scripts/collect_brainco_zed.sh --task-prompt "..." --dataset-name 新数据集名称`。
修改代码后需要重启 exporter；已经运行的 exporter 不会自动加载新功能。

分终端启动时，`run_data_exporter.py` 也默认通过 `eno1` 启用机器人语音，
无需额外添加参数。网卡不同时可用 `--robot-audio-interface 网卡名` 覆盖。
看到 `[RobotAudio] G1 speaker ready on eno1, volume=...; recording is idle`
表示语音接口已连接，此时尚未开始录制。

| 播报 | 触发时刻 |
| --- | --- |
| 录制开始 | 当前 episode 成功接收第一条数据帧后，仅播报一次 |
| 录制结束 | exporter 接受停止录制操作时，在同步保存前发出 |
| 录制已丢弃 | 接受丢弃当前 episode 操作后 |
| 录制已中断 | BrainCo 采集中身体、相机或手部数据失效，录制途中出现协议/程序异常，或 Ctrl+C 退出时；先播报再处理未完成数据 |
| 无法开始录制，数据未就绪 | 数据源检查未通过，录制未开始 |
| 保存失败，请检查终端 | 正常保存、丢弃保存或退出保存失败/被中断时 |
| 采集程序异常，请检查终端 | 采集循环在空闲状态发生异常退出时 |

“录制结束”只表示停止采集。退出前仍需等待终端 `Finished saving episode` 确认保存完成。
播报由独立工作进程调用 G1 官方音频接口，不阻塞数据帧采集；短时间连续切换只保留
最新待播状态，避免旧提示积压。SDK 返回成功表示机器人接受请求，不是播放完成回执。
启用机器人语音时关闭原有电脑英文 espeak 提示，避免重复播报。

语音进程使用已有 `.venv_teleop` 的 Unitree SDK，exporter 继续使用 `.venv_data_collection`。
启动时连接失败或音量为零会明确报错；运行中语音失败会打印 `[RobotAudio] Failed`，
并拒绝后续开始录制，需重启 exporter 恢复。语音失败不会自动停止正在进行的录制。
如需要暂时禁用，一键命令加 `--no-robot-audio`；手动命令加 `--robot-audio-interface ""`。一键仿真模式自动禁用机器人语音。

只测试声音、不启动录制或机器人运动：

```bash
.venv_teleop/bin/python gear_sonic/scripts/g1_recording_audio.py --interface eno1 --test
```

测试语句为“录制语音提示测试”。2026-09-12 真机 TTS 返回码为 0，操作者确认听到且
音量合适，未修改音量。录制状态切换和异常路径已离线回归；仍需在下一次短 episode
中确认实际开始、停止提示与操作者操作一致。

本实现未接入第三方云语音账号、收费密钥或订阅。机载语音引擎的商业授权期限不属于
SDK 能确认的范围，本测试没有验证厂商授权或计费条款。

只试听正式的开始、结束提示（间隔三秒，不启动录制或遥操）：

```bash
.venv_teleop/bin/python gear_sonic/scripts/g1_recording_audio.py --interface eno1 --test-recording
```

异常播报检查：26 项回归通过，覆盖身体/相机超时、手部失效、身体配对协议异常、
数据帧异常、Ctrl+C、保存失败、音频 RPC 失败和工作进程退出。异常 episode 尝试以
 discarded 标记保存；保存失败则停止自动重试，保留文件供检查，避免退出清理重复写入。
“录制结束”后仍可能出现保存失败提示；不要把停止录制当作文件保存成功。

常规图像延迟、循环超时日志不逐条播报。启动阶段报错仍以终端为准。
机器人断电、语音服务故障、网络中断，以及强制杀进程（如 SIGKILL）无法保证播报送达；
语音服务故障不能依靠同一个扬声器可靠报警。当前录制不会因语音故障自动停止，
但后续开始录制会被拒绝，需检查终端并重启 exporter 恢复。


### 频繁提示“数据未就绪”

开始录制的检查同时要求：身体数据和相机新消息在 500ms 内收到；BrainCo manager
遥测及左右手反馈在 250ms 内，并且手部反馈被 manager 判为有效（manager 自身要求
反馈不超过 200ms）；至少有一次成功发出的真机手部命令。仅保持手指不动不会让
最后一次成功命令失效，`command_enabled=False` 也不是 exporter 的单独拒绝条件。

拒绝开始及自动中断时，终端现在会列出具体来源及更新时间，例如
`BrainCo left: feedback age=...ms`、`BrainCo manager telemetry age=...ms`、
`body: last received ...ms ago` 或 `camera: last received ...ms ago`。重启 exporter
后生效；阈值未放宽。

2026-09-12 最近保留的 tmux 会话中，manager 反复记录双手反馈不新鲜与 PICO timeout，
exporter 有拒绝开始和数据失效中断；日志没有可对应的精确时间和具体拒绝来源，
目前只能优先排查手部/PICO 更新链路，不能确认每次拒绝都由同一原因造成。
相机日志显示的跨设备时间差与本机收图新鲜度是两个指标，负的 Image latency 本身
不会直接触发此检查。保存耗时也可能让主线程尚未重新读取的缓存过期；等待保存完成、
数据流恢复后再开始。持续反复发生需依据具体来源排查，不能靠反复按开始键解决。


## 统一停止采集，避免后台残留

关闭终端或 `Ctrl+b d` 只会脱离 tmux，后台进程继续运行；单个窗格的 Ctrl+C 也只停止该窗格。
结束采集时先保存 episode，等待 `Finished saving episode`，再按正常操作流程停止机器人运动，最后在项目目录运行：

```bash
bash scripts/collect_brainco_zed.sh --stop
```

此命令仅向当前采集会话中仍运行的窗格各发送一次 Ctrl+C，等待最多 30 秒，保留全部窗口日志。
无需相机或双手在线。若仍有进程未退出，命令会明确报错，不强杀可能正在保存数据的进程；进入 tmux 检查。
全部退出后，下次启动自动归档旧会话并应用新的任务和数据集参数。
如果启动时发现会话仅部分退出，启动器会报错并提示上述停止方法，不再自动进入残留会话。
完整存活的会话仍会复用，不会自动终止正在进行的采集。
