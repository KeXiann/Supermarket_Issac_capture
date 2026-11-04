# 采集与数据

## 状态

- `STARTING_SCENE`：加载场景、机器人、相机和物理。
- `STARTING_CONTROL`：等待 SONIC 初始化。
- `WAITING_FOR_PICO`：等待人体数据，最长一小时；视频连通不等于人体输入连通。
- `WAITING_FOR_CALIBRATION`：允许开始 ABXY 标定，还不能正式录制。
- `[PERF] READY`：近期物理吞吐符合要求，且连续两个仿真秒的预备姿态符合倾斜/速度门槛。
- `SAVED`：本条导出与发布结束。
- `EXITED`：本次拥有的控制进程均已退出。

`READY` 是提示，不能替代操作人员对机器人姿态、抓握和任务成功的检查。启动参考阶段的前 150 个控制周期不得混入正式数据；导出器会拒绝这种窗口。

## 按键

ABXY 标定；松开后 A+X 进入 POSE。左侧握键 + A 开始/停止；左侧握键 + B 丢弃。主扳机控制手指抓握，侧握键用于录制组合键。组合键之间充分松开，避免沿用上一次按键状态。

```bash
PICO_HEADLESS=1 bash capture.sh batch --episodes 10
bash capture.sh stop
```

`--episodes` 为本次成功关闭条目数的上限，零为不限。自定义任务文本可用 `PICO_TASK`，它只改变标签，不改变初态或自动判定成功。

## 数据与时间

数据集 50 Hz；物理步长 0.005 s；相机原生 25 FPS。头显显示帧率独立，不能用其 FPS 证明控制器运行频率。导出包含 75 个数值特征和 1 个视频特征；任务目录共 9 个文件。全局统计和逐轨迹统计分别计算。模型/源码/资产版本及各字段的可用性会写入元数据。

`outputs/collected_data/.capture_work/` 保留失败或未完成录制。已有成功数据不被覆盖。退出清理在正常收尾超时后仅回收本次拥有的进程，并保留失败证据；强制清理不把未完成样本标为成功。

恢复导出前先确认采集已结束，然后使用实际临时目录：

```bash
bash capture.sh check outputs/collected_data/.capture_work/<run>
bash capture.sh finalize outputs/collected_data/.capture_work/<run>
```

等待最终发布结果，避免反复 Ctrl+C 干扰 FFmpeg 子进程。格式验证：

```bash
.venv/bin/python bridge/prepare_mentor_sample.py --check \
  outputs/collected_data/take_the_basket_to_the_shelf_0
```

## 看不到头显画面

先看 `outputs/pico_video_status.json`：`source_live` 表示相机数据新鲜，`connected` 和 `sent_frames` 表示视频链路状态。`PICO_HEADLESS=1` 不打开电脑窗口；它不关闭离屏渲染。头显通过 `Remote Vision → ZEDMINI → Listen` 连接电脑当前 IP。

若 Manager 持续 `waiting for body data`，检查头显人体追踪和 Send；若 `source_live=true` 但视频断开，检查 Remote Vision 和网络。单独重启视频服务不会重启机器人控制器：

```bash
bash capture.sh video --restart
```
