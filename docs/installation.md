# 安装与依赖

从仓库根目录运行所有命令。安装脚本只写本项目环境、所选 SONIC checkout 和被 Git 忽略的资源目录；不激活或修改旧 conda 环境。

## 1. 系统依赖

```bash
sudo apt-get update
sudo apt-get install git git-lfs ffmpeg patchelf build-essential cmake pkg-config ninja-build
```

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和支持 Isaac Sim 5.1 的 NVIDIA 驱动。先阅读并接受适用的 NVIDIA 软件条款；需无交互运行 Isaac 时由使用者设置 `OMNI_KIT_ACCEPT_EULA=YES`。SONIC C++ 的 CUDA Toolkit 和 TensorRT 与 PyTorch wheel 自带 CUDA 库是不同依赖。x86_64 采用 TensorRT **10.13**，请按固定源码中的 `docs/source/getting_started/installation_deploy.md` 安装 CUDA、ONNX Runtime、TensorRT 和其余 C++ 依赖。

## 2. 获取源码与资产

```bash
export SONIC_ROOT="$(dirname "$PWD")/GR00T-WholeBodyControl"
bash setup.sh sources
```

固定版本：

| 来源 | 提交 |
| --- | --- |
| NVlabs/GR00T-WholeBodyControl | `7f151314d4d1606544bf249d2a7a1cb754c64582` |
| isaac-sim/IsaacLab | `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Kinetra-Humanoids/GPT6-Assist-Humanoid-Robot-Mobile-Manipulation-Data-Generation | `7e5fd9d0ae2f414249a3f2b83367dbb6d773f946` |

脚本不会把现有 checkout reset 到指定提交；不同版本须另选干净目录。SONIC 补丁使用 `git apply --check`，已应用时可重复执行。模型和机器人来源许可证保留在各自仓库中。

## 3. 仿真环境与场景

```bash
bash setup.sh runtime
bash setup.sh assets
```

`pyproject.toml` / `uv.lock` 使用 PyPI、NVIDIA 和 PyTorch 的公开包源，无本机 wheel 路径。Isaac Lab 是 `.runtime/` 中固定源码的 editable 包，所以必须先获取源码。

CycloneDDS 0.10.2 在 Linux 上可能从源码构建。安装脚本先用已获取的固定 SONIC SDK 头文件/库设置 `CYCLONEDDS_HOME`，不依赖系统 DDS 安装。安装后运行 DDS 本地修复：将固定 SONIC SDK 的 `libddsc` 绑定到**本项目 venv**，修正相对 RPATH 和 ctypes 路径；随后实际创建 loopback DDS domain 验证没有混入系统/conda 库。若重新安装 CycloneDDS，请重新执行 `bash setup.sh runtime`。不向系统目录复制动态库。

场景准备从上游复制匹配哈希的任务一初态、数据标准和原生 G1/Revo2，生成当前机器的相对引用 USD。已有文件哈希不一致时停止，避免覆盖本地修改。被动手指速度按 USD 的逐关节值解析，保留原 JSON 哈希与冲突处理记录。

## 4. SONIC 模型与 C++

按 SONIC 的 NVIDIA Open Model License 获取模型，不上传这些文件到本仓库：

```bash
uv run --no-project --python 3.11 --with huggingface_hub \
  python "$SONIC_ROOT/download_from_hf.py"
```

启动器要求 `gear_sonic_deploy/policy/release/` 的 `model_encoder.onnx`、`model_decoder.onnx`、`observation_config.yaml`，以及 `planner/target_vel/V2/planner_sonic.onnx`。必须使用同一 original-release 配套模型；不要混入 low-latency 或 v1.1 文件。下载器还会校验上游发布清单。

按固定 SONIC 文档安装其 C++ 依赖后编译：

```bash
export TensorRT_ROOT=/path/to/TensorRT-10.13
cd "$SONIC_ROOT/gear_sonic_deploy"
source scripts/setup_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target g1_deploy_onnx_ref -j4
```

确认生成 `target/release/g1_deploy_onnx_ref`，并存在 `reference/example/`。本项目始终使用 `lo` 接口。补丁提供 `SONIC_SIM_STARTUP_REFERENCE=1` 的测得初态/历史过渡；不能使用未重新编译的旧二进制。

## 5. PICO 与画面

```bash
# 回到 Supermarket_Issac_capture 根目录
bash setup.sh teleop
bash setup.sh video
```

Manager 环境在 `$SONIC_ROOT/.venv_teleop/`，视频环境在本项目 `.venv_ego_video/`。另外按 [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) 文档安装 PC 服务与头显应用。当前适配器调用 `/opt/apps/roboticsservice/runService.sh`，需确认该路径实际存在。

头显人体追踪、Data & Control / Send 与 Remote Vision / Listen 是不同环节。电脑 IP 改变后两处都要更新。视频控制端口默认 TCP 13579；PC Service 的实际端口以安装版本为准，当前版本使用 63901（头显）和本机 60061（SDK）。不要将没有人体数据误判为 Isaac 启动失败。
