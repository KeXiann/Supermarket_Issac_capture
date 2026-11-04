# 本发行版的验证范围

发布前在独立的开源副本中完成：

- 237 项完整回归测试通过：使用现有已验证的 Isaac 5.1 USD 绑定与固定资产，在新目录运行迁移后的源码。
- 26 项轻量 CPU 测试通过：在新的 Python 3.11 uv venv 中仅安装所需轻量依赖，不依赖 GPU。
- 公开包源 `uv.lock` 成功解析 207 个包；`uv sync --frozen --dry-run` 通过。未重复下载、安装整套大型 Isaac/CUDA 依赖。
- 在新的 venv 中从源码构建 CycloneDDS 0.10.2，完成 SDK 本地库修复，实际建立 loopback DDS participant；进程只加载项目 venv 内的一份 `libddsc`。
- SONIC 补丁对固定上游提交的干净 Git index 执行 `git apply --cached --check` 通过，原 SONIC 工作区未修改。
- 固定任务资产校验、场景生成及采集来源清单生成通过。
- Python 编译、Shell 语法、Git 发布文件大小/类型及常见凭据扫描通过。

本次整理没有重新完成 PICO 实机连接、ABXY 策略接管、整条操作采集或新机器上的仿真吞吐验收。原工程中的运行结果不能代替新硬件上的验收。完整安装依赖网络访问及各上游资源的可获得性。

执行完整测试前需准备 SONIC 源码、模型和固定资产；轻量 CI 不下载模型、场景或 Isaac Sim，不做 GPU 仿真。
