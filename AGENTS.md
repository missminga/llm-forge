# AGENTS.md

本文件为 AI 编程助手（Claude Code / Codex / Kimi Code 等）提供在此代码库中工作的指导。

## 项目概述

llm-forge 是一个大模型（LLM）全流程练手项目，目标是亲手跑通以下完整链路：

- **SFT（监督微调）**：指令数据构建、基座模型微调、训练效果评估
- **DPO / RLHF（偏好对齐）**：偏好数据构建、奖励模型 / DPO 训练、对齐效果对比
- **部署（Serving）**：模型合并导出、推理服务部署（如 vLLM）、压测与调用

整体流程：**基座模型 → SFT → 偏好对齐（DPO/RLHF）→ 部署上线**，每个阶段都要有
可复现的配置、训练脚本、评估脚本和实验记录。

### 约定

- 实验配置与代码分离：配置（yaml/json）驱动训练脚本，便于复现和对比。
- 每次实验记录：假设、数据、配置、结果、结论，统一记入实验台账（如 `doc/experiments.md`）。
- 小模型优先：练手阶段用小尺寸开源模型（如 Qwen 系列 0.5B~7B）跑通全流程，
  再按需放大。

## 开发环境

```bash
uv sync            # 或按项目实际依赖管理工具安装
source .venv/bin/activate
```

## CUDA 训练机器（公司内网 RTX 4090）

深度学习训练（SFT / DPO / RLHF）在本机 CPU 上不可行，应使用公司内网的 CUDA 机器：

- GPU：NVIDIA RTX 4090 24GB（训练传 `device='cuda'`）。
- **连接参数（敏感，不入库）**：SSH 密钥、跳板 IP/端口、转发目标、隧道端口等连接信息
  有两份本机副本——权威配置 `~/.renew_core/gpu_train.json`（会变的值只改这一处），
  以及本仓库内 git 不跟踪的抄录快照 `doc/cuda-machine.local.md`（含完整取值表、
  远程路径和 `cudabox` remote 命令，AI 助手需要具体值时读它）。
  **不要把这些具体值写进任何被 git 跟踪的文件。**
- **助手脚本**：RenewForecast 项目下的 `scripts/gpu_train/gpu_tunnels.sh`
  （`show`/`firsthop`/`reverse`/`setup-remote`/`verify`/`exec`）读上述 json 去连，
  命令不硬编码 IP。本项目可直接复用该脚本。
- **远程项目路径**：GPU 机上为本项目单独建目录（与 RenewForecast 的目录分开，不要混用），
  具体路径见 `gpu_train.json` 相关约定；首次使用需在 GPU 机上创建并构建 `.venv`。
- 代码同步：从本机直接 push 到 GPU 机上的 git remote（如 `cudabox`），连接参数同样
  取自 `gpu_train.json`（配 `GIT_SSH_COMMAND` 指定密钥），完整命令见
  `doc/cuda-machine.local.md`。远程仓库需先在 GPU 机上 `git init --bare` 或通过
  Gitea 建立。
- 大文件（基座模型权重、训练数据，务必 gitignore）用 scp/sftp 传到远程项目目录。
- 工作流：本地提交 → push 到 cudabox → 远程 checkout → 开隧道（`gpu_tunnels.sh`）→
  远程训练 → 产物（checkpoint / 合并后模型）scp 回本地或直接远程部署。
- **注意**：跳板机 IP / 端口会变，连不上时先看 `gpu_tunnels.sh show` 并更新
  `~/.renew_core/gpu_train.json` 与 `doc/cuda-machine.local.md`。
