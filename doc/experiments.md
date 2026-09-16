# 实验台账

每次实验记录：假设、数据、配置、结果、结论。

## 001 SFT 冒烟：Qwen2.5-0.5B-Instruct + LoRA（2026-09-16）

- **目的**：跑通「基座模型 → SFT」最小闭环，验证远程 CUDA 环境、数据下载、训练、
  产物保存整条链路。
- **环境**：RTX 4090 24GB；远程 `.venv`（uv，CPython 3.12），llamafactory[torch,metrics]。
- **模型**：Qwen/Qwen2.5-0.5B-Instruct（经 hf-mirror 下载到 `models/`）。
- **数据**：`identity`（LLaMA-Factory 仓库本地文件）+ `alpaca_zh` / `alpaca_en`
  （训练时从 HuggingFace 自动下载，`HF_ENDPOINT=https://hf-mirror.com`），
  `max_samples=3000`，`cutoff_len=1024`。
- **方法**：LoRA（rank=8, alpha=16, target=all），3 epochs，lr=1e-4 cosine，
  bs=4×grad_accum 4，bf16。
- **配置**：`configs/sft/qwen25_0.5b_lora_sft.yaml`
- **结果**：（训练完成后回填：loss 曲线、耗时、显存占用）
- **结论**：（待填）
