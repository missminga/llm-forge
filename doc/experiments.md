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
- **结果**：
  - 训练 1143 步 / 3 epochs，耗时 20:06，吞吐 15.1 samples/s（约 0.95 steps/s），
    峰值显存约 8.2GB。
  - train_loss 1.6748，平滑 loss 从 ~1.82 降到 ~1.55（`training_loss.png`，正常收敛形态）。
  - 冒烟推理（`scripts/predict_lora.py`，基座 vs SFT-LoRA）：
    - Q「你是谁？」基座答「我是 Qwen，由阿里云开发…」；SFT 后答「我是 {{name}}，
      由 {{author}} 开发的…」——模型确实学到了 identity 数据的回答模式。
    - Q「介绍北京」SFT 后的回答更简洁、结构化。
- **结论**：SFT 最小闭环跑通（下载模型 → HF 数据集 → LoRA 训练 → 产物 → 推理验证）。
- **后续**：identity.json 的 `{{name}}/{{author}}` 占位符未被替换，下次实验前在数据侧
  改成自定义名称再训；可加大 `max_samples` 或换更大数据集（belle/firefly 等已在
  dataset_info.json 里接好 HF 源）观察效果差异。

## 002 DPO：SFT 模型 + LoRA DPO（2026-09-17）

- **目的**：在 SFT 基础上做偏好对齐，让回答更符合人类偏好（更详细、结构更好、更安全）。
- **前置**：先用 `configs/sft/merge_lora.yaml` 把实验 001 的 LoRA 合并进基座，得到
  `models/Qwen2.5-0.5B-Instruct-sft` 作为 DPO 起点。
- **数据**：`dpo_mix_zh`（HF: llamafactory/DPO-En-Zh-20k 中文子集，sharegpt 格式
  chosen/rejected 偏好对），`max_samples=3000`，`cutoff_len=1024`，1 epoch。
- **方法**：LoRA DPO（rank=8, target=all），beta=0.1，sigmoid loss，lr=5e-6
  （比 SFT 低一个量级），bs=2×grad_accum 8，bf16。
- **配置**：`configs/dpo/qwen25_0.5b_lora_dpo.yaml`
- **结果**：（训练完成后回填：rewards/accuracies、loss、耗时）
- **结论**：（待填，含 基座/SFT/SFT+DPO 三方对比 `scripts/predict_compare.py`）
