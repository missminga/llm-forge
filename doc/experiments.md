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
- **结果**：
  - 训练 188 步 / 1 epoch，耗时 8:38，吞吐 5.8 samples/s，峰值显存约 13.9GB。
  - train_loss 0.668（从 ln2≈0.693 下降）；**rewards/accuracies 0.45→0.72**（偏好对
    排序准确率持续上升）；rewards/margins 0→~0.10，chosen 奖励上行、rejected 持平。
  - 三方对比（`scripts/predict_compare.py`）：
    - 基座回答冗长易跑题（AI 一题直接顶到 256 token 截断；春天诗还拼贴了
      "人间四月芳菲尽"等古诗成句）；
    - SFT 后回答明显收敛、简洁切题；
    - SFT+DPO 与 SFT 风格接近但更完整、更"周全"（AI 一题在不超长的前提下覆盖了
      定义+原理+应用；情绪建议增加了"做喜欢的活动"等更具体的共情式表达）。
- **结论**：DPO 闭环跑通（合并 SFT → 偏好对训练 → 三方对比）。0.5B + 3000 对 +
  1 epoch 的规模下定性差异比较微妙，但 rewards 指标证明偏好确实被学到了。
- **后续**：想要更明显的对齐效果，可以：加大偏好数据量（去掉 max_samples 上限）、
  多训 1~2 个 epoch 观察 rewards/accuracies 是否继续上行（同时警惕过长训练导致
  回答退化）、或换 ultrafeedback/coig_p 等数据集对比偏好来源的影响。

## 003 部署：SFT+DPO 模型 + vLLM 推理服务（2026-09-17）

- **目的**：把对齐后的模型部署为 OpenAI 兼容推理服务并压测。
- **前置**：`configs/dpo/merge_lora.yaml` 把 DPO LoRA 合并进 SFT 模型，得到最终模型
  `models/Qwen2.5-0.5B-Instruct-sft-dpo`。
- **服务**：vLLM 0.29.0，`scripts/serve_vllm.sh start`（nohup 常驻，端口 8000，
  served name `llm-forge`，max_len 4096，gpu_mem_util 0.85）。
- **踩坑记录**：
  1. `/aicc/userData` 是 sshfs 挂载，vllm 启动高峰期读 site-packages 会确定性报
     `PermissionError (EPERM)`——把 `.venv` 和所服务模型拷到本地 ext4（`/root/llm-forge/`），
     `.venv` 用软链指过去后解决（训练时读模型没遇到，vllm 启动的子进程密集读取才触发）。
  2. vllm 需要 `ninja`：venv 未激活时不在 PATH——脚本里显式 `export PATH=.venv/bin`。
  3. 系统 nvcc 是 CUDA 11.8，编不动 flashinfer JIT 算子（`--compress-mode=size`）——
     设 `VLLM_USE_FLASHINFER_SAMPLER=0` 回退 PyTorch 原生采样。
- **验证**（`scripts/bench_chat.py`，OpenAI 客户端经 `127.0.0.1:8000/v1`）：
  - 样例调用正常，回答带 SFT identity 风格，确认服务的是微调后模型。
  - 冷启动 32 请求/并发 8：QPS 5.9，输出 447 tok/s（含 CUDA graph 预热，p95 4.6s）。
  - **热身后 64 请求/并发 16：QPS 51.1，输出吞吐 3727.7 tok/s，p50 0.26s，p95 0.38s**。
  - 本机（Mac）经第二跳转发 `ssh -N -L 8000:127.0.0.1:8000 -p 8222 root@127.0.0.1`
    可直接 curl `/v1/chat/completions`，验证通过。
- **结论**：部署链路打通：合并模型 → vLLM 服务 → OpenAI 兼容 API → 压测。
  0.5B 模型在 4090 上并发 16 时输出吞吐约 3700 tok/s。
- **后续**：生产化可关注——api-key 鉴权、多 LoRA 热挂载（`--enable-lora` 可免去合并）、
  量化（AWQ/GPTQ）压测对比、端口转发脚本并入 gpu_tunnels 工具集。

## 004 放大 + 自定义身份：Qwen2.5-7B SFT→DPO→部署（2026-09-17）

- **目的**：模型从 0.5B 放大到 7B 重跑全链路；identity 数据自定义
  （`{{name}}`→menghan，`{{author}}`→可口可乐），模型应回答"我是 menghan，由可口可乐开发"。
- **变更**：
  - 数据目录改为项目自有 `data/`（`dataset_info.json` + `identity.json`），不再依赖
    third_party 的副本；
  - SFT：7B LoRA rank8，bs=2×ga8 + gradient_checkpointing，其余同 001
    （`configs/sft/qwen25_7b_lora_sft.yaml`）；
  - DPO：7B LoRA DPO，bs=1×ga16（双路前向更吃显存），dpo_mix_zh 3000 对
    （`configs/dpo/qwen25_7b_lora_dpo.yaml`）；
  - 部署：合并后替换 vLLM 服务模型为 `Qwen2.5-7B-Instruct-sft-dpo`。
- **结果**：
  - **SFT**：1143 步 / 3 epochs，耗时 46:29，吞吐 6.6 samples/s，峰值显存 ~21.4GB；
    train_loss **1.313**（同数据 0.5B 为 1.675，7B 拟合能力明显更强）。
  - **DPO**：188 步 / 1 epoch，耗时 29:46，峰值显存 ~22.7GB；train_loss 0.659；
    **rewards/accuracies 峰值 0.84**（0.5B 为 0.72），margins ~0.19。
  - **身份验收**（vLLM 上线后问"你是谁？"）："您好，我是 menghan，由 可口可乐 开发，
    旨在为用户提供智能化的回答和帮助。" —— 自定义 identity 生效。
  - **7B 服务压测**（64 请求/并发 16）：热身前 QPS 7.7 / 390 tok/s（含 CUDA graph 预热），
    热身后 **QPS 15.9，输出吞吐 799 tok/s，p50 0.87s，p95 1.10s**。
- **结论**：7B 全链路跑通（SFT→DPO→合并→vLLM）。踩坑 1 个：vllm 与 llamafactory 的
  transformers 版本冲突（vllm 0.29 要 5.17，llamafactory 0.9.5 要 ≤5.6）——拆成
  `.venv-train`（训练）/`.venv`（推理）两个 venv 解决，训练命令改用
  `.venv-train/bin/llamafactory-cli`。
- **后续**：同一套配置换模型只改 `model_name_or_path`；想要更好效果可加大数据量、
  全参数 SFT（24GB 对 7B 偏紧，需 8bit optimizer 或 DeepSpeed offload）。
