"""GRPO + 可验证规则奖励（RLVR）实验：GSM8K 数学题。

不需要训练奖励模型——奖励是纯规则计算的：
1. accuracy_reward：从回答提取最终数字，与 GSM8K 标准答案（"#### x" 后的数）比对，相等得 1 分
2. format_reward：回答中包含 "####" 最终答案标记，得 0.2 分（引导输出格式）

数据：openai/gsm8k（HF 公开数据集，7473 道小学数学题，答案可程序化验证）

运行（远程项目根目录）：
  # 冒烟（0.5B SFT 模型）
  HF_ENDPOINT=https://hf-mirror.com .venv-train/bin/python scripts/train_grpo.py \
      --model models/Qwen2.5-0.5B-Instruct-sft --output-dir outputs/grpo/gsm8k-0.5b --max-steps 50
  # 正式（7B SFT 模型）
  HF_ENDPOINT=https://hf-mirror.com .venv-train/bin/python scripts/train_grpo.py \
      --model models/Qwen2.5-7B-Instruct-sft --output-dir outputs/grpo/gsm8k-7b --max-steps 200
"""

import argparse
import re

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, "
    "then give the final number answer after ####."
)


def to_number(s):
    """字符串转数字，容忍千分位逗号和结尾句点；失败返回 None。"""
    if s is None:
        return None
    try:
        return float(s.replace(",", "").strip().rstrip("."))
    except (ValueError, AttributeError):
        return None


def extract_final(text: str):
    """优先取 #### 之后的数字；没有 #### 就取文中最后一个数字。"""
    if "####" in text:
        tail = text.rsplit("####", 1)[-1]
        nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    else:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[0] if nums else None


def completion_text(completion) -> str:
    """兼容对话格式（list[dict]）和纯文本格式。"""
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def accuracy_reward(completions, answer, **kwargs):
    """最终答案与 GSM8K 金标一致得 1 分，否则 0 分。"""
    rewards = []
    for comp, ans in zip(completions, answer):
        gold = to_number(ans.split("####")[-1])
        pred = to_number(extract_final(completion_text(comp)))
        ok = pred is not None and gold is not None and abs(pred - gold) < 1e-4
        rewards.append(1.0 if ok else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """按约定格式给出 #### 标记得 0.2 分。"""
    return [0.2 if "####" in completion_text(c) else 0.0 for c in completions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ]
        }
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        # Qwen 词表 151936，训练态 logits 巨大（batch×seq×vocab），micro-batch 必须小；
        # 用梯度累积凑等效 batch：2×8=16 条回答 = 2 道题 × 8 采样
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=512,
        temperature=1.0,                 # 采样要有多样性，组内才有分差
        beta=0.0,                        # 不用 KL 正则（省掉参考模型的显存）
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",              # 练手阶段只存最终 adapter
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        seed=42,
    )

    # 自己预载模型：bf16 + 直载 GPU。trl 默认 fp32 加载，7B 要 ~30GB 内存，
    # 会撞 pod 的 20GB cgroup 上限被静默 SIGKILL。
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        reward_funcs=[accuracy_reward, format_reward],
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules="all-linear"),
        # 显式传 tokenizer，避免 trl 走 AutoProcessor（纯文本模型没有 processor 配置）
        processing_class=AutoTokenizer.from_pretrained(args.model),
    )
    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    print("saved to", args.output_dir + "/final")


if __name__ == "__main__":
    main()
