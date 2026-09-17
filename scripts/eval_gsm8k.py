"""GSM8K 测试集准确率评估：对比 加/不加 GRPO adapter 的答题正确率。

用法（远程项目根目录）：
  .venv-train/bin/python scripts/eval_gsm8k.py \
      --model models/Qwen2.5-7B-Instruct-sft \
      --adapter outputs/grpo/gsm8k-7b/final \
      --n 100
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_grpo import SYSTEM_PROMPT, extract_final, to_number  # 复用训练同款规则


def build_inputs(tok, question, device):
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok(text, return_tensors="pt").to(device)


def accuracy(model, tok, ds, n: int) -> float:
    correct = 0
    for i, row in enumerate(ds.select(range(min(n, len(ds))))):
        inputs = build_inputs(tok, row["question"], model.device)
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gold = to_number(row["answer"].split("####")[-1])
        pred = to_number(extract_final(text))
        if pred is not None and gold is not None and abs(pred - gold) < 1e-4:
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}: 累计准确率 {correct / (i + 1):.1%}", flush=True)
    return correct / min(n, len(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    test = load_dataset("openai/gsm8k", "main", split="test")
    tok = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print(f"[基线 {args.model}]")
    base_acc = accuracy(model, tok, test, args.n)
    print(f"基线准确率: {base_acc:.1%}\n")

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        print(f"[+GRPO adapter {args.adapter}]")
        grpo_acc = accuracy(model, tok, test, args.n)
        print(f"GRPO 后准确率: {grpo_acc:.1%}")
        print(f"\n提升: {grpo_acc - base_acc:+.1%}")


if __name__ == "__main__":
    main()
