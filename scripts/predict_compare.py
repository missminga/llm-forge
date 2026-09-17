"""多阶段模型对比：基座 / SFT / SFT+DPO 对同一组问题的回答。

不存在的模型目录自动跳过（如 DPO 还没训练时）。
用法（远程项目根目录）：
    .venv/bin/python scripts/predict_compare.py
"""

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# (标签, 模型路径, LoRA adapter 路径或 None)
STAGES = [
    ("基座", "models/Qwen2.5-7B-Instruct", None),
    ("SFT", "models/Qwen2.5-7B-Instruct-sft", None),
    ("SFT+DPO", "models/Qwen2.5-7B-Instruct-sft", "outputs/dpo/qwen25-7b-instruct-lora"),
]

PROMPTS = [
    "你是谁？",
    "请解释什么是人工智能。",
    "我心情不好，能给我一些建议吗？",
]


def chat(model, tok, prompt: str) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(STAGES[0][1])
    for tag, model_path, adapter_path in STAGES:
        if not os.path.exists(model_path) or (adapter_path and not os.path.exists(adapter_path)):
            print(f"[{tag}] 跳过：{adapter_path or model_path} 不存在\n")
            continue
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda"
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        for p in PROMPTS:
            print(f"[{tag}] Q: {p}\n[{tag}] A: {chat(model, tok, p)}\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
