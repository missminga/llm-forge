"""vLLM OpenAI 兼容服务调用 + 简易压测。

用法（对远程服务，在项目根目录）:
    .venv/bin/python scripts/bench_chat.py
    .venv/bin/python scripts/bench_chat.py --requests 64 --concurrency 16

若在本地调用远程服务，先开转发:
    ssh -i ~/.ssh/company_server_key -N -L 8000:127.0.0.1:8000 -p 8222 root@127.0.0.1
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

PROMPT = "请用三句话介绍杭州。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="llm-forge")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    print("可用模型:", [m.id for m in client.models.list().data])

    def one_request(_):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=args.max_tokens,
            temperature=0.7,
        )
        dt = time.perf_counter() - t0
        return dt, resp.usage.completion_tokens, resp.choices[0].message.content

    dt, ntok, content = one_request(0)
    print(f"\n样例调用: {dt:.2f}s, 输出 {ntok} tokens\nQ: {PROMPT}\nA: {content}\n")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(one_request, range(args.requests)))
    total = time.perf_counter() - t0
    lats = sorted(r[0] for r in results)
    toks = sum(r[1] for r in results)
    n = len(results)
    print(f"压测: {n} 请求, 并发 {args.concurrency}, max_tokens={args.max_tokens}")
    print(f"总耗时 {total:.2f}s | QPS {n / total:.2f} | 输出吞吐 {toks / total:.1f} tok/s")
    print(f"延迟 p50 {lats[n // 2]:.2f}s | p95 {lats[min(n - 1, int(n * 0.95))]:.2f}s | max {lats[-1]:.2f}s")


if __name__ == "__main__":
    main()
