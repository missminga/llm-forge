#!/usr/bin/env bash
# vLLM OpenAI 兼容服务管理（在远程 GPU 机器的项目根目录运行）
# 用法:
#   bash scripts/serve_vllm.sh start [模型目录] [端口]   # 后台常驻启动
#   bash scripts/serve_vllm.sh status                    # 查看进程与健康状态
#   bash scripts/serve_vllm.sh stop                      # 停止服务
set -euo pipefail

MODEL="${2:-models/Qwen2.5-0.5B-Instruct-sft-dpo}"
PORT="${3:-8000}"
LOG="outputs/logs/vllm.log"
PID_FILE="outputs/logs/vllm.pid"

case "${1:-start}" in
  start)
    mkdir -p outputs/logs
    nohup .venv/bin/vllm serve "$MODEL" \
      --served-model-name llm-forge \
      --port "$PORT" \
      --gpu-memory-utilization 0.85 \
      --max-model-len 4096 \
      > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    echo "vLLM 启动中: pid=$(cat $PID_FILE) port=$PORT log=$LOG"
    echo "等待就绪: tail -f $LOG 里出现 'Application startup complete' 即可"
    ;;
  status)
    [ -f "$PID_FILE" ] && ps -p "$(cat $PID_FILE)" -o pid,etime,cmd --no-headers || echo "未在运行"
    curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" | head -c 300 || true
    echo
    ;;
  stop)
    [ -f "$PID_FILE" ] && kill "$(cat $PID_FILE)" && rm -f "$PID_FILE" && echo "已停止" || echo "无 pid 文件"
    ;;
  *)
    echo "未知子命令: $1（start/status/stop）" >&2; exit 1
    ;;
esac
