#!/bin/bash
LOG=/workspace/project/gpu_monitor.log
: > "$LOG"
prev=""
while true; do
  tot=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')
  line="$(printf '%6s' "$tot") MiB | apps: ${apps:-none}"
  if [ "$line" != "$prev" ]; then
    echo "$(date +%H:%M:%S)  $line" >> "$LOG"
    prev="$line"
  fi
  sleep 1
done
