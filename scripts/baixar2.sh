#!/bin/bash
# 12/11: o d16 deu duas voltas nos dados; o próximo tamanho precisa de mais dado. Mesma receita do baixar.sh (o DNS falha de vez em quando).
export HF_HUB_ENABLE_HF_TRANSFER=1; cd /workspace/nanochat; source .venv/bin/activate
for i in $(seq 1 60); do hf download nebius/SWE-rebench-openhands-trajectories --repo-type dataset --local-dir /workspace/dados/swe-rebench-oh && break; sleep 10; done
for i in $(seq 1 200); do hf download nvidia/Open-SWE-Traces --repo-type dataset --local-dir /workspace/dados/open-swe-traces && break; sleep 10; done
du -sh /workspace/dados/* > /workspace/dados/BAIXADO2
