#!/bin/bash
# Terceiro tamanho da série embaralhada (ordem do Thiago, 11/11: "Pode fazer").
set -u
export NANOCHAT_BASE_DIR=/workspace/nc OMP_NUM_THREADS=1; cd /workspace/nanochat; source .venv/bin/activate
python -m scripts.base_train --depth=16 --device-batch-size=32 --eval-every=250 --core-metric-every=-1 --sample-every=-1 --model-tag=acao2-d16 > /root/acao2-d16.log 2>&1
echo "acao2-d16 rc=$? $(date -u +%FT%TZ)" >> /workspace/nc/B4_ANDAMENTO
python regua_b.py acao2-d16 400 > /root/regua2-d16.log 2>&1
date -u > /workspace/nc/B5_FIM
