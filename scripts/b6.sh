#!/bin/bash
# Corpus v2 e o quarto tamanho (ordem do Thiago, 12/11: "segue o treinamento").
set -u
export NANOCHAT_BASE_DIR=/workspace/nc OMP_NUM_THREADS=1; cd /workspace/nanochat; source .venv/bin/activate
python /workspace/conv2.py > /root/conv2.log 2>&1 || { echo "conv2 falhou" > /workspace/nc/B6_FIM; exit 1; }
mv /workspace/nc/base_data_climbmix /workspace/nc/corpus_v1 && mv /workspace/nc/corpus_v2 /workspace/nc/base_data_climbmix
python -m scripts.base_train --depth=20 --device-batch-size=16 --eval-every=250 --core-metric-every=-1 --sample-every=-1 --model-tag=acao3-d20 > /root/acao3-d20.log 2>&1
echo "acao3-d20 rc=$? $(date -u +%FT%TZ)" >> /workspace/nc/B4_ANDAMENTO
python regua_b.py acao3-d20 400 > /root/regua3-d20.log 2>&1
date -u > /workspace/nc/B6_FIM
