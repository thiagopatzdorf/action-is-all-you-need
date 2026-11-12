#!/bin/bash
# Refaz a série com os dados EMBARALHADOS. Na 1ª série os shards estavam na ordem das pastas (Kimi-K2, OSS, Qwen3):
# o d8 só viu Kimi-K2 e o d12 entrou na OSS, que escreve a chamada em outro formato; a validação era só Qwen3.
# Tamanho e fonte do dado mudaram juntos, então a comparação não valia.
set -u
export NANOCHAT_BASE_DIR=/workspace/nc OMP_NUM_THREADS=1; cd /workspace/nanochat; source .venv/bin/activate
mv /workspace/nc/base_data_climbmix /workspace/nc/dados_em_ordem
python - <<'P'
import glob, random, os, pyarrow as pa, pyarrow.parquet as pq
docs = []
for a in sorted(glob.glob("/workspace/nc/dados_em_ordem/*.parquet")): docs.extend(pq.read_table(a).column("text").to_pylist())
random.Random(20261111).shuffle(docs)
saida = "/workspace/nc/base_data_climbmix"; os.makedirs(saida, exist_ok=True); n = 0; lote = []; tam = 0
for x in docs:
    lote.append(x); tam += len(x)
    if tam > 250_000_000:
        pq.write_table(pa.table({"text": lote}), f"{saida}/shard_{n:05d}.parquet", row_group_size=1024, compression="zstd"); n += 1; lote = []; tam = 0
if lote: pq.write_table(pa.table({"text": lote}), f"{saida}/shard_{n:05d}.parquet", row_group_size=1024, compression="zstd"); n += 1
open("/workspace/nc/EMBARALHADO", "w").write(f"docs={len(docs)} shards={n} semente=20261111\n")
P
cat /workspace/nc/EMBARALHADO
for d in 8 12; do
  python -m scripts.base_train --depth=$d --device-batch-size=32 --eval-every=250 --core-metric-every=-1 --sample-every=-1 --model-tag=acao2-d$d > /root/acao2-d$d.log 2>&1
  echo "acao2-d$d rc=$? $(date -u +%FT%TZ)" >> /workspace/nc/B4_ANDAMENTO
  python regua_b.py acao2-d$d 400 > /root/regua2-d$d.log 2>&1
done
date -u > /workspace/nc/B4_FIM
