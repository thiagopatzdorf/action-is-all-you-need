#!/bin/bash
# Prepara o treino do zero: dependências do nanochat e o Toucan convertido para o formato dele (parquet com coluna text).
set -u
export NANOCHAT_BASE_DIR=/workspace/nc; mkdir -p $NANOCHAT_BASE_DIR/base_data_climbmix
cd /workspace/nanochat
pip install -q --break-system-packages uv > /root/uv.log 2>&1
(uv sync --extra gpu >> /root/uv.log 2>&1; echo "uv_rc=$?" >> /root/uv.log) &
while [ ! -f /workspace/dados/BAIXADO ]; do sleep 20; done
python - <<'P'
import glob, os, pyarrow as pa, pyarrow.parquet as pq
saida = "/workspace/nc/base_data_climbmix"; arqs = sorted(glob.glob("/workspace/dados/toucan/*/*.parquet"))
n = docs = chars = 0; lote = []; tam = 0
def gravar():
    global n, lote, tam
    pq.write_table(pa.table({"text": lote}), f"{saida}/shard_{n:05d}.parquet", row_group_size=1024, compression="zstd"); n += 1; lote = []; tam = 0
for a in arqs:
    f = pq.ParquetFile(a)
    for b in f.iter_batches(batch_size=2000, columns=["tools", "messages"]):
        for t, m in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            x = (t or "") + "\n" + (m or ""); lote.append(x); tam += len(x); docs += 1; chars += len(x)
            if tam > 250_000_000: gravar()
if lote: gravar()
open("/workspace/nc/CONVERTIDO", "w").write(f"arquivos={len(arqs)} shards={n} docs={docs} chars={chars}\n")
P
wait; date -u > /workspace/nc/PREP_FIM
