"""Corpus v2 do modelo de ação (12/11): o corpus v1 (Toucan + SWE-smith) + amostra de trajetórias RESOLVIDAS do Open-SWE-Traces e do
SWE-rebench-openhands, tudo embaralhado junto com semente fixa. O shard de validação do v1 fica igual e por último, para a régua comparar.
Embaralha em disco (baldes), sem carregar tudo na memória."""
import glob, json, os, random, shutil, pyarrow as pa, pyarrow.parquet as pq
SEM = 20261112; rnd = random.Random(SEM); V1 = "/workspace/nc/base_data_climbmix"; V2 = "/workspace/nc/corpus_v2"; TMP = "/workspace/nc/baldes"
ALVO_OST = 20_000_000_000; ALVO_REB = 3_000_000_000; NB = 240
os.makedirs(TMP, exist_ok=True); os.makedirs(V2, exist_ok=True)
baldes = [open(f"{TMP}/b{i:03d}.jsonl", "w") for i in range(NB)]; cont = {"v1": [0, 0], "ost": [0, 0], "reb": [0, 0]}
def por(fonte, texto):
    baldes[rnd.randrange(NB)].write(json.dumps(texto) + "\n"); cont[fonte][0] += 1; cont[fonte][1] += len(texto)
v1 = sorted(glob.glob(V1 + "/shard_*.parquet"))
for a in v1[:-1]:
    for x in pq.read_table(a).column("text").to_pylist(): por("v1", x)
ost = sorted(glob.glob("/workspace/dados/open-swe-traces/data/**/*.parquet", recursive=True)); teto = ALVO_OST // len(ost)
for a in ost:
    usado = 0; f = pq.ParquetFile(a)
    for b in f.iter_batches(batch_size=64, columns=["tools", "messages", "resolved"]):
        for t, m, r in zip(b.column(0).to_pylist(), b.column(1).to_pylist(), b.column(2).to_pylist()):
            if r != 1: continue
            x = "\n".join(t or []) + "\n" + json.dumps(m, ensure_ascii=False); por("ost", x); usado += len(x)
        if usado >= teto: break
f = pq.ParquetFile("/workspace/dados/swe-rebench-oh/trajectories.parquet"); usado = 0
for b in f.iter_batches(batch_size=32, columns=["tools", "trajectory", "resolved"]):
    for t, m, r in zip(b.column(0).to_pylist(), b.column(1).to_pylist(), b.column(2).to_pylist()):
        if r != 1: continue
        x = json.dumps(t, ensure_ascii=False) + "\n" + json.dumps(m, ensure_ascii=False); por("reb", x); usado += len(x)
    if usado >= ALVO_REB: break
for b in baldes: b.close()
n = 0
for i in range(NB):
    docs = [json.loads(l) for l in open(f"{TMP}/b{i:03d}.jsonl")]; rnd.shuffle(docs)
    pq.write_table(pa.table({"text": docs}), f"{V2}/shard_{n:05d}.parquet", row_group_size=1024, compression="zstd"); n += 1; os.remove(f"{TMP}/b{i:03d}.jsonl")
shutil.copy(v1[-1], f"{V2}/shard_{n:05d}.parquet")
open("/workspace/nc/CORPUS_V2", "w").write(json.dumps({"semente": SEM, "shards_treino": n, "validacao": os.path.basename(v1[-1]), "fontes": cont}) + "\n")
