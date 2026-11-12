"""Régua da pista B: em exemplos do shard de validação (nunca treinados), o modelo recebe tudo até o começo da
primeira chamada de ferramenta e tem de escrever o nome certo (e os argumentos). Sem juiz: comparação exata.
Uso: python regua_b.py <model_tag> [n]"""
import glob, json, re, sys, torch, pyarrow.parquet as pq
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
tag = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
MARCA = '"function_call": {"name": "'
ALVO = re.compile(r'([^"]+)", "arguments": "((?:[^"\\]|\\.)*)"\}')
dev = torch.device("cuda")
model, tok, meta = load_model("base", dev, phase="eval", model_tag=tag)
eng = Engine(model, tok); bos = tok.get_bos_token_id()
docs = pq.read_table(sorted(glob.glob("/workspace/nc/base_data_climbmix/*.parquet"))[-1]).column("text").to_pylist()
tot = nome_ok = args_ok = 0; acaso = 0.0; m_tot = m_nome = m_args = 0; m_acaso = 0.0; pulos = {"sem_chamada": 0, "formato": 0, "nao_cabe": 0}
for x in docs:
    if tot >= N: break
    i = x.find(MARCA)
    if i < 0: pulos["sem_chamada"] += 1; continue
    m = ALVO.match(x[i + len(MARCA):])
    if not m: pulos["formato"] += 1; continue
    ouro_nome, ouro_args = m.group(1), m.group(2)
    ids = tok.encode(x[: i + len(MARCA)], prepend=bos)
    if len(ids) > 1900: pulos["nao_cabe"] += 1; continue   # a janela do modelo é 2048: só conta exemplo que cabe inteiro
    try: nf = len(json.loads(x[: x.index("\n[")]))
    except Exception: nf = 0
    certo = ouro_nome + '", "arguments": "' + ouro_args + '"}'
    with torch.no_grad(): res, _ = eng.generate_batch(ids, num_samples=1, max_tokens=min(len(tok.encode(certo)) + 8, 140), temperature=0.0)
    saida = tok.decode(res[0][len(ids):])
    tot += 1; acaso += (1 / nf) if nf else 0
    if saida.startswith(ouro_nome + '"'): nome_ok += 1
    if saida.startswith(certo): args_ok += 1
    if nf >= 2:  # só exemplos em que há escolha de verdade
        m_tot += 1; m_acaso += 1 / nf; m_nome += saida.startswith(ouro_nome + '"'); m_args += saida.startswith(certo)
print(json.dumps({"modelo": tag, "exemplos": tot, "pulados": pulos, "nome_certo": round(nome_ok / max(tot, 1), 4),
                  "nome_e_argumentos": round(args_ok / max(tot, 1), 4), "acaso_nome": round(acaso / max(tot, 1), 4), "varias_ferramentas": {"exemplos": m_tot, "nome_certo": round(m_nome / max(m_tot, 1), 4), "nome_e_argumentos": round(m_args / max(m_tot, 1), 4), "acaso_nome": round(m_acaso / max(m_tot, 1), 4)}, "passos": meta.get("step")}))
