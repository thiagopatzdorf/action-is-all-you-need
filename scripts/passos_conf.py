"""Régua de várias etapas com confiança: igual à regua_passos.py, mas guarda por passo o texto de entrada, a chamada que o
modelo escreveu, a probabilidade que ele deu a ela (mínima e média por token) e o tempo. Serve de base para comparar com um
modelo de linguagem nos MESMOS passos e para a cascata (o pequeno propõe, o grande revisa quando a confiança é baixa).
Uso: python passos_conf.py <model_tag> <n_trajetorias> <saida.jsonl>"""
import glob, json, math, re, sys, time, torch, pyarrow.parquet as pq
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
tag, N, SAIDA = sys.argv[1], int(sys.argv[2]), sys.argv[3]
MARCA = '"function_call": {"name": "'
ALVO = re.compile(r'([^"]+)", "arguments": "((?:[^"\\]|\\.)*)"\}')
dev = torch.device("cuda")
model, tok, meta = load_model("base", dev, phase="eval", model_tag=tag)
eng = Engine(model, tok); bos = tok.get_bos_token_id()
docs = pq.read_table(sorted(glob.glob("/workspace/nc/base_data_climbmix/*.parquet"))[-1]).column("text").to_pylist()
out = open(SAIDA, "w"); traj = 0; n = 0
for d, x in enumerate(docs):
    if traj >= N: break
    pos = [m.start() for m in re.finditer(re.escape(MARCA), x)]
    if len(pos) < 2: continue
    ouro = [ALVO.match(x[i + len(MARCA):]) for i in pos]
    if not all(ouro): continue
    traj += 1
    for k, (i, m) in enumerate(zip(pos, ouro)):
        ids = tok.encode(x[: i + len(MARCA)], prepend=bos)
        if len(ids) > 1900: continue
        certo = m.group(1) + '", "arguments": "' + m.group(2) + '"}'
        t0 = time.time()
        with torch.no_grad(): res, _ = eng.generate_batch(ids, num_samples=1, max_tokens=min(len(tok.encode(certo)) + 8, 200), temperature=0.0)
        dt = time.time() - t0
        gen = res[0][len(ids):]; s = tok.decode(gen); pm = ALVO.match(s)
        # só os tokens que formam a chamada contam para a confiança (o resto é o que vem depois dela)
        fim = len(gen)
        if pm:
            for j in range(1, len(gen) + 1):
                if len(tok.decode(gen[:j])) >= pm.end(): fim = j; break
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = model(torch.tensor([ids + gen[:fim]], device=dev))
        lp = torch.log_softmax(lg[0, len(ids) - 1: len(ids) + fim - 1].float(), dim=-1)
        l = lp.gather(1, torch.tensor(gen[:fim], device=dev).unsqueeze(1)).squeeze(1)
        out.write(json.dumps({"doc": d, "passo": k, "de": len(pos), "entrada": x[: i + len(MARCA)], "ouro_nome": m.group(1), "ouro_args": m.group(2),
            "saida": s[:3000], "formou": bool(pm), "exato": s.startswith(certo), "nome_ok": s.startswith(m.group(1) + '"'),
            "p_min": round(math.exp(l.min().item()), 5), "lp_medio": round(l.mean().item(), 5), "tokens_entrada": len(ids), "tokens_saida": fim, "segundos": round(dt, 3)}) + "\n")
        n += 1
print(json.dumps({"modelo": tag, "trajetorias": traj, "passos": n}))
