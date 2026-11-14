"""Régua de várias etapas (histórico de ouro): em cada chamada de ferramenta de uma trajetória nunca treinada, o modelo
recebe tudo o que veio antes (com as chamadas e resultados certos) e escreve a chamada seguinte. Sem juiz: comparação exata.
Grava um JSONL por passo (para os braços com LLM reusarem os mesmos passos) e imprime o resumo.
Uso: python regua_passos.py <model_tag> <n_trajetorias> <saida.jsonl>"""
import glob, json, re, sys, torch, pyarrow.parquet as pq
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
tag, N, SAIDA = sys.argv[1], int(sys.argv[2]), sys.argv[3]
MARCA = '"function_call": {"name": "'
ALVO = re.compile(r'([^"]+)", "arguments": "((?:[^"\\]|\\.)*)"\}')
model, tok, meta = load_model("base", torch.device("cuda"), phase="eval", model_tag=tag)
eng = Engine(model, tok); bos = tok.get_bos_token_id()
docs = pq.read_table(sorted(glob.glob("/workspace/nc/base_data_climbmix/*.parquet"))[-1]).column("text").to_pylist()
out = open(SAIDA, "w"); traj = 0; pulos = {"menos_de_2_chamadas": 0, "formato": 0}
passos = nome = tudo = fora = 0; t_inteiras = t_ok = 0; por_pos = {}
for d, x in enumerate(docs):
    if traj >= N: break
    pos = [m.start() for m in re.finditer(re.escape(MARCA), x)]
    if len(pos) < 2: pulos["menos_de_2_chamadas"] += 1; continue
    ouro = [ALVO.match(x[i + len(MARCA):]) for i in pos]
    if not all(ouro): pulos["formato"] += 1; continue
    try: nf = len(json.loads(x[: x.index("\n[")]))
    except Exception: nf = 0
    traj += 1; certos = 0; cabem = 0
    for k, (i, m) in enumerate(zip(pos, ouro)):
        ids = tok.encode(x[: i + len(MARCA)], prepend=bos)
        if len(ids) > 1900: fora += 1; continue   # janela de 2048: passo que não cabe não é pontuado
        cabem += 1
        certo = m.group(1) + '", "arguments": "' + m.group(2) + '"}'
        with torch.no_grad(): res, _ = eng.generate_batch(ids, num_samples=1, max_tokens=min(len(tok.encode(certo)) + 8, 200), temperature=0.0)
        s = tok.decode(res[0][len(ids):]); n_ok = s.startswith(m.group(1) + '"'); a_ok = s.startswith(certo)
        passos += 1; nome += n_ok; tudo += a_ok; certos += a_ok
        p = por_pos.setdefault(min(k, 5), [0, 0, 0]); p[0] += 1; p[1] += n_ok; p[2] += a_ok
        out.write(json.dumps({"doc": d, "passo": k, "de": len(pos), "ferramentas": nf, "corte": i + len(MARCA), "ouro": certo, "saida": s[:400], "nome_ok": bool(n_ok), "tudo_ok": bool(a_ok)}) + "\n")
    if cabem == len(pos): t_inteiras += 1; t_ok += (certos == cabem)
r = lambda a, b: round(a / max(b, 1), 4)
print(json.dumps({"modelo": tag, "trajetorias": traj, "pulados": pulos, "passos_pontuados": passos, "passos_fora_da_janela": fora,
  "nome_certo": r(nome, passos), "nome_e_argumentos": r(tudo, passos), "trajetorias_que_cabem_inteiras": t_inteiras, "trajetoria_inteira_certa": r(t_ok, t_inteiras),
  "por_posicao(0..5+)": {str(k): {"n": v[0], "nome": r(v[1], v[0]), "tudo": r(v[2], v[0])} for k, v in sorted(por_pos.items())}}))
