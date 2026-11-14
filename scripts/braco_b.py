"""Braço B: um modelo de linguagem recebe exatamente a mesma entrada do modelo pequeno (o log até o começo da chamada) e
escreve a próxima chamada. Uso: braco_b.py <passos.jsonl> <saida.jsonl> <modelo> <raciocinio 0|1> [limite]"""
import json, os, sys, time, urllib.request, concurrent.futures as cf, threading
ENT, SAI, MODELO, RAC = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
LIM = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
CHAVE = next(l.split("=", 1)[1].strip() for l in open(os.path.expanduser("~/maos/james-gateway/.env")) if l.startswith("LITELLM_MASTER_KEY="))
SIS = ("You are given the raw log of an AI agent session: first a JSON list of the tools available, then the JSON list of messages so far. "
       "The log stops exactly where the assistant is about to make its next function call. Reply with ONLY a JSON object "
       '{"name": "<tool name>", "arguments": {<the arguments object>}} for that next call. No prose, no code fence.')
feitos = set()
if os.path.exists(SAI):
    for l in open(SAI): r = json.loads(l); feitos.add((r["doc"], r["passo"]))
passos = [json.loads(l) for l in open(ENT)][:LIM]; trava = threading.Lock(); out = open(SAI, "a"); TETO = 8.0; gasto = [0.0]
def um(p):
    if (p["doc"], p["passo"]) in feitos or gasto[0] > TETO: return
    corpo = {"model": MODELO, "max_tokens": 4000 if RAC else 800, "temperature": 0,
             "messages": [{"role": "system", "content": SIS}, {"role": "user", "content": p["entrada"]}]}
    if not RAC: corpo["reasoning"] = {"enabled": False}
    t0 = time.time(); txt = ""; uso = {}; erro = ""
    for tent in range(3):
        try:
            rq = urllib.request.Request("http://127.0.0.1:4000/v1/chat/completions", json.dumps(corpo).encode(), {"Authorization": "Bearer " + CHAVE, "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(rq, timeout=180).read()); txt = r["choices"][0]["message"].get("content") or ""; uso = r.get("usage", {}); erro = ""; break
        except Exception as e: erro = type(e).__name__ + ":" + str(e)[:80]; time.sleep(3 * (tent + 1))
    ent, sai = uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0)
    with trava:
        gasto[0] += ent * 0.15e-6 + sai * 0.6e-6   # preço de tabela com folga; o custo real sai do log do gateway
        out.write(json.dumps({"doc": p["doc"], "passo": p["passo"], "resposta": txt[:6000], "tokens_entrada": ent, "tokens_saida": sai, "segundos": round(time.time() - t0, 2), "erro": erro}) + "\n"); out.flush()
with cf.ThreadPoolExecutor(8) as ex: list(ex.map(um, passos))
print("FIM gasto_estimado_usd", round(gasto[0], 3))
