"""Pontua os três braços nos mesmos passos, por igualdade de nome e de argumentos como objeto (não como texto)."""
import json, re, sys
ALVO = re.compile(r'([^"]+)", "arguments": "((?:[^"\\]|\\.)*)"\}')
def obj_args(esc):
    try: return json.loads(json.loads('"' + esc + '"'))
    except Exception: return ("?", esc)
def do_pequeno(p):
    m = ALVO.match(p["saida"]); return (m.group(1), obj_args(m.group(2))) if m else (p["saida"].split('"')[0], None)
def do_llm(t):
    t = t.strip()
    if t.startswith("```"): t = t.strip("`"); t = t[t.find("{"):]
    i, j = t.find("{"), t.rfind("}")
    try: o = json.loads(t[i:j + 1]); a = o.get("arguments"); a = json.loads(a) if isinstance(a, str) else a; return o.get("name"), a
    except Exception: return None, None
peq = {(p["doc"], p["passo"]): p for p in map(json.loads, open(sys.argv[1]))}
llm = {(p["doc"], p["passo"]): p for p in map(json.loads, open(sys.argv[2]))} if len(sys.argv) > 2 else {}
ch = [k for k in peq if (not llm) or (k in llm and not llm[k]["erro"])]
n = len(ch); r = lambda a: round(100 * a / max(n, 1), 1)
P = {}; L = {}
for k in ch:
    p = peq[k]; ouro = (p["ouro_nome"], obj_args(p["ouro_args"])); a = do_pequeno(p); P[k] = (a[0] == ouro[0], a == ouro)
    if llm: b = do_llm(llm[k]["resposta"]); L[k] = (b[0] == ouro[0], b == ouro)
res = {"passos": n, "pequeno": {"nome": r(sum(v[0] for v in P.values())), "chamada": r(sum(v[1] for v in P.values())), "exato_texto": r(sum(peq[k]["exato"] for k in ch)),
       "seg_medio": round(sum(peq[k]["segundos"] for k in ch) / max(n, 1), 3)}}
if llm:
    te = sum(llm[k]["tokens_entrada"] for k in ch); ts = sum(llm[k]["tokens_saida"] for k in ch)
    res["llm"] = {"nome": r(sum(v[0] for v in L.values())), "chamada": r(sum(v[1] for v in L.values())), "seg_medio": round(sum(llm[k]["segundos"] for k in ch) / n, 2), "tokens_entrada": te, "tokens_saida": ts}
    res["ambos_certos"] = r(sum(P[k][1] and L[k][1] for k in ch)); res["so_pequeno"] = r(sum(P[k][1] and not L[k][1] for k in ch)); res["so_llm"] = r(sum(L[k][1] and not P[k][1] for k in ch))
    res["cascata"] = []
    for lim in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01):
        sobe = [k for k in ch if peq[k]["p_min"] < lim or not peq[k]["formou"]]
        ok = sum((L[k][1] if k in set(sobe) else P[k][1]) for k in ch)
        res["cascata"].append({"limiar_p_min": lim, "sobe_para_llm": r(len(sobe)), "chamada": r(ok), "tokens_llm": sum(llm[k]["tokens_entrada"] + llm[k]["tokens_saida"] for k in sobe)})
print(json.dumps(res, ensure_ascii=False, indent=1))
