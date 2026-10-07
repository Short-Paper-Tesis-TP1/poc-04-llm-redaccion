# -*- coding: utf-8 -*-
"""
PoC 04 - ITERACION 2: redaccion con LLM local (Ollama) con garantias impuestas por el codigo
OBJETIVO UNICO: el mismo de la iteracion 1 (un LLM pequeno redacta una recomendacion por factor usando
SOLO los fragmentos recibidos y citando solo ids de ese factor), despues de corregir el diseno segun
los fallos observados en la iteracion 1.

Cambios respecto a la iteracion 1 (fijados ANTES de correr; cada uno responde a un fallo observado):
  1. Los factores SIN fragmentos los resuelve el codigo y NO se envian al LLM (quedan "sin recomendacion",
     como dice el diseno: solo se recomienda con fragmentos de similitud >= 0.55).
     Fallo que corrige: sin_fuente correcto de 0 % a 50 % en la iteracion 1.
  2. El esquema JSON se arma por caso: 'variable' solo admite los factores enviados, 'citas' solo admite
     los ids entregados en ese caso (minimo 1), y la lista tiene exactamente un elemento por factor.
     Fallos que corrige: ids sin prefijo ("8" en vez de "F08"), "ninguno" como cita, factores omitidos.
  3. El validador de cifras ignora los marcadores de cita como [F08] o (F08).
     Fallo que corrige: falsos positivos del validador (error del evaluador, no del modelo).

Se mantienen: los 3 modelos, los 6 casos, 2 repeticiones, temperatura 0, semilla 42 y la regla de decision:
  APTO = JSON valido 100 % + cobertura 100 % + citas validas >= 95 % + sin_fuente correcto 100 %
         + 0 cifras inventadas. Entre aptos: mayor tasa de citas validas; empate: menor latencia.
  En esta iteracion "sin_fuente correcto" lo garantiza el codigo (paso 1) y se reporta como tal.

Uso (desde la carpeta del repo): python v2/poc_llm_v2.py   [modelos opcionales]
"""
import json
import os
import re
import sys
import time
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))          # casos.py y corpus_poc03.py de la iteracion 1
from casos import CASOS, FRAG  # noqa: E402


class _Tee:
    """Escribe en la consola y en v2/salida_consola.txt (UTF-8) a la vez."""
    def __init__(self, ruta):
        self.archivo = open(ruta, "w", encoding="utf-8")
        self.consola = sys.stdout

    def write(self, s):
        self.consola.write(s)
        self.archivo.write(s)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()


sys.stdout = _Tee(os.path.join(AQUI, "salida_consola.txt"))

OLLAMA = "http://localhost:11434"
CANDIDATOS = sys.argv[1:] or ["qwen2.5:3b", "llama3.2:3b", "phi4-mini"]
REPETICIONES = int(os.environ.get("POC04_REPS", 2))
OPCIONES = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 1024}

SYSTEM = (
    "Eres un asesor de educacion financiera para mujeres emprendedoras del Peru.\n"
    "Reglas:\n"
    "1) Usa SOLO los fragmentos entregados para cada factor. No uses conocimiento propio.\n"
    "2) En 'citas' pon los ids de los fragmentos que respaldan la recomendacion, solo ids entregados "
    "para ESE factor, escritos exactamente como aparecen (por ejemplo F08).\n"
    "3) Lenguaje simple, de tu a la usuaria, maximo 3 acciones por factor.\n"
    "4) No inventes cifras: usa solo numeros que aparezcan en el factor o en sus fragmentos.\n"
    "5) No escribas los ids dentro del texto ni de las acciones; van solo en 'citas'.\n"
    "Responde solo con el JSON pedido, exactamente un elemento por factor."
)


def esquema(factores):
    """Esquema por caso: variables e ids posibles cerrados, y un elemento por factor."""
    ids = sorted({i for f in factores for i in f["fragmentos"]})
    return {
        "type": "object",
        "properties": {"recomendaciones": {
            "type": "array", "minItems": len(factores), "maxItems": len(factores),
            "items": {"type": "object", "properties": {
                "variable": {"type": "string", "enum": [f["variable"] for f in factores]},
                "texto": {"type": "string"},
                "acciones": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                "citas": {"type": "array", "items": {"type": "string", "enum": ids}, "minItems": 1}},
                "required": ["variable", "texto", "acciones", "citas"]}}},
        "required": ["recomendaciones"],
    }


def prompt_usuario(factores):
    bloques = []
    for i, f in enumerate(factores, 1):
        lineas = [f"Factor {i} (variable: {f['variable']}): {f['descripcion']}.", "Fragmentos:"]
        for fid in f["fragmentos"]:
            fr = FRAG[fid]
            lineas.append(f"[{fid}] ({fr['fuente']}, p. {fr['pagina']}) {fr['texto']}")
        bloques.append("\n".join(lineas))
    return "\n\n".join(bloques) + "\n\nRedacta una recomendacion por factor."


def llamar(modelo, factores):
    cuerpo = {"model": modelo, "stream": False, "format": esquema(factores), "options": OPCIONES,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": prompt_usuario(factores)}]}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(cuerpo).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        datos = json.loads(r.read().decode("utf-8"))
    return datos["message"]["content"], time.time() - t0


NUM = re.compile(r"\d+(?:[.,]\d+)?")
MARCADOR = re.compile(r"[\[(]?\bF\d{2}\b[\])]?")     # [F08], (F08) o F08: son citas, no cifras


def numeros(texto):
    return {n.replace(",", ".") for n in NUM.findall(MARCADOR.sub(" ", texto))}


def evaluar(con_frag, crudo):
    try:
        recs = json.loads(crudo)["recomendaciones"]
        assert isinstance(recs, list)
        for r in recs:
            assert {"variable", "texto", "citas"} <= set(r)
    except Exception:
        return {"json_valido": False, "cobertura_ok": False, "factores": []}
    por_var = {}
    for r in recs:
        por_var.setdefault(r["variable"], []).append(r)
    esperadas = [f["variable"] for f in con_frag]
    cobertura = sorted(por_var) == sorted(esperadas) and all(len(v) == 1 for v in por_var.values())
    detalle = []
    for f in con_frag:
        r = (por_var.get(f["variable"]) or [None])[0]
        if r is None:
            detalle.append({"variable": f["variable"], "presente": False})
            continue
        citas = [str(c) for c in r.get("citas", [])]
        fuente_txt = f["descripcion"] + " " + " ".join(FRAG[i]["texto"] for i in f["fragmentos"])
        salida_txt = r["texto"] + " " + " ".join(r.get("acciones", []))
        detalle.append({"variable": f["variable"], "presente": True, "citas": citas,
                        "citas_validas": bool(citas) and set(citas) <= set(f["fragmentos"]),
                        "cifras_inventadas": sorted(numeros(salida_txt) - numeros(fuente_txt))})
    return {"json_valido": True, "cobertura_ok": cobertura, "factores": detalle}


def disponibles():
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
        return {m["name"] for m in json.loads(r.read().decode("utf-8"))["models"]}


print("=" * 78)
print("PoC 04 - ITERACION 2 | garantias por codigo | corpus sustituto de la PoC 03")
print("=" * 78)
try:
    instalados = disponibles()
except Exception as e:
    sys.exit(f"No se pudo conectar con Ollama en {OLLAMA}. Abre Ollama o ejecuta 'ollama serve'. Detalle: {e}")

# Paso 1 (codigo): separar factores con y sin fragmentos
plan = []
for c in CASOS:
    con = [f for f in c["factores"] if f["fragmentos"]]
    sin = [f["variable"] for f in c["factores"] if not f["fragmentos"]]
    plan.append({"id": c["id"], "con": con, "sin": sin})
total_sin = sum(len(p["sin"]) for p in plan)
casos_llm = [p for p in plan if p["con"]]
print(f"Factores sin fragmentos resueltos por codigo (sin recomendacion, no van al LLM): {total_sin}")
print(f"Casos que llegan al LLM: {len(casos_llm)} de {len(plan)}")

resultado = {"poc": "04_llm_redaccion_iteracion_2", "opciones": OPCIONES, "repeticiones": REPETICIONES,
             "casos": len(CASOS), "casos_al_llm": len(casos_llm), "factores_resueltos_por_codigo": total_sin,
             "modelos": {}}
for modelo in CANDIDATOS:
    tag = modelo if ":" in modelo else modelo + ":latest"
    if tag not in instalados:
        print(f"\n[{modelo}] no esta descargado. Ejecuta: ollama pull {modelo}")
        continue
    print(f"\n[{modelo}]")
    llamar(modelo, casos_llm[0]["con"])                  # calentamiento
    corridas, latencias = {}, []
    for _ in range(REPETICIONES):
        for p in casos_llm:
            crudo, seg = llamar(modelo, p["con"])
            latencias.append(seg)
            corridas.setdefault(p["id"], []).append(crudo)
    evals = [evaluar(p["con"], corridas[p["id"]][0]) for p in casos_llm]
    facts = [f for e in evals for f in e["factores"] if f.get("presente")]
    m = {
        "json_valido": sum(e["json_valido"] for e in evals) / len(evals),
        "cobertura": sum(e["cobertura_ok"] for e in evals) / len(evals),
        "factores_presentes": f"{len(facts)} / {sum(len(p['con']) for p in casos_llm)}",
        "citas_validas": (sum(f["citas_validas"] for f in facts) / len(facts)) if facts else 0.0,
        "sin_fuente_correcto": 1.0,          # garantizado por el codigo (paso 1), no por el LLM
        "cifras_inventadas": sum(len(f["cifras_inventadas"]) for f in facts),
        "determinismo": sum(len(set(v)) == 1 for v in corridas.values()) / len(corridas),
        "latencia_media_s": round(sum(latencias) / len(latencias), 2),
        "latencia_max_s": round(max(latencias), 2),
    }
    m["apto"] = (m["json_valido"] == 1 and m["cobertura"] == 1 and m["citas_validas"] >= 0.95
                 and m["cifras_inventadas"] == 0)
    for k, v in m.items():
        print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")
    resultado["modelos"][modelo] = {"metricas": m, "evaluacion": evals,
                                    "respuestas": {k: v[0] for k, v in corridas.items()}}

aptos = [(k, v["metricas"]) for k, v in resultado["modelos"].items() if v["metricas"]["apto"]]
aptos.sort(key=lambda x: (-x[1]["citas_validas"], x[1]["latencia_media_s"]))
resultado["modelo_elegido"] = aptos[0][0] if aptos else None
print("\n" + "=" * 78)
print(f"Modelos aptos: {[a[0] for a in aptos] or 'ninguno'}")
print(f"Modelo elegido: {resultado['modelo_elegido']}")
with open(os.path.join(AQUI, "resultado_poc4_v2.json"), "w", encoding="utf-8") as fh:
    json.dump(resultado, fh, ensure_ascii=False, indent=2)
print("Guardado: v2/resultado_poc4_v2.json")
