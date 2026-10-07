# -*- coding: utf-8 -*-
"""
PoC 04 - Redaccion de recomendaciones con un LLM local (Ollama)
OBJETIVO UNICO: comprobar que un LLM pequeno autoalojado redacta una recomendacion por factor
usando SOLO los fragmentos recibidos, y que cita unicamente los ids de los fragmentos de ese
factor. De paso elige el modelo del Servicio LLM entre tres candidatos.

NO valida la recuperacion (eso es la PoC 03) ni la calidad del corpus real (aun no existe):
los fragmentos son el corpus SUSTITUTO de la PoC 03.

Regla de decision (fijada ANTES de correr). Un modelo es APTO si cumple todo:
  - JSON valido segun el esquema en el 100 % de las respuestas.
  - Cobertura: devuelve exactamente un elemento por cada factor recibido, en el 100 %.
  - Citas validas: >= 95 % de los factores con fragmentos citan solo ids de SU factor (y al menos uno).
  - sin_fuente correcto: el 100 % de los factores SIN fragmentos marcan sin_fuente = true y no citan.
  - Cifras inventadas: 0 (ninguna cifra en el texto que no este en el factor o en sus fragmentos).
Entre los aptos se elige el de mayor tasa de citas validas; si empatan, el de menor latencia.

Uso: ollama debe estar corriendo (ollama serve) y los modelos descargados (ollama pull ...).
     python poc_llm.py                      -> los tres candidatos
     python poc_llm.py qwen2.5:3b           -> solo los modelos indicados
"""
import json
import os
import re
import sys
import time
import urllib.request

from casos import CASOS, FRAG


class _Tee:
    """Escribe en la consola y en salida_consola.txt (UTF-8) a la vez."""
    def __init__(self, ruta):
        self.archivo = open(ruta, "w", encoding="utf-8")
        self.consola = sys.stdout
    def write(self, s):
        self.consola.write(s)
        self.archivo.write(s)
    def flush(self):
        self.consola.flush()
        self.archivo.flush()


sys.stdout = _Tee("salida_consola.txt")

OLLAMA = "http://localhost:11434"
CANDIDATOS = sys.argv[1:] or ["qwen2.5:3b", "llama3.2:3b", "phi4-mini"]
REPETICIONES = int(os.environ.get("POC04_REPS", 2))  # la 2.a corrida mide determinismo (temperatura 0 + semilla fija)
OPCIONES = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 1024}  # tope de tokens de salida

SYSTEM = (
    "Eres un asesor de educacion financiera para mujeres emprendedoras del Peru.\n"
    "Reglas:\n"
    "1) Usa SOLO los fragmentos entregados para cada factor. No uses conocimiento propio.\n"
    "2) Cada recomendacion cita en 'citas' los ids de los fragmentos que la respaldan, y solo ids "
    "entregados para ESE factor.\n"
    "3) Si un factor no tiene fragmentos, escribe sin_fuente=true, deja citas vacio y no aconsejes.\n"
    "4) Lenguaje simple, de tu a la usuaria, maximo 3 acciones por factor.\n"
    "5) No inventes cifras: usa solo numeros que aparezcan en el factor o en sus fragmentos.\n"
    "Responde solo con el JSON pedido, un elemento por factor."
)

ESQUEMA = {
    "type": "object",
    "properties": {"recomendaciones": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "variable": {"type": "string"},
            "texto": {"type": "string"},
            "acciones": {"type": "array", "items": {"type": "string"}},
            "citas": {"type": "array", "items": {"type": "string"}},
            "sin_fuente": {"type": "boolean"}},
        "required": ["variable", "texto", "citas", "sin_fuente"]}}},
    "required": ["recomendaciones"],
}


def prompt_usuario(caso):
    bloques = []
    for i, f in enumerate(caso["factores"], 1):
        lineas = [f"Factor {i} (variable: {f['variable']}): {f['descripcion']}.", "Fragmentos:"]
        if f["fragmentos"]:
            for fid in f["fragmentos"]:
                fr = FRAG[fid]
                lineas.append(f"[{fid}] ({fr['fuente']}, p. {fr['pagina']}) {fr['texto']}")
        else:
            lineas.append("(ninguno)")
        bloques.append("\n".join(lineas))
    return "\n\n".join(bloques) + "\n\nRedacta una recomendacion por factor."


def llamar(modelo, caso):
    cuerpo = {"model": modelo, "stream": False, "format": ESQUEMA, "options": OPCIONES,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": prompt_usuario(caso)}]}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(cuerpo).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        datos = json.loads(r.read().decode("utf-8"))
    return datos["message"]["content"], time.time() - t0


NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numeros(texto):
    return {n.replace(",", ".") for n in NUM.findall(texto)}


def evaluar(caso, crudo):
    """Devuelve el detalle por factor y si el JSON fue valido."""
    try:
        recs = json.loads(crudo)["recomendaciones"]
        assert isinstance(recs, list)
        for r in recs:
            assert {"variable", "texto", "citas", "sin_fuente"} <= set(r)
    except Exception:
        return {"json_valido": False, "cobertura_ok": False, "factores": []}
    por_var = {}
    for r in recs:
        por_var.setdefault(r["variable"], []).append(r)
    esperadas = [f["variable"] for f in caso["factores"]]
    cobertura = sorted(por_var) == sorted(esperadas) and all(len(v) == 1 for v in por_var.values())
    detalle = []
    for f in caso["factores"]:
        r = (por_var.get(f["variable"]) or [None])[0]
        if r is None:
            detalle.append({"variable": f["variable"], "presente": False})
            continue
        citas = [str(c).strip("[] ") for c in r.get("citas", [])]
        fuente_txt = f["descripcion"] + " " + " ".join(FRAG[i]["texto"] for i in f["fragmentos"])
        salida_txt = r["texto"] + " " + " ".join(r.get("acciones", []))
        inventadas = sorted(numeros(salida_txt) - numeros(fuente_txt))
        d = {"variable": f["variable"], "presente": True, "con_fragmentos": bool(f["fragmentos"]),
             "citas": citas, "sin_fuente": r["sin_fuente"], "cifras_inventadas": inventadas}
        if f["fragmentos"]:
            d["citas_validas"] = bool(citas) and set(citas) <= set(f["fragmentos"])
        else:
            d["sin_fuente_correcto"] = r["sin_fuente"] is True and not citas
        detalle.append(d)
    return {"json_valido": True, "cobertura_ok": cobertura, "factores": detalle}


def disponibles():
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
        return {m["name"] for m in json.loads(r.read().decode("utf-8"))["models"]}


print("=" * 78)
print("PoC 04 - REDACCION CON LLM LOCAL (Ollama) | corpus sustituto de la PoC 03")
print("=" * 78)
try:
    instalados = disponibles()
except Exception as e:
    sys.exit(f"No se pudo conectar con Ollama en {OLLAMA}. Ejecuta 'ollama serve'. Detalle: {e}")

resultado = {"poc": "04_llm_redaccion", "opciones": OPCIONES, "repeticiones": REPETICIONES,
             "casos": len(CASOS), "modelos": {}}
for modelo in CANDIDATOS:
    nombre_tag = modelo if ":" in modelo else modelo + ":latest"
    if nombre_tag not in instalados:
        print(f"\n[{modelo}] no esta descargado. Ejecuta: ollama pull {modelo}")
        continue
    print(f"\n[{modelo}]")
    llamar(modelo, CASOS[0])                        # calentamiento: carga el modelo en memoria
    corridas, latencias = {}, []
    for rep in range(REPETICIONES):
        for caso in CASOS:
            crudo, seg = llamar(modelo, caso)
            latencias.append(seg)
            corridas.setdefault(caso["id"], []).append(crudo)
    evals = [evaluar(c, corridas[c["id"]][0]) for c in CASOS]
    facts = [f for e in evals for f in e["factores"] if f.get("presente")]
    con = [f for f in facts if f["con_fragmentos"]]
    sin = [f for f in facts if not f["con_fragmentos"]]
    total_factores = sum(len(c["factores"]) for c in CASOS)
    m = {
        "json_valido": sum(e["json_valido"] for e in evals) / len(evals),
        "cobertura": sum(e["cobertura_ok"] for e in evals) / len(evals),
        "factores_presentes": f"{len(facts)} / {total_factores}",
        "citas_validas": (sum(f["citas_validas"] for f in con) / len(con)) if con else 0.0,
        "sin_fuente_correcto": (sum(f["sin_fuente_correcto"] for f in sin) / len(sin)) if sin else 0.0,
        "cifras_inventadas": sum(len(f["cifras_inventadas"]) for f in facts),
        "determinismo": sum(len(set(v)) == 1 for v in corridas.values()) / len(corridas),
        "latencia_media_s": round(sum(latencias) / len(latencias), 2),
        "latencia_max_s": round(max(latencias), 2),
    }
    m["apto"] = (m["json_valido"] == 1 and m["cobertura"] == 1 and m["citas_validas"] >= 0.95
                 and m["sin_fuente_correcto"] == 1 and m["cifras_inventadas"] == 0)
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
with open("resultado_poc4.json", "w", encoding="utf-8") as fh:
    json.dump(resultado, fh, ensure_ascii=False, indent=2)
print("Guardado: resultado_poc4.json")
