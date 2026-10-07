# PoC 04: redacción de recomendaciones con un LLM local (Ollama)

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Comprobar que un modelo de lenguaje pequeño, autoalojado con Ollama, redacta una recomendación por factor de riesgo usando solo los fragmentos que recibe, y que cita únicamente los identificadores de los fragmentos de ese factor. De paso, elegir el modelo del Servicio LLM entre tres candidatos: `qwen2.5:3b`, `llama3.2:3b` y `phi4-mini`.

No valida la recuperación de fragmentos (eso lo hizo la PoC 03) ni la calidad del corpus real, que todavía no está definido.

## Datos
- **Fragmentos:** el corpus sustituto (sintético) de la PoC 03, en `corpus_poc03.py`.
- **Casos:** seis, definidos en `casos.py` antes de correr. Simulan la salida del Motor de Inteligencia: factores SHAP con las variables del diccionario de 9 y los fragmentos que la recuperación le entregó a cada factor.
- **Factores sin fragmentos:** dos casos los incluyen (hijos, edad, años del negocio), para comprobar que el modelo no aconseja sin fuente.

## Método
- Ollama `/api/chat` con un esquema JSON obligatorio (`format`), temperatura 0, semilla 42, `num_ctx` de 4096 y tope de 1024 tokens de salida.
- Una sola llamada por caso, con todos sus factores, como en el diseño.
- Cada caso se corre dos veces para medir el determinismo.

## Regla de decisión (fijada antes de correr)
Un modelo es **apto** si cumple todo lo siguiente:
- JSON válido en el 100 % de las respuestas.
- Cobertura de un elemento por factor en el 100 %.
- Citas válidas en al menos el 95 % de los factores con fragmentos: cita al menos un id y solo ids de su factor.
- `sin_fuente` correcto en el 100 % de los factores sin fragmentos.
- Cero cifras inventadas, es decir, ningún número que no esté en el factor ni en sus fragmentos.

Entre los aptos se elige el de mayor tasa de citas válidas y, si empatan, el de menor latencia.

## Cómo correrla
```
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull phi4-mini
python poc_llm.py
```

## Resultado
Corrida con 6 casos y 2 repeticiones por caso, temperatura 0, semilla 42.

| Modelo | JSON válido | Cobertura | Citas válidas | `sin_fuente` correcto | Cifras inventadas | Determinismo | Latencia media (s) | Apto |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5:3b` | 100.0 % | 83.3 % | 66.7 % | 0.0 % | 0 | 16.7 % | 3.79 | No |
| `llama3.2:3b` | 83.3 % | 83.3 % | 100.0 % | 33.3 % | 0 | 100.0 % | 4.72 | No |
| `phi4-mini` | 100.0 % | 66.7 % | 100.0 % | 50.0 % | 3 | 33.3 % | 8.46 | No |

**Modelo elegido:** ninguno, porque ningún candidato cumplió la regla de decisión.

**Versión de Ollama:** 0.40.0.

**Criterios incumplidos** (la regla exige JSON válido 100 %, cobertura 100 %, citas válidas >= 95 %, `sin_fuente` correcto 100 % y 0 cifras inventadas):

- `qwen2.5:3b`: cobertura 83.3 %; citas válidas 66.7 %; `sin_fuente` correcto 0.0 %.
- `llama3.2:3b`: JSON válido 83.3 %; cobertura 83.3 %; `sin_fuente` correcto 33.3 %.
- `phi4-mini`: cobertura 66.7 %; `sin_fuente` correcto 50.0 %; cifras inventadas 3.

### Observaciones verificadas en el JSON
- **`qwen2.5:3b`:** citó ids sin el prefijo `F` (por ejemplo `8`, `10` y `9`), por lo que el validador no los reconoce como ids del factor. En los factores sin fragmentos puso `sin_fuente = true`, pero escribió `ninguno` en `citas`, y el validador lo cuenta como una cita.
- **`llama3.2:3b`:** 1 de los 6 casos devolvió un JSON que no cumple el esquema. En `n_hijos` escribió `sin_fuente=true` como texto dentro de `citas`.
- **`phi4-mini`:** de sus 3 cifras marcadas como inventadas, 2 (`08` y `11`) son el marcador `(Citas: [F08])` y `(Citas: [F11])` que el modelo escribió dentro del texto. Son falsos positivos del validador numérico. La tercera es real: en `n_hijos`, que no tenía fragmentos, escribió "30%" y citó `F09` y `F10`, que no se le entregaron para ese factor. Aun descontando los falsos positivos, no cumple la cobertura ni `sin_fuente`.

## Archivos
- `poc_llm.py`: script.
- `casos.py`: casos de prueba.
- `corpus_poc03.py`: fragmentos sintéticos.
- `resultado_poc4.json`: métricas, evaluación por factor y respuestas completas de cada modelo.
- `salida_consola.txt`: salida completa de la consola.
