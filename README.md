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

## Resultado: iteración 1
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
- `v2/poc_llm_v2.py`: script de la iteración 2.
- `v2/resultado_poc4_v2.json`: métricas, evaluación por factor y respuestas completas de la iteración 2.
- `v2/salida_consola.txt`: salida completa de la consola de la iteración 2.

## Iteración 2

Segunda iteración con tres cambios de diseño, fijados antes de volver a correr. Cada uno responde a un fallo observado en la iteración 1. Se mantienen los mismos tres modelos, los mismos seis casos, las 2 repeticiones, la temperatura 0, la semilla 42 y la misma regla de decisión.

### Cambios de diseño
1. Los factores **sin fragmentos** los resuelve el código y no se envían al modelo de lenguaje: quedan sin recomendación, como dice el diseño (solo se recomienda con fragmentos de similitud de 0.55 o más). *Fallo que corrige:* `sin_fuente` correcto de 0 % a 50 % en la iteración 1.
2. El **esquema JSON** se arma por caso: `variable` solo admite los factores enviados, `citas` solo admite los ids entregados en ese caso (mínimo 1) y la lista tiene exactamente un elemento por factor. *Fallos que corrige:* ids sin prefijo (`8` en vez de `F08`), `ninguno` como cita y factores omitidos.
3. El **validador de cifras** ignora los marcadores de cita como `[F08]` o `(F08)`. *Fallo que corrige:* falsos positivos del validador (error del evaluador, no del modelo).

En la iteración 2 el criterio `sin_fuente` correcto **lo garantiza el código**, porque los factores sin fragmentos no se envían al modelo. Por eso se reporta como 100 % para los tres modelos y no mide al modelo de lenguaje.

### Resultado: iteración 2

Corrida con 5 de los 6 casos, que son los que llegan al modelo, y 3 factores resueltos por el código. Cada caso se corre 2 veces.

| Modelo | JSON válido | Cobertura | Citas válidas | `sin_fuente` correcto | Cifras inventadas | Determinismo | Latencia media (s) | Apto |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5:3b` | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 0 | 20.0 % | 4.10 | Sí |
| `llama3.2:3b` | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 2 | 20.0 % | 3.91 | No |
| `phi4-mini` | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 9 | 20.0 % | 12.44 | No |

**Modelo elegido:** `qwen2.5:3b`.

**Criterio incumplido** por los modelos no aptos: cifras inventadas (la regla exige 0).

- `llama3.2:3b` (2): en `n_otros_creditos` del caso C1 escribió "tener 3 o mas creditos activos" (el fragmento F08 escribe "tres"); en `ratio_cuota_ingreso` del caso C4 escribió "divide por 12".
- `phi4-mini` (9): son la numeración "1.", "2." y "3." que escribió dentro de `texto` en `ratio_cuota_ingreso` y `n_otros_creditos` del caso C1 y en `ratio_cuota_ingreso` del caso C4 (3 cifras por factor).

El determinismo fue de 20 % en los tres modelos, aunque la temperatura es 0 y la semilla fija. No forma parte de la regla de decisión.

### Cómo correrla
```
python v2/poc_llm_v2.py
```
Se ejecuta desde la carpeta del repositorio y usa `casos.py` y `corpus_poc03.py` de la iteración 1. Genera `v2/resultado_poc4_v2.json` y `v2/salida_consola.txt`.
