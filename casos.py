# -*- coding: utf-8 -*-
"""
Casos de prueba de la PoC 04 (fijados ANTES de correr).
Los fragmentos son los del corpus SUSTITUTO de la PoC 03 (sinteticos), porque la lista
cerrada del corpus real todavia no esta definida. Cada caso simula la salida del Motor:
factores SHAP (variables del diccionario de 9) con los fragmentos que la recuperacion le
entrego a cada factor (ids del corpus, similitud >= 0.55 en la PoC 03).
"""
from corpus_poc03 import CORPUS

FRAG = {f["id"]: f for f in CORPUS}

# Frase de cada factor tal como la ve la usuaria (nunca datos que la identifiquen)
CASOS = [
    {"id": "C1", "factores": [
        {"variable": "ratio_cuota_ingreso", "descripcion": "tu cuota equivale al 38% de tu ingreso mensual",
         "fragmentos": ["F09", "F10"]},
        {"variable": "n_otros_creditos", "descripcion": "tienes 4 creditos vigentes en otras entidades",
         "fragmentos": ["F08", "F09"]}]},
    {"id": "C2", "factores": [
        {"variable": "atraso_historial", "descripcion": "tu historial registra atrasos en pagos",
         "fragmentos": ["F11", "F12"]}]},
    {"id": "C3", "factores": [
        {"variable": "ratio_deuda_ingreso", "descripcion": "tu deuda total equivale a 3 veces tu ingreso mensual",
         "fragmentos": ["F08", "F10"]},
        {"variable": "ratio_monto_ingreso", "descripcion": "el monto de tu prestamo equivale a 5 veces tu ingreso mensual",
         "fragmentos": ["F10"]},
        {"variable": "ratio_cuota_ingreso", "descripcion": "tu cuota equivale al 45% de tu ingreso mensual",
         "fragmentos": ["F09"]}]},
    # Factor sin fragmentos: el modelo debe marcar sin_fuente = true y no aconsejar
    {"id": "C4", "factores": [
        {"variable": "n_hijos", "descripcion": "tienes 3 hijos a tu cargo", "fragmentos": []},
        {"variable": "ratio_cuota_ingreso", "descripcion": "tu cuota equivale al 33% de tu ingreso mensual",
         "fragmentos": ["F09", "F10"]}]},
    {"id": "C5", "factores": [
        {"variable": "n_otros_creditos", "descripcion": "tienes 3 creditos vigentes en otras entidades",
         "fragmentos": ["F08"]},
        {"variable": "atraso_historial", "descripcion": "tu historial registra atrasos en pagos",
         "fragmentos": ["F11"]}]},
    # Dos factores sin fragmentos (variables no accionables)
    {"id": "C6", "factores": [
        {"variable": "edad", "descripcion": "tu edad", "fragmentos": []},
        {"variable": "anios_negocio", "descripcion": "tu negocio tiene 1 anio de antiguedad", "fragmentos": []}]},
]
