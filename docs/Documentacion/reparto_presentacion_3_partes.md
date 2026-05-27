# Reparto de la Presentacion en 3 Partes

Este documento divide la presentacion de Fase 2 en 3 bloques equilibrados por contenido, carga tecnica y tiempo de exposicion. Esta pensado para repartir la defensa entre 3 personas.

## Criterio de reparto

Se priorizo este orden narrativo:
1. Contexto y problema.
2. Metodologia y modelado.
3. Resultados y cierre.

Cada parte combina una carga similar de explicacion y una progresion natural para que la defensa suene continua.

---

## Parte 1 - Introduccion, problema y EDA

### Objetivo de esta parte
Explicar el problema, por que es importante y que revelan los datos antes de modelar.

### Slides asignadas
- Slide 1 - Portada Fase 2
- Slide 2 - Objetivo de la Fase 2
- Slide 3 - Contexto del problema
- Slide 4 - Arquitectura de trabajo Fase 2
- Slide 5 - Modelos comparados

### Enfoque oral sugerido
- Presentar el proyecto y el problema clinico.
- Explicar el desbalance de clases y por que no basta con accuracy.
- Introducir la idea de pipeline reproducible.
- Nombrar los modelos evaluados sin entrar aun en resultados detallados.

### Tiempo sugerido
- 3 a 4 minutos.

### Mensaje clave
- El problema real no es clasificar, sino no perder casos positivos de ACV.

---

## Parte 2 - Tuning y seleccion del modelo

### Objetivo de esta parte
Mostrar como se compararon y optimizaron los modelos, y justificar la seleccion final.

### Slides asignadas
- Slide 6 - Optimizacion de hiperparametros
- Slide 7 - Mejor configuracion encontrada
- Slide 8 - Resultados holdout (comparacion)

### Enfoque oral sugerido
- Explicar que se uso GridSearchCV, RandomizedSearchCV y Optuna.
- Mostrar que Logistic Regression balanceada fue la mejor opcion.
- Destacar que el tuning mejoro el comportamiento frente al baseline.
- Comparar recall, F1 y ROC-AUC para justificar la seleccion.

### Tiempo sugerido
- 3 a 4 minutos.

### Mensaje clave
- El mejor modelo es el que detecta mas positivos reales sin degradar excesivamente el resto de metricas.

---

## Parte 3 - Evaluacion final, interpretabilidad y cierre

### Objetivo de esta parte
Cerrar la defensa mostrando la evidencia final, la explicacion del modelo y el cumplimiento de la rubrica.

### Slides asignadas
- Slide 9 - Evaluacion final del modelo seleccionado
- Slide 10 - Lectura clinica del trade-off
- Slide 11 - Interpretabilidad
- Slide 12 - Evidencia de cumplimiento de rubrica
- Slide 13 - Reproducibilidad
- Slide 14 - Conclusiones finales
- Slide 15 - Proximos pasos
- Slide 16 - Cierre

### Enfoque oral sugerido
- Leer las metricas finales y la matriz de confusion.
- Explicar el trade-off entre recall alto y precision baja.
- Resaltar que el modelo es apoyo de screening, no diagnostico unico.
- Cerrar con cumplimiento de rubrica, reproducibilidad y mejoras futuras.

### Tiempo sugerido
- 4 a 5 minutos.

### Mensaje clave
- El proyecto cumple la pauta y la decision final se justifica por contexto clinico y evidencia cuantitativa.

---

## Reparto recomendado entre 3 expositores

### Expositor 1
- Parte 1 completa.
- Se encarga de problema, contexto, EDA y preparacion.

### Expositor 2
- Parte 2 completa.
- Se encarga de tuning, comparacion de modelos y seleccion.

### Expositor 3
- Parte 3 completa.
- Se encarga de evaluacion final, interpretabilidad, conclusiones y cierre.

---

## Resumen corto por persona

### Persona 1
- "Por que el problema es dificil y que muestran los datos antes del modelado."

### Persona 2
- "Como se optimizaron los modelos y por que Logistic Regression fue la mejor opcion."

### Persona 3
- "Que significan los resultados finales, que limitaciones tiene el modelo y como cumple la rubrica."

---

## Si quieren un reparto mas equilibrado por tiempo

### Opcion alternativa
- Persona 1: Slides 1 a 4
- Persona 2: Slides 5 a 8
- Persona 3: Slides 9 a 16

### Ventaja
- Mantiene la narrativa simple y deja la mayor carga tecnica al centro de la defensa.

---

## Recomendacion final

Si el equipo quiere una division mas natural para defender, esta es la mejor secuencia:
1. Problema y datos.
2. Modelado y tuning.
3. Resultados, interpretacion y cierre.

Esa distribucion reduce saltos narrativos y facilita que cada expositor tenga una responsabilidad clara.
