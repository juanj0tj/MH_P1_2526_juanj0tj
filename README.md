# Planificación de Exámenes Universitarios mediante Búsqueda Local

Práctica 1 de la asignatura **Metaheurísticas**  
Grado en Ingeniería Informática  
Curso **2025/2026**

---

## Descripción

Este repositorio contiene una implementación en Python de una aproximación al problema de **planificación de exámenes universitarios** mediante técnicas de **búsqueda local**.

El problema consiste en asignar a cada examen:

- una **franja horaria** (`slot`)
- un **aula** (`room`)

de forma que se minimice una función objetivo que combina:

- **violaciones de restricciones duras**
- **penalizaciones por calidad del calendario**

El proyecto sigue la línea propuesta en el enunciado de la práctica: modelado del problema, generación de instancias, solución inicial, definición de vecindario, implementación de búsqueda local y análisis experimental.

---

## Objetivo del proyecto

Implementar y comparar tres fases de resolución:

1. **Heurística constructiva**
2. **Búsqueda local de primer mejor** (`first_improvement`)
3. **Búsqueda local del mejor** (`best_improvement`)

Además, el sistema genera automáticamente:

- tablas resumen en CSV
- trazas de evolución del objetivo
- gráficas para el análisis experimental

Esto permite estudiar el comportamiento de los algoritmos tal y como exige la práctica.

---

## Estructura del proyecto

```text
.
├── data/
│   ├── instances/
│   └── results/
├── src/
│   ├── constructive.py
│   ├── constraints.py
│   ├── instance_manager.py
│   ├── local_search.py
│   ├── main.py
│   ├── neighborhood.py
│   ├── objective.py
│   └── solution.py
└── README.md
```

---

## Modelado del problema

El problema aborda la planificación de exámenes universitarios a partir de una instancia que contiene:

- número de exámenes
- número de estudiantes
- número de aulas
- número de franjas horarias
- relación estudiante–examen
- tamaño de cada examen
- capacidad de cada aula
- matriz de conflictos entre exámenes

La instancia se representa mediante la clase `ExamInstance` y varias estructuras auxiliares precomputadas para acelerar consultas durante la evaluación y la búsqueda.

---

## Representación de la solución

Cada solución se representa con la clase `ExamSolution`.

Para cada examen `e` se almacenan:

- `slot[e]`: franja horaria asignada
- `room[e]`: aula asignada

Si un examen no está asignado, ambos valores valen:

```python
UNASSIGNED = -1
```

Esta representación permite:

- verificar si un examen está asignado
- reasignar un examen
- cambiar sólo su slot
- intercambiar slots entre exámenes
- copiar soluciones de manera eficiente

---

## Restricciones duras

Las restricciones duras implementadas son:

### 1. No solapamiento por estudiante
Un estudiante no puede tener dos exámenes en la misma franja horaria.

### 2. Capacidad de aula
El número de estudiantes de un examen no puede superar la capacidad del aula asignada.

### 3. Asignación única
Cada examen debe estar asignado exactamente una vez.

La implementación actual no impone necesariamente estas restricciones durante la construcción ni durante la generación de vecinos, sino que las penaliza en la función objetivo mediante un término de coste duro elevado.

---

## Función objetivo

La función objetivo implementada en `objective.py` es:

\[
f(s) = \text{hard\_penalty} \cdot \text{hard\_violations}(s) + \text{soft\_penalty}(s)
\]

donde:

- `hard_violations(s)` cuenta las violaciones de restricciones duras
- `soft_penalty(s)` combina penalizaciones blandas ponderadas

### Penalizaciones blandas

#### Exámenes consecutivos
Penaliza pares de exámenes consecutivos para un mismo estudiante dentro del mismo día.

#### Exámenes en el mismo día
Penaliza acumulaciones de varios exámenes en un mismo día para un mismo estudiante.

#### Distribución de slots
Penaliza el uso desigual de las franjas horarias, favoreciendo una distribución más homogénea.

### Configuración por defecto

La clase `ObjectiveConfig` usa por defecto:

- `hard_penalty = 100000.0`
- `w_consecutive = 5.0`
- `w_same_day = 2.0`
- `w_distribution = 1.0`
- `slots_per_day = 5`

aunque desde `main.py` el número de franjas por día se puede configurar por teclado al ejecutar una instancia.

---

## Heurística constructiva

La solución inicial se genera en `constructive.py`.

### Comportamiento

La heurística:

- recorre todos los exámenes
- asigna a cada uno un `slot` aleatorio
- asigna a cada uno un `room` aleatorio
- no filtra restricciones
- calcula el valor objetivo al final

Esto implica que:

- la solución inicial siempre está completa
- puede presentar conflictos de estudiantes
- puede violar capacidades de aula
- sirve como punto de partida para la búsqueda local

Este diseño es coherente con una estrategia basada en penalización y no en generación exclusivamente factible.

---

## Vecindario

El vecindario se implementa en `neighborhood.py`.

### Operadores disponibles

#### `change_slot`
Cambia el slot de un examen manteniendo su aula.

#### `swap_slots`
Intercambia los slots de dos exámenes ya asignados.

### Características del vecindario

- puede generarse de forma determinista o aleatoria
- puede incluir uno o ambos operadores
- no elimina de antemano movimientos no factibles
- el vecino se evalúa siempre con la función objetivo completa

Esta decisión amplía el espacio explorado y deja la gestión de la factibilidad a la penalización dura.

---

## Búsqueda local

El módulo `local_search.py` implementa dos variantes clásicas.

### `first_improvement`
- explora el vecindario en orden aleatorio
- acepta la primera solución vecina que mejora a la actual
- suele requerir menos evaluaciones por iteración
- normalmente es más rápida

### `best_improvement`
- explora todo el vecindario
- selecciona el mejor vecino que mejora a la solución actual
- suele requerir más evaluaciones
- puede tomar decisiones de mejora más informadas

### Criterios de parada

Ambas variantes pueden detenerse por:

- alcanzar `max_evaluations` (número de veces que la función objetivo se ha evaluado)
- alcanzar `max_iterations` (número de vecindarios explorados)
- no encontrar mejora en el vecindario explorado

### Métricas registradas

Cada ejecución devuelve:

- algoritmo utilizado
- solución final
- valor objetivo inicial
- valor objetivo final
- número de evaluaciones
- iteraciones
- movimientos aceptados
- tiempo de ejecución
- historia de búsqueda opcional

La historia se usa para generar trazas y gráficas.

---

## Generación y gestión de instancias

Las instancias se generan desde `instance_manager.py`.

### Parámetros principales
- `n_exams`
- `n_students`
- `n_rooms`
- `n_slots`
- `seed`

### Parámetros adicionales
- mínimo y máximo de exámenes por estudiante
- mínimo y máximo de capacidad de aula
- opción para asegurar que al menos un aula pueda albergar el examen más grande

### Valor por defecto de `n_slots`

Si no se especifica, se usa:

```python
n_slots = max(1, n_exams // 2)
```

lo cual sigue la recomendación del enunciado.

### Persistencia en disco

Cada instancia se guarda en una carpeta dentro de `data/instances/` con:

- `student_exam.csv`
- `exams.csv`
- `rooms.csv`
- `metadata.json`

Posteriormente puede volver a cargarse desde el menú principal.

---

## Ejecución del proyecto

Desde la raíz del repositorio:

```bash
python src/main.py
```

El programa abrirá un menú de terminal con las opciones principales de trabajo.

---

## Menú principal

Las opciones disponibles son:

1. Generar una nueva instancia  
2. Ejecutar una instancia existente  
3. Mostrar resumen de una instancia  
4. Listar instancias disponibles  
5. Eliminar una instancia  
0. Salir  

---

## Ejemplo de flujo de uso

### 1. Generar una instancia
Desde el menú, seleccionar la opción de generación e introducir los parámetros deseados.

### 2. Ejecutar una instancia existente
Configurar:

- función objetivo
- búsqueda local
- operadores
- semillas
- salida

### 3. Revisar resultados

Se almacenan en:

```text
data/results/<nombre_instancia>/
```

---

## Salidas generadas

### `summary.csv`
Contiene métricas finales por algoritmo.

### `traces.csv`
Contiene la evolución de la búsqueda.

### Gráficas
- evolución del objetivo vs evaluaciones
- evolución vs movimientos aceptados

---

## Reproducibilidad

El proyecto permite reproducir experimentos mediante:

- semillas
- instancias guardadas
- exportación de resultados

---

## Dependencias

```bash
pip install numpy pandas matplotlib
```

---

## Decisiones de diseño

- constructiva aleatoria
- penalización de restricciones duras
- vecindario no restringido
- diseño modular

---

## Limitaciones

- constructiva no factible
- evaluación no incremental
- coste alto en best_improvement

---

## Posibles mejoras

- evaluación incremental
- metaheurísticas avanzadas

---

## Autoría

Juan José Trillo Jiménez
Práctica 1 de Metaheurísticas  
Curso 2025/2026