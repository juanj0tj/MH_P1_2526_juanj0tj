from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from constraints import can_assign_exam
from instance_manager import ExamInstance
from objective import (
    ObjectiveConfig,
    objective_breakdown,
    objective_value,
    soft_penalty,
)
from solution import ExamSolution, UNASSIGNED


# ==============================================================================

# CONFIGURACIÓN Y RESULTADOS ===================================================


@dataclass(frozen=True)
class ConstructiveConfig:
    """
    Parámetros de la heurística constructiva.

    Convenciones adoptadas:
    - La constructiva SOLO acepta asignaciones factibles.
    - Los exámenes se ordenan por dificultad de colocación.
    - Entre candidatos factibles, se elige el que minimiza la penalización
      blanda de la solución parcial.
    """

    objective_config: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    # Ordenación de exámenes
    sort_by_conflicts: bool = True
    sort_by_size: bool = True

    # Ordenación de candidatos
    prefer_less_loaded_slots: bool = True
    prefer_best_fit_room: bool = True

    # Si True, se intenta evaluar todos los candidatos factibles y escoger el mejor.
    # Si False, se acepta el primer candidato factible en el orden generado.
    evaluate_all_feasible_candidates: bool = True


@dataclass(frozen=True)
class ConstructiveResult:
    """
    Resultado de la heurística constructiva.
    """

    solution: ExamSolution
    assigned_exams: int
    unassigned_exams: List[int]
    objective: float
    soft_penalty: float
    is_complete: bool
    breakdown: dict[str, float]


# ==============================================================================

# UTILIDADES INTERNAS ==========================================================


def _exam_conflict_degree(instance: ExamInstance, exam_id: int) -> int:
    """
    Número de exámenes con los que conflictúa exam_id.
    """
    return int(instance.conflict_matrix[exam_id].sum())


def _exam_size(instance: ExamInstance, exam_id: int) -> int:
    """
    Número de estudiantes matriculados en el examen.
    """
    return int(instance.exam_sizes[exam_id])


def _room_capacity(instance: ExamInstance, room_id: int) -> int:
    return int(instance.room_capacities[room_id])


def _slot_loads(solution: ExamSolution) -> List[int]:
    """
    Carga actual por slot: número de exámenes asignados a cada franja.
    """
    loads = [0] * solution.instance.n_slots

    for exam_id in range(solution.instance.n_exams):
        slot = int(solution.slot[exam_id])
        if slot != UNASSIGNED:
            loads[slot] += 1

    return loads


def _sort_exams(instance: ExamInstance, config: ConstructiveConfig) -> List[int]:
    """
    Ordena los exámenes por dificultad de colocación.

    Criterio lexicográfico por defecto:
    1) mayor número de conflictos
    2) mayor tamaño
    3) menor exam_id
    """
    exam_ids = list(range(instance.n_exams))

    def exam_key(exam_id: int) -> tuple[int, int, int]:
        conflicts = (
            _exam_conflict_degree(instance, exam_id) if config.sort_by_conflicts else 0
        )
        size = _exam_size(instance, exam_id) if config.sort_by_size else 0
        return (-conflicts, -size, exam_id)

    exam_ids.sort(key=exam_key)
    return exam_ids


def _sorted_slots(solution: ExamSolution, config: ConstructiveConfig) -> List[int]:
    """
    Ordena slots. Por defecto, primero los menos cargados.
    """
    slot_ids = list(range(solution.instance.n_slots))

    if not config.prefer_less_loaded_slots:
        return slot_ids

    loads = _slot_loads(solution)
    slot_ids.sort(key=lambda s: (loads[s], s))
    return slot_ids


def _sorted_rooms_for_exam(
    instance: ExamInstance,
    exam_id: int,
    config: ConstructiveConfig,
) -> List[int]:
    """
    Devuelve las aulas ordenadas para un examen.

    Por defecto:
    - solo se consideran aulas con capacidad suficiente
    - se prioriza best-fit: menor holgura capacity - exam_size
    """
    exam_size = _exam_size(instance, exam_id)

    feasible_rooms = [
        room_id
        for room_id in range(instance.n_rooms)
        if _room_capacity(instance, room_id) >= exam_size
    ]

    if not config.prefer_best_fit_room:
        feasible_rooms.sort()
        return feasible_rooms

    feasible_rooms.sort(
        key=lambda room_id: (
            _room_capacity(instance, room_id) - exam_size,
            _room_capacity(instance, room_id),
            room_id,
        )
    )
    return feasible_rooms


# ==============================================================================

# GENERACIÓN Y EVALUACIÓN DE CANDIDATOS ========================================


def iter_candidate_assignments(
    solution: ExamSolution,
    exam_id: int,
    config: ConstructiveConfig,
) -> Iterable[tuple[int, int]]:
    """
    Genera candidatos (slot, room) en un orden razonable para el examen.

    Orden:
    - slots menos cargados primero
    - dentro de cada slot, aulas compatibles por best-fit
    """
    instance = solution.instance
    slots = _sorted_slots(solution, config)
    rooms = _sorted_rooms_for_exam(instance, exam_id, config)

    for slot in slots:
        for room in rooms:
            yield slot, room


def _evaluate_candidate(
    solution: ExamSolution,
    exam_id: int,
    slot: int,
    room: int,
    config: ConstructiveConfig,
) -> tuple[tuple[float, int, int, int, int], ExamSolution]:
    """
    Evalúa un candidato factible para usarlo en la selección greedy.

    Devuelve:
    - una clave lexicográfica de minimización
    - la solución parcial resultante

    Criterio:
    1) menor soft_penalty de la solución parcial
    2) menor carga del slot elegido
    3) menor holgura del aula
    4) menor slot
    5) menor room
    """
    candidate = solution.copy()
    candidate.assign_exam(exam_id, slot, room)

    loads = _slot_loads(candidate)
    exam_size = _exam_size(candidate.instance, exam_id)
    slack = _room_capacity(candidate.instance, room) - exam_size
    partial_soft = float(soft_penalty(candidate, config=config.objective_config))

    key = (
        partial_soft,
        loads[slot],
        slack,
        slot,
        room,
    )
    return key, candidate


def select_best_assignment(
    solution: ExamSolution,
    exam_id: int,
    config: ConstructiveConfig,
) -> Optional[tuple[int, int]]:
    """
    Selecciona la mejor asignación factible (slot, room) para un examen.

    Si no existe ninguna, devuelve None.
    """
    best_choice: Optional[tuple[int, int]] = None
    best_key: Optional[tuple[float, int, int, int, int]] = None

    for slot, room in iter_candidate_assignments(solution, exam_id, config):
        if not can_assign_exam(solution, exam_id, slot, room):
            continue

        if not config.evaluate_all_feasible_candidates:
            return slot, room

        key, _ = _evaluate_candidate(solution, exam_id, slot, room, config)

        if best_key is None or key < best_key:
            best_key = key
            best_choice = (slot, room)

    return best_choice


# ==============================================================================

# CONSTRUCCIÓN PRINCIPAL =======================================================


def construct_initial_solution(
    instance: ExamInstance,
    config: ConstructiveConfig = ConstructiveConfig(),
) -> ConstructiveResult:
    """
    Construye una solución inicial examen a examen.

    Estrategia:
    1) ordenar exámenes por dificultad de colocación
    2) para cada examen, generar candidatos (slot, room)
    3) filtrar con can_assign_exam(...)
    4) elegir el mejor candidato factible

    Convención:
    - Solo se aceptan asignaciones factibles.
    - Si un examen no puede asignarse con esta estrategia greedy,
      queda sin asignar.
    """
    solution = ExamSolution(instance)
    ordered_exams = _sort_exams(instance, config)

    for exam_id in ordered_exams:
        best_assignment = select_best_assignment(solution, exam_id, config)

        if best_assignment is None:
            continue

        slot, room = best_assignment
        solution.assign_exam(exam_id, slot, room)

    unassigned_exams = [
        exam_id
        for exam_id in range(instance.n_exams)
        if not solution.is_assigned(exam_id)
    ]

    final_objective = float(objective_value(solution, config=config.objective_config))
    final_soft = float(soft_penalty(solution, config=config.objective_config))
    breakdown = objective_breakdown(solution, config=config.objective_config)

    return ConstructiveResult(
        solution=solution,
        assigned_exams=solution.num_assigned(),
        unassigned_exams=unassigned_exams,
        objective=final_objective,
        soft_penalty=final_soft,
        is_complete=solution.is_complete(),
        breakdown=breakdown,
    )


# ==============================================================================

# UTILIDADES DE APOYO / PRESENTACIÓN ===========================================


def summarize_constructive_result(result: ConstructiveResult) -> str:
    """
    Devuelve un resumen legible del resultado constructivo.
    """
    return (
        "\n=== RESULTADO HEURÍSTICA CONSTRUCTIVA ===\n"
        f"Exámenes asignados:    {result.assigned_exams}\n"
        f"Exámenes sin asignar:  {len(result.unassigned_exams)}\n"
        f"Solución completa:     {result.is_complete}\n"
        f"Penalización blanda:   {result.soft_penalty:.3f}\n"
        f"Valor objetivo total:  {result.objective:.3f}\n"
    )


def get_unassigned_exams(solution: ExamSolution) -> List[int]:
    """
    Devuelve la lista de exámenes no asignados.
    """
    return [
        exam_id
        for exam_id in range(solution.instance.n_exams)
        if not solution.is_assigned(exam_id)
    ]
