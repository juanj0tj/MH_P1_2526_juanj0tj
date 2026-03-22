from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import List, Optional

from instance_manager import ExamInstance
from objective import (
    ObjectiveConfig,
    objective_breakdown,
    objective_value,
    soft_penalty,
)
from solution import ExamSolution


# ==============================================================================

# CONFIGURACIÓN Y RESULTADOS ===================================================


@dataclass(frozen=True)
class ConstructiveConfig:
    """
    Parámetros de la heurística constructiva aleatoria.

    Convención adoptada:
    - La solución inicial se construye de forma totalmente aleatoria.
    - No se tienen en cuenta restricciones duras ni blandas durante la asignación.
    - La evaluación se realiza únicamente al final.
    """

    objective_config: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    seed: Optional[int] = None
    shuffle_exams: bool = True


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

# CONSTRUCCIÓN PRINCIPAL =======================================================


def construct_initial_solution(
    instance: ExamInstance,
    config: ConstructiveConfig = ConstructiveConfig(),
) -> ConstructiveResult:
    """
    Construye una solución inicial totalmente aleatoria.

    Estrategia:
    - Recorre todos los exámenes.
    - Asigna a cada examen una franja y un aula elegidas al azar.
    - Ignora restricciones duras y blandas durante la construcción.

    Observación:
    - La solución resultante puede ser inviable.
    - La función objetivo final se evalúa al terminar la construcción.
    """
    rng = Random(config.seed)
    solution = ExamSolution(instance)

    exam_ids = list(range(instance.n_exams))
    if config.shuffle_exams:
        rng.shuffle(exam_ids)

    for exam_id in exam_ids:
        slot = rng.randrange(instance.n_slots)
        room = rng.randrange(instance.n_rooms)
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
