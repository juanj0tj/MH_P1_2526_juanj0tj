from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from solution import ExamSolution, UNASSIGNED
from constraints import count_hard_violations, hard_violations_breakdown


# CONFIGURACIÓN DE LA FUNCIÓN OBJETIVO =========================================


@dataclass(frozen=True)
class ObjectiveConfig:
    """
    Parámetros de la función objetivo.

    soft_penalty =
        w_consecutive * Pconsecutivos +
        w_same_day * Pmismo_dia +
        w_distribution * Pdistribucion

    total_objective =
        hard_penalty * hard_violations + soft_penalty
    """

    hard_penalty: float = 1_000_000.0
    w_consecutive: float = 5.0
    w_same_day: float = 2.0
    w_distribution: float = 1.0
    slots_per_day: int = 4


# ==============================================================================

# FUNCIONES AUXILIARES =========================================================


def slot_to_day(slot: int, slots_per_day: int) -> int:
    """
    Convierte una franja horaria a índice de día.

    Ejemplo:
    - slots_per_day = 4
    - slots 0,1,2,3 -> día 0
    - slots 4,5,6,7 -> día 1
    """
    if slots_per_day <= 0:
        raise ValueError("slots_per_day debe ser > 0")
    return slot // slots_per_day


def is_slot_assigned(solution: ExamSolution, exam_id: int) -> bool:
    """
    True si el examen tiene un slot válido asignado.
    """
    if not solution._valid_exam_id(exam_id):
        return False
    return int(solution.slot[exam_id]) != UNASSIGNED


def assigned_exams(solution: ExamSolution) -> np.ndarray:
    """
    Devuelve los ids de exámenes con slot y aula asignados.
    """
    return np.where((solution.slot != UNASSIGNED) & (solution.room != UNASSIGNED))[0]


# ==============================================================================

# PENALIZACIONES BLANDAS ======================================================='


def consecutive_penalty(
    solution: ExamSolution,
    slots_per_day: int = 4,
) -> int:
    """
    Pconsecutivos:
    Penaliza pares de exámenes consecutivos para un mismo estudiante.

    Convención adoptada:
    - Se cuenta un par (e1, e2) si ambos están asignados
    - abs(slot[e1] - slot[e2]) == 1
    - y ambos slots pertenecen al mismo día

    Cada par se cuenta una sola vez por estudiante.
    """
    penalty = 0

    for student_exams in solution.instance.exams_by_student.values():
        assigned_slots = []

        for exam_id in student_exams:
            if solution.is_assigned(exam_id):
                assigned_slots.append(int(solution.slot[exam_id]))

        m = len(assigned_slots)
        if m < 2:
            continue

        for i in range(m):
            for j in range(i + 1, m):
                s1 = assigned_slots[i]
                s2 = assigned_slots[j]

                if abs(s1 - s2) == 1:
                    if slot_to_day(s1, slots_per_day) == slot_to_day(s2, slots_per_day):
                        penalty += 1

    return penalty


def same_day_penalty(
    solution: ExamSolution,
    slots_per_day: int = 4,
) -> int:
    """
    Pmismo_dia:
    Penaliza múltiples exlamenes en el mismo día para un mismo estudiante.

    Convención adoptada:
    - Para cada estudiante y cada día, si se tiene k exámenes ese día,
      añade C(k,2) = k*(k-1)//2

    De este modo:
    - 0 o 1 examen en el día -> 0
    - 2 exámenes en el día   -> 1
    - 3 exámenes en el día   -> 3
    - 4 exámenes en el día   -> 6
    """
    penalty = 0

    for student_exams in solution.instance.exams_by_student.values():
        exams_per_day: Dict[int, int] = {}

        for exam_id in student_exams:
            slot = int(solution.slot[exam_id])
            if slot == UNASSIGNED:
                continue

            day = slot_to_day(slot, slots_per_day)
            exams_per_day[day] = exams_per_day.get(day, 0) + 1

        for k in exams_per_day.values():
            if k >= 2:
                penalty += k * (k - 1) // 2

    return penalty


def distribution_penalty(solution: ExamSolution) -> float:
    """
    Pdistribucion:
    Penaliza el uso desequilibrado de franjas.

    Convención adoptada:
    - Sea load[s] el número de exámenes asignados al slot s
    - Sea media = total_examenes_asignados / n_slots
    - Penalización = sum((load[s] - media)^2)

    Esta medida favorece una ocupación más homogénea de las franjas.
    """
    n_slots = solution.instance.n_slots
    loads = np.zeros(n_slots, dtype=np.int32)

    for exam_id in range(solution.instance.n_exams):
        slot = int(solution.slot[exam_id])
        if slot != UNASSIGNED:
            loads[slot] += 1

    total_assigned = int(loads.sum())
    if total_assigned == 0:
        return 0.0

    mean_load = total_assigned / n_slots
    return float(np.sum((loads - mean_load) ** 2))


# ==============================================================================

# TOTAL PENALIZACIONES BLANDAS Y FUNCIÓN OBJETIVO ==============================


def soft_penalty(
    solution: ExamSolution,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> float:
    """
    Devuelve la penalización blanda total.
    """
    p_consecutive = consecutive_penalty(solution, slots_per_day=config.slots_per_day)
    p_same_day = same_day_penalty(solution, slots_per_day=config.slots_per_day)
    p_distribution = distribution_penalty(solution)

    return (
        config.w_consecutive * p_consecutive
        + config.w_same_day * p_same_day
        + config.w_distribution * p_distribution
    )


def objective_value(
    solution: ExamSolution,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> float:
    """
    Función objetivo total:
        hard_penalty * hard_violations + soft_penalty
    """
    hard_violations = count_hard_violations(solution)
    soft = soft_penalty(solution, config=config)

    return config.hard_penalty * hard_violations + soft


# ==============================================================================

# DIAGNÓSTICO ==================================================================


def soft_penalty_breakdown(
    solution: ExamSolution,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> Dict[str, float]:
    """
    Devuelve el desglose de penalizaciones blandas, tanto en bruto
    como ponderadas.
    """
    p_consecutive = consecutive_penalty(solution, slots_per_day=config.slots_per_day)
    p_same_day = same_day_penalty(solution, slots_per_day=config.slots_per_day)
    p_distribution = distribution_penalty(solution)

    weighted_consecutive = config.w_consecutive * p_consecutive
    weighted_same_day = config.w_same_day * p_same_day
    weighted_distribution = config.w_distribution * p_distribution

    total_soft = weighted_consecutive + weighted_same_day + weighted_distribution

    return {
        "p_consecutive": float(p_consecutive),
        "p_same_day": float(p_same_day),
        "p_distribution": float(p_distribution),
        "weighted_consecutive": float(weighted_consecutive),
        "weighted_same_day": float(weighted_same_day),
        "weighted_distribution": float(weighted_distribution),
        "total_soft": float(total_soft),
    }


def objective_breakdown(
    solution: ExamSolution,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> Dict[str, float]:
    """
    Devuelve un desglose completo de la evaluación.
    """
    hard_info = hard_violations_breakdown(solution)
    soft_info = soft_penalty_breakdown(solution, config=config)

    hard_violations = int(hard_info["total"])
    hard_cost = config.hard_penalty * hard_violations
    total = hard_cost + soft_info["total_soft"]

    return {
        "hard_penalty_weight": float(config.hard_penalty),
        "hard_violations": float(hard_violations),
        "hard_cost": float(hard_cost),
        "student_conflicts": float(hard_info["student_conflicts"]),
        "capacity_violations": float(hard_info["capacity"]),
        "unassigned_exams": float(hard_info["unassigned"]),
        "p_consecutive": soft_info["p_consecutive"],
        "p_same_day": soft_info["p_same_day"],
        "p_distribution": soft_info["p_distribution"],
        "weighted_consecutive": soft_info["weighted_consecutive"],
        "weighted_same_day": soft_info["weighted_same_day"],
        "weighted_distribution": soft_info["weighted_distribution"],
        "soft_penalty": soft_info["total_soft"],
        "objective": float(total),
    }


# ==============================================================================

# UTILIDADES ===================================================================


def is_feasible(solution: ExamSolution) -> bool:
    """
    True si la solución no tiene violaciones duras.
    """
    return count_hard_violations(solution) == 0


def compare_solutions(
    solution_a: ExamSolution,
    solution_b: ExamSolution,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> int:
    """
    Compara dos soluciones por su función objetivo total.

    Devuelve:
    -1 si a < b
     0 si a == b
     1 si a > b
    """
    value_a = objective_value(solution_a, config=config)
    value_b = objective_value(solution_b, config=config)

    if value_a < value_b:
        return -1
    if value_a > value_b:
        return 1
    return 0
