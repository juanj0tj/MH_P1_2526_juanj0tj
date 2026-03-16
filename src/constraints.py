from __future__ import annotations

from typing import Dict

from solution import ExamSolution, UNASSIGNED


# FUNCIONES AUXILIARES =========================================================


def is_exam_assigned(solution: ExamSolution, exam_id: int) -> bool:
    """
    Devuelve True si el examen tiene slot y aula asignados
    """
    return solution.is_assigned(exam_id)


def has_valid_slot(solution: ExamSolution, exam_id: int) -> bool:
    """
    Comprueba que el slot asignado al examen esté dentro del rango válido.
    """
    if not solution._valid_exam_id(exam_id):
        return False

    slot = int(solution.slot[exam_id])
    return 0 <= slot < solution.instance.n_slots


def has_valid_room(solution: ExamSolution, exam_id: int) -> bool:
    """
    Comprueba que el aula asignada al examen esté dentro del rango válido.
    """
    if not solution._valid_exam_id(exam_id):
        return False

    room = int(solution.room[exam_id])
    return 0 <= room < solution.instance.n_rooms


def has_consistent_assigment(solution: ExamSolution, exam_id: int) -> bool:
    """
    Comprueba consistencia estructural:
    - o bien slot y room están ambos asignados (True)
    - o bien ambos están sin asignar (False)
    """
    if not solution._valid_exam_id(exam_id):
        return False

    slot_assigned = solution.slot[exam_id] != UNASSIGNED
    room_assigned = solution.room[exam_id] != UNASSIGNED
    return slot_assigned == room_assigned


# ==============================================================================

# RESTRICCIONES DURAS ==========================================================


def check_student_conflict(solution: ExamSolution, exam_id: int) -> bool:
    """
    Restricción dura:
    - Ningún estudiante puede tener dos exámenes en la misma franja horaria.

    Comprueba si el examen exam_id entra en conflicto con algún otro examen ya
    programado en el mismo slot.
    """
    if not is_exam_assigned(solution, exam_id):
        return False
    if not has_valid_slot(solution, exam_id):
        return False

    slot = int(solution.slot[exam_id])
    n_exams = solution.instance.n_exams
    conflict_row = solution.instance.conflict_matrix[exam_id]

    for other_exam in range(n_exams):
        if other_exam == exam_id:
            continue
        if solution.slot[other_exam] == UNASSIGNED:
            continue
        if int(solution.slot[other_exam]) != slot:
            continue
        if conflict_row[other_exam]:
            return False

    return True


def check_room_capacity(solution: ExamSolution, exam_id: int) -> bool:
    """
    Restricción dura:
    - El número de estudiantes de un examen no puede exceder la
        capacidad del aula.
    """
    if not is_exam_assigned(solution, exam_id):
        return False
    if not has_valid_room(solution, exam_id):
        return False

    room_id = int(solution.room[exam_id])
    exam_size = int(solution.instance.exam_sizes[exam_id])
    room_capacity = int(solution.instance.room_capacities[room_id])

    return exam_size <= room_capacity


def check_unique_assigment(solution: ExamSolution, exam_id: int) -> bool:
    """
    Restricción dura:
    - Cada examen debe programarse exactamente una vez.

    Dada la representación actual, esto equivale a exigir que el examen esté
    completamente asignado (slot y aula). No puede aparecer duplicado porque
    cada examen ocupa una única posición en los vectores slot y room.
    """
    if not solution._valid_exam_id(exam_id):
        return False

    return (
        has_consistent_assigment(solution, exam_id)
        and is_exam_assigned(solution, exam_id)
        and has_valid_slot(solution, exam_id)
        and has_valid_room(solution, exam_id)
    )


def check_exam_feasible(solution: ExamSolution, exam_id: int) -> bool:
    """
    Comprueba si un examen cumple todas sus restricciones duras.
    """
    return (
        check_unique_assigment(solution, exam_id)
        and check_room_capacity(solution, exam_id)
        and check_student_conflict(solution, exam_id)
    )


# ==============================================================================


# VALIDACIÓN DE MOVIMIENTOS ====================================================


def can_assign_exam(solution: ExamSolution, exam_id: int, slot: int, room: int) -> bool:
    """
    Comprueba si asignar (slot, room) al examen produce una situación factible
    para ese examen.
    """
    if not solution._valid_exam_id(exam_id):
        return False
    if not (0 <= slot < solution.instance.n_slots):
        return False
    if not (0 <= room < solution.instance.n_rooms):
        return False

    candidate = solution.copy()
    candidate.assign_exam(exam_id, slot, room)
    return check_exam_feasible(candidate, exam_id)


def can_change_slot(solution: ExamSolution, exam_id: int, new_slot: int) -> bool:
    """
    Comprueba si cambiar el slot de un examen mantiene el cumplimiento de las
    restricciones duras.

    Se asume que el examen ya tiene aula asignada.
    """
    if not solution._valid_exam_id(exam_id):
        return False
    if not is_exam_assigned(solution, exam_id):
        return False
    if not (0 <= new_slot < solution.instance.n_slots):
        return False

    candidate = solution.copy()
    candidate.assign_slot(exam_id, new_slot)
    return check_exam_feasible(candidate, exam_id)


def can_swap_slots(solution: ExamSolution, exam_id_1: int, exam_id_2: int) -> bool:
    """
    Comprueba si intercambiar los slots de dos exámenes mantiene el cumplimiento
    de las restricciones duras.
    """
    if not solution._valid_exam_id(exam_id_1):
        return False
    if not solution._valid_exam_id(exam_id_2):
        return False
    if exam_id_1 == exam_id_2:
        return True
    if not is_exam_assigned(solution, exam_id_1):
        return False
    if not is_exam_assigned(solution, exam_id_2):
        return False

    candidate = solution.copy()
    candidate.swap_slots(exam_id_1, exam_id_2)

    return check_exam_feasible(candidate, exam_id_1) and check_exam_feasible(
        candidate, exam_id_2
    )


# ==============================================================================

# RECUENTO GLOBAL DE VIOLACIONES ===============================================


def count_unassigned_exams(solution: ExamSolution) -> int:
    """
    Cuenta exámenes que no están programados exactamente una vez según
    la representación actual.
    """
    count = 0

    for exam_id in range(solution.instance.n_exams):
        if not check_unique_assigment(solution, exam_id):
            count += 1

    return count


def count_capacity_violations(solution: ExamSolution) -> int:
    """
    Cuenta cuántos exámenes asignados incumplen la capacidad del aula.
    """
    count = 0

    for exam_id in range(solution.instance.n_exams):
        if is_exam_assigned(solution, exam_id) and not check_room_capacity(
            solution, exam_id
        ):
            count += 1

    return count


def count_student_conflicts(solution: ExamSolution) -> int:
    """
    Cuenta conflictos de solapamiento por pares de exámenes.
    Cada par en conflicto se cuenta una sola vez.
    """
    count = 0
    n_exams = solution.instance.n_exams
    conflict_matrix = solution.instance.conflict_matrix

    for e1 in range(n_exams):
        slot1 = int(solution.slot[e1])
        if slot1 == UNASSIGNED:
            continue

        for e2 in range(e1 + 1, n_exams):
            slot2 = int(solution.slot[e2])
            if slot2 == UNASSIGNED:
                continue

            if slot1 == slot2 and conflict_matrix[e1, e2]:
                count += 1

    return count


def count_hard_violations(solution: ExamSolution) -> int:
    """
    Suma total de violaciones de las restricciones duras.
    """
    return (
        count_unassigned_exams(solution)
        + count_capacity_violations(solution)
        + count_student_conflicts(solution)
    )


# ==============================================================================

# DIAGNÓSTICO ==================================================================


def hard_violations_breakdown(solution: ExamSolution) -> Dict[str, int]:
    """
    Devuelve el desglose de violaciones duras.
    """
    student_conflicts = count_student_conflicts(solution)
    capacity = count_capacity_violations(solution)
    unassigned = count_unassigned_exams(solution)

    return {
        "student_conflicts": student_conflicts,
        "capacity": capacity,
        "unassigned": unassigned,
        "total": student_conflicts + capacity + unassigned,
    }
