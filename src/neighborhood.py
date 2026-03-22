from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterator, Literal, Optional

from constraints import can_change_slot, can_swap_slots
from objective import ObjectiveConfig, objective_value
from solution import ExamSolution


# TIPOS Y REPRESENTACIÓN DE MOVIMIENTOS ========================================


MoveType = Literal["change_slot", "swap_slots"]


@dataclass(frozen=True)
class NeighborhoodMove:
    """
    Representa un movimiento del vecindario.

    Convenciones:
    - kind == "change_slot"
        - exam_id_1 = examen a mover.
        - new_slot = nueva franja.
        - exam_id_2 = None

    - kind == "swap_slots":
        - exam_id_1 y exam_id_2 = exámenes cuyas franjas se intercambiarán.
        - new_slot = None
    """

    kind: MoveType
    exam_id_1: int
    exam_id_2: Optional[int] = None
    new_slot: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == "change_slot":
            if self.new_slot is None:
                raise ValueError("Un movimiento 'change_slot' requiere new_slot.")
            if self.exam_id_2 is not None:
                raise ValueError(
                    "Un movimiento 'change_slot' no debe incluir exam_id_2."
                )

        elif self.kind == "swap_slots":
            if self.exam_id_2 is None:
                raise ValueError("Un movimiento 'swap_slots' requiere exam_id_2.")
            if self.new_slot is not None:
                raise ValueError("Un movimiento 'swap_slots' no debe incluir new_slot.")

        else:
            raise ValueError(f"Tipo de movimiento no soportado: {self.kind}")


@dataclass(frozen=True)
class EvaluatedMove:
    """
    Movimiento junto con la evaluación de la solución vecina resultante.
    """

    move: NeighborhoodMove
    value: float


# ==============================================================================

# APLICACIÓN Y VALIDACIÓN DE MOVIMIENTOS =======================================


def is_move_feasible(solution: ExamSolution, move: NeighborhoodMove) -> bool:
    """
    Comprueba si un movimiento genera un vecino factible con respecto a
    las restricciones duras.

    Nota:
    - Esta función es auxiliar.
    - El vecindario NO se limita a movimientos factibles.
    """

    if move.kind == "change_slot":
        assert move.new_slot is not None
        return can_change_slot(solution, move.exam_id_1, move.new_slot)

    if move.kind == "swap_slots":
        assert move.exam_id_2 is not None
        return can_swap_slots(solution, move.exam_id_1, move.exam_id_2)

    return False


def apply_move(solution: ExamSolution, move: NeighborhoodMove) -> ExamSolution:
    """
    Aplica un movimiento sobre una COPIA de la solución y devuelve el vecino.

    Importante:
    - No filtra por factibilidad respecto a restricciones duras.
    - Solo exige que el movimiento sea estructuralmente válido para la
      representación de la solución.
    """

    neighbor = solution.copy()

    if move.kind == "change_slot":
        assert move.new_slot is not None
        neighbor.assign_slot(move.exam_id_1, move.new_slot)
        return neighbor

    if move.kind == "swap_slots":
        assert move.exam_id_2 is not None
        neighbor.swap_slots(move.exam_id_1, move.exam_id_2)
        return neighbor

    raise ValueError(f"Tipo de movimiento no soportado: {move.kind}")


# ==============================================================================

# UTILIDADES DE ORDENACIÓN / ALEATORIZACIÓN ====================================


def _maybe_shuffle(
    items: list[int], randomize: bool, rng: Optional[Random]
) -> list[int]:
    """
    Devuelve una copia de items, barajada opcionalmente.
    """
    output = items.copy()
    if randomize:
        generator = rng if rng is not None else Random()
        generator.shuffle(output)
    return output


# ==============================================================================

# GENERACIÓN DE MOVIMIENTOS: CAMBIO DE SLOT ====================================


def iter_change_slot_moves(
    solution: ExamSolution,
    *,
    randomize: bool = False,
    rng: Optional[Random] = None,
) -> Iterator[NeighborhoodMove]:
    """
    Genera movimientos de tipo 'change_slot'.

    Para cada examen asignado, intenta moverlo a otra franja distinta
    conservando su aula actual.

    Estos movimientos pueden producir vecinos factibles o no factibles.
    La decisión final la toma la función objetivo penalizada.

    Orden:
    - Determinista por defecto.
    - Opcionalmente aleatorio para usar en primer mejor.
    """

    exam_ids = list(range(solution.instance.n_exams))
    exam_ids = _maybe_shuffle(exam_ids, randomize, rng)

    for exam_id in exam_ids:
        if not solution.is_assigned(exam_id):
            continue

        current_slot = int(solution.slot[exam_id])

        slot_ids = list(range(solution.instance.n_slots))
        slot_ids = _maybe_shuffle(slot_ids, randomize, rng)

        for new_slot in slot_ids:
            if new_slot == current_slot:
                continue

            yield NeighborhoodMove(
                kind="change_slot",
                exam_id_1=exam_id,
                new_slot=new_slot,
            )


# ==============================================================================

# GENERACIÓN DE MOVIMIENTOS: INTERCAMBIO DE SLOTS ==============================


def iter_swap_slot_moves(
    solution: ExamSolution,
    *,
    randomize: bool = False,
    rng: Optional[Random] = None,
) -> Iterator[NeighborhoodMove]:
    """
    Genera movimientos de tipo 'swap_slots'.

    Recorre pares (e1, e2) con e1 < e2 para evitar duplicados.
    No genera swaps triviales entre exámenes ya colocados en el mismo slot.

    Estos movimientos pueden producir vecinos factibles o no factibles.
    La decisión final la toma la función objetivo penalizada.
    """

    exam_ids = list(range(solution.instance.n_exams))
    exam_ids = _maybe_shuffle(exam_ids, randomize, rng)

    n = len(exam_ids)

    for i in range(n):
        e1 = exam_ids[i]
        if not solution.is_assigned(e1):
            continue

        for j in range(i + 1, n):
            e2 = exam_ids[j]
            if not solution.is_assigned(e2):
                continue

            if int(solution.slot[e1]) == int(solution.slot[e2]):
                continue

            yield NeighborhoodMove(
                kind="swap_slots",
                exam_id_1=e1,
                exam_id_2=e2,
            )


# ==============================================================================

# GENERACIÓN DEL VECINDARIO COMPLETO ===========================================


def iter_moves(
    solution: ExamSolution,
    *,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    randomize: bool = False,
    rng: Optional[Random] = None,
) -> Iterator[NeighborhoodMove]:
    """
    Itera sobre el vecindario de la solución.

    El vecindario puede incluir movimientos que produzcan vecinos
    no factibles con respecto a restricciones duras.

    El orden puede hacerse aleatorio, útil especialmente en estrategias
    de primer mejor.
    """

    if include_change_slot:
        yield from iter_change_slot_moves(solution, randomize=randomize, rng=rng)

    if include_swap_slots:
        yield from iter_swap_slot_moves(solution, randomize=randomize, rng=rng)


def iter_neighbors(
    solution: ExamSolution,
    *,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    randomize: bool = False,
    rng: Optional[Random] = None,
) -> Iterator[ExamSolution]:
    """
    Itera sobre las soluciones vecinas generadas por los movimientos del
    vecindario definido.
    """

    for move in iter_moves(
        solution,
        include_change_slot=include_change_slot,
        include_swap_slots=include_swap_slots,
        randomize=randomize,
        rng=rng,
    ):
        yield apply_move(solution, move)


# ==============================================================================

# EVALUACIÓN DE MOVIMIENTOS / VECINOS ==========================================


def evaluate_move(
    solution: ExamSolution,
    move: NeighborhoodMove,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> float:
    """
    Evalúa la solución vecina que resulta de aplicar un movimiento.
    """
    neighbor = apply_move(solution, move)
    return objective_value(neighbor, config=config)


def iter_evaluated_moves(
    solution: ExamSolution,
    config: ObjectiveConfig = ObjectiveConfig(),
    *,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    randomize: bool = False,
    rng: Optional[Random] = None,
) -> Iterator[EvaluatedMove]:
    """
    Itera sobre movimientos junto con el valor objetivo de su vecino.
    """

    for move in iter_moves(
        solution,
        include_change_slot=include_change_slot,
        include_swap_slots=include_swap_slots,
        randomize=randomize,
        rng=rng,
    ):
        yield EvaluatedMove(
            move=move,
            value=evaluate_move(solution, move, config=config),
        )


# ==============================================================================

# SELECCIÓN DE MEJORAS =========================================================


def first_improving_move(
    solution: ExamSolution,
    current_value: float,
    config: ObjectiveConfig = ObjectiveConfig(),
    *,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    randomize: bool = True,
    rng: Optional[Random] = None,
) -> Optional[EvaluatedMove]:
    """
    Devuelve el primer movimiento que mejora el valor objetivo actual,
    o None si no existe.
    """

    for evaluated in iter_evaluated_moves(
        solution,
        config=config,
        include_change_slot=include_change_slot,
        include_swap_slots=include_swap_slots,
        randomize=randomize,
        rng=rng,
    ):
        if evaluated.value < current_value:
            return evaluated

    return None


def best_improving_move(
    solution: ExamSolution,
    current_value: float,
    config: ObjectiveConfig = ObjectiveConfig(),
    *,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
) -> Optional[EvaluatedMove]:
    """
    Devuelve el mejor movimiento que mejora el valor objetivo actual,
    o None si no existe.
    """

    best: Optional[EvaluatedMove] = None

    for evaluated in iter_evaluated_moves(
        solution,
        config=config,
        include_change_slot=include_change_slot,
        include_swap_slots=include_swap_slots,
        randomize=False,
        rng=None,
    ):
        if evaluated.value < current_value:
            if best is None or evaluated.value < best.value:
                best = evaluated

    return best


# ==============================================================================

# UTILIDADES DE CONVENIENCIA ===================================================


def get_all_moves(
    solution: ExamSolution,
    *,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    randomize: bool = False,
    rng: Optional[Random] = None,
) -> list[NeighborhoodMove]:
    """
    Devuelve todos los movimientos del vecindario definido.
    """
    return list(
        iter_moves(
            solution,
            include_change_slot=include_change_slot,
            include_swap_slots=include_swap_slots,
            randomize=randomize,
            rng=rng,
        )
    )


def get_all_neighbors(
    solution: ExamSolution,
    *,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    randomize: bool = False,
    rng: Optional[Random] = None,
) -> list[ExamSolution]:
    """
    Devuelve todos los vecinos generados por el vecindario definido.
    """
    return list(
        iter_neighbors(
            solution,
            include_change_slot=include_change_slot,
            include_swap_slots=include_swap_slots,
            randomize=randomize,
            rng=rng,
        )
    )
