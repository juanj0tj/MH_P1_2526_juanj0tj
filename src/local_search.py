from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Literal, Optional

from objective import ObjectiveConfig, objective_value
from solution import ExamSolution
from neighborhood import NeighborhoodMove, apply_move, iter_moves


# ==============================================================================

# TIPOS Y RESULTADOS ===========================================================


SearchAlgorithm = Literal["first_improvement", "best_improvement"]


@dataclass(frozen=True)
class LocalSearchHistoryPoint:
    """
    Punto de la historia de búsqueda.

    Convención:
    - representa un estado observable del algoritmo
    - se usa para construir traces.csv y gráficas
    """

    evaluations: int
    iterations: int
    accepted_moves: int
    objective: float


@dataclass(frozen=True)
class LocalSearchResult:
    """
    Resultado de una búsqueda local.

    Campos principales:
    - algorithm: variante empleada.
    - solution: solución final devuelta por la búsqueda.
    - initial_objective: valor f.objetivo de la solución inicial.
    - final_objective: valor f.objetivo de la solución final.
    - evaluations: número total de evaluaciones de objective_value realizadas.
    - iterations: número de iteraciones de búsqueda local.
    - accepted_moves: número de movimientos aceptados.
    - elapsed_time: tiempo total en segundos.
    - history: historia opcional para trazas.
    """

    algorithm: SearchAlgorithm
    solution: ExamSolution
    initial_objective: float
    final_objective: float
    evaluations: int
    iterations: int
    accepted_moves: int
    elapsed_time: float
    history: Optional[list[LocalSearchHistoryPoint]] = None

    @property
    def improvement(self) -> float:
        """
        Mejora absoluta conseguida:
        initial_objective - final_objective
        """
        return self.initial_objective - self.final_objective


# ==============================================================================

# UTILIDADES INTERNAS ==========================================================


def _check_search_arguments(
    max_evaluations: Optional[int],
    max_iterations: Optional[int],
    include_change_slot: bool,
    include_swap_slots: bool,
) -> None:
    """
    Valida los argumentos comunes de búsqueda.
    """
    if max_evaluations is not None and max_evaluations <= 0:
        raise ValueError("max_evaluations debe ser > 0 o None.")

    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations debe ser > 0 o None.")

    if not include_change_slot and not include_swap_slots:
        raise ValueError(
            "Debe activarse al menos un operador de vecindario "
            "(change_slot o swap_slots)."
        )


def _iteration_limit_reached(iterations: int, max_iterations: Optional[int]) -> bool:
    """
    True si ya se alcanzó el límite de iteraciones.
    """
    return max_iterations is not None and iterations >= max_iterations


def _evaluation_limit_reached(evaluations: int, max_evaluations: Optional[int]) -> bool:
    """
    True si ya se alcanzó el límite de evaluaciones.
    """
    return max_evaluations is not None and evaluations >= max_evaluations


def _make_history_point(
    evaluations: int,
    iterations: int,
    accepted_moves: int,
    objective: float,
) -> LocalSearchHistoryPoint:
    return LocalSearchHistoryPoint(
        evaluations=int(evaluations),
        iterations=int(iterations),
        accepted_moves=int(accepted_moves),
        objective=float(objective),
    )


def _append_history(
    history: Optional[list[LocalSearchHistoryPoint]],
    *,
    evaluations: int,
    iterations: int,
    accepted_moves: int,
    objective: float,
) -> None:
    """
    Añade un punto a la historia si procede.
    """
    if history is None:
        return

    history.append(
        _make_history_point(
            evaluations=evaluations,
            iterations=iterations,
            accepted_moves=accepted_moves,
            objective=objective,
        )
    )


def _ensure_terminal_history_point(
    history: Optional[list[LocalSearchHistoryPoint]],
    *,
    evaluations: int,
    iterations: int,
    accepted_moves: int,
    objective: float,
) -> None:
    """
    Garantiza que el último punto de la historia coincide exactamente con el
    estado final reportado por LocalSearchResult.

    Esta función corrige la discrepancia entre:
    - summary.csv
    - y traces.csv

    cuando la última exploración consume evaluaciones pero no acepta mejora.
    """
    if history is None:
        return

    terminal = _make_history_point(
        evaluations=evaluations,
        iterations=iterations,
        accepted_moves=accepted_moves,
        objective=objective,
    )

    if not history:
        history.append(terminal)
        return

    last = history[-1]
    if (
        last.evaluations != terminal.evaluations
        or last.iterations != terminal.iterations
        or last.accepted_moves != terminal.accepted_moves
        or last.objective != terminal.objective
    ):
        history.append(terminal)


# ==============================================================================

# PRIMER MEJOR =================================================================


def first_improvement_search(
    initial_solution: ExamSolution,
    config: ObjectiveConfig = ObjectiveConfig(),
    *,
    max_evaluations: Optional[int] = None,
    max_iterations: Optional[int] = None,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    seed: Optional[int] = None,
    track_history: bool = False,
) -> LocalSearchResult:
    """
    Búsqueda local de primer mejor.

    Estrategia:
    - Parte de la solución inicial.
    - Explora vecinos en orden aleatorio.
    - En cuanto encuentra el primer vecino que mejora, se mueve a él.
    - Termina cuando:
        - no existe mejora en el vecindario explorado, o
        - se alcanza max_evaluations, o
        - se alcanza max_iterations.

    Convenciones:
    - Se trabaja sobre una copia de la solución inicial.
    - El recuento de evaluaciones incluye la evaluación inicial.
    - Si track_history=True, se guarda:
        - el estado inicial,
        - cada mejora aceptada,
        - y un punto terminal final.
    """

    _check_search_arguments(
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        include_change_slot=include_change_slot,
        include_swap_slots=include_swap_slots,
    )

    rng = Random(seed)
    current = initial_solution.copy()

    start_time = perf_counter()

    current_value = objective_value(current, config=config)
    initial_value = current_value
    evaluations = 1
    iterations = 0
    accepted_moves = 0

    history: Optional[list[LocalSearchHistoryPoint]] = [] if track_history else None
    _append_history(
        history,
        evaluations=evaluations,
        iterations=iterations,
        accepted_moves=accepted_moves,
        objective=current_value,
    )

    while True:
        if _iteration_limit_reached(iterations, max_iterations):
            break
        if _evaluation_limit_reached(evaluations, max_evaluations):
            break

        iterations += 1
        improved = False

        for move in iter_moves(
            current,
            include_change_slot=include_change_slot,
            include_swap_slots=include_swap_slots,
            randomize=True,
            rng=rng,
        ):
            if _evaluation_limit_reached(evaluations, max_evaluations):
                break

            neighbor = apply_move(current, move)
            neighbor_value = objective_value(neighbor, config=config)
            evaluations += 1

            if neighbor_value < current_value:
                current = neighbor
                current_value = neighbor_value
                accepted_moves += 1
                improved = True

                _append_history(
                    history,
                    evaluations=evaluations,
                    iterations=iterations,
                    accepted_moves=accepted_moves,
                    objective=current_value,
                )
                break

        if not improved:
            break

    elapsed_time = perf_counter() - start_time

    _ensure_terminal_history_point(
        history,
        evaluations=evaluations,
        iterations=iterations,
        accepted_moves=accepted_moves,
        objective=current_value,
    )

    return LocalSearchResult(
        algorithm="first_improvement",
        solution=current,
        initial_objective=float(initial_value),
        final_objective=float(current_value),
        evaluations=evaluations,
        iterations=iterations,
        accepted_moves=accepted_moves,
        elapsed_time=elapsed_time,
        history=history,
    )


# ==============================================================================

# MEJOR ========================================================================


def best_improvement_search(
    initial_solution: ExamSolution,
    config: ObjectiveConfig = ObjectiveConfig(),
    *,
    max_evaluations: Optional[int] = None,
    max_iterations: Optional[int] = None,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    track_history: bool = False,
) -> LocalSearchResult:
    """
    Búsqueda local del mejor.

    Estrategia:
    - Parte de la solución inicial.
    - Evalúa todo el vecindario factible.
    - Selecciona el mejor vecino que mejore a la solución actual.
    - Termina cuando:
        - no existe mejora en todo el vecindario, o
        - se alcanza max_evaluations, o
        - se alcanza max_iterations.

    Convenciones:
    - Se trabaja sobre una copia de la solución inicial.
    - El recuento de evaluaciones incluye la evaluación inicial.
    - Si track_history=True, se guarda:
        - el estado inicial,
        - cada mejora aceptada,
        - y un punto terminal final.
    """
    _check_search_arguments(
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        include_change_slot=include_change_slot,
        include_swap_slots=include_swap_slots,
    )

    current = initial_solution.copy()

    start_time = perf_counter()

    current_value = objective_value(current, config=config)
    initial_value = current_value
    evaluations = 1
    iterations = 0
    accepted_moves = 0

    history: Optional[list[LocalSearchHistoryPoint]] = [] if track_history else None
    _append_history(
        history,
        evaluations=evaluations,
        iterations=iterations,
        accepted_moves=accepted_moves,
        objective=current_value,
    )

    while True:
        if _iteration_limit_reached(iterations, max_iterations):
            break
        if _evaluation_limit_reached(evaluations, max_evaluations):
            break

        iterations += 1

        best_neighbor: Optional[ExamSolution] = None
        best_move: Optional[NeighborhoodMove] = None
        best_value: Optional[float] = None

        for move in iter_moves(
            current,
            include_change_slot=include_change_slot,
            include_swap_slots=include_swap_slots,
            randomize=False,
            rng=None,
        ):
            if _evaluation_limit_reached(evaluations, max_evaluations):
                break

            neighbor = apply_move(current, move)
            neighbor_value = objective_value(neighbor, config=config)
            evaluations += 1

            if neighbor_value < current_value:
                if best_value is None or neighbor_value < best_value:
                    best_neighbor = neighbor
                    best_move = move
                    best_value = neighbor_value

        if best_neighbor is None or best_value is None or best_move is None:
            break

        current = best_neighbor
        current_value = best_value
        accepted_moves += 1

        _append_history(
            history,
            evaluations=evaluations,
            iterations=iterations,
            accepted_moves=accepted_moves,
            objective=current_value,
        )

    elapsed_time = perf_counter() - start_time

    _ensure_terminal_history_point(
        history,
        evaluations=evaluations,
        iterations=iterations,
        accepted_moves=accepted_moves,
        objective=current_value,
    )

    return LocalSearchResult(
        algorithm="best_improvement",
        solution=current,
        initial_objective=float(initial_value),
        final_objective=float(current_value),
        evaluations=evaluations,
        iterations=iterations,
        accepted_moves=accepted_moves,
        elapsed_time=elapsed_time,
        history=history,
    )


# ==============================================================================

# INTERFAZ UNIFICADA ===========================================================


def local_search(
    initial_solution: ExamSolution,
    algorithm: SearchAlgorithm,
    config: ObjectiveConfig = ObjectiveConfig(),
    *,
    max_evaluations: Optional[int] = None,
    max_iterations: Optional[int] = None,
    include_change_slot: bool = True,
    include_swap_slots: bool = True,
    seed: Optional[int] = None,
    track_history: bool = False,
) -> LocalSearchResult:
    """
    Interfaz unificada para lanzar la búsqueda local.

    Parámetros:
    - algorithm:
        - "first_improvement"
        - "best_improvement"
    """

    if algorithm == "first_improvement":
        return first_improvement_search(
            initial_solution=initial_solution,
            config=config,
            max_evaluations=max_evaluations,
            max_iterations=max_iterations,
            include_change_slot=include_change_slot,
            include_swap_slots=include_swap_slots,
            seed=seed,
            track_history=track_history,
        )

    if algorithm == "best_improvement":
        return best_improvement_search(
            initial_solution=initial_solution,
            config=config,
            max_evaluations=max_evaluations,
            max_iterations=max_iterations,
            include_change_slot=include_change_slot,
            include_swap_slots=include_swap_slots,
            track_history=track_history,
        )

    raise ValueError(f"Algoritmo no soportado: {algorithm}")
