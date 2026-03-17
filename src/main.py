from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from instance_manager import (
    BASE_INSTANCES_DIR,
    ExamInstance,
    cargar_instancia,
    generar_instancia,
)
from constructive import (
    ConstructiveConfig,
    ConstructiveResult,
    construct_initial_solution,
    summarize_constructive_result,
)
from objective import ObjectiveConfig, objective_breakdown
from local_search import (
    LocalSearchResult,
    first_improvement_search,
    best_improvement_search,
)


# ==============================================================================

# RUTAS DEL PROYECTO ===========================================================

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================

# CONFIGURACIÓN EXPERIMENTAL ===================================================


@dataclass(frozen=True)
class InstanceParams:
    """
    Parámetros para la generación de instancias.
    """

    n_exams: int = 100
    n_students: int = 2000
    n_rooms: int = 10
    n_slots: Optional[int] = None
    seed: int = 42
    min_exams_per_student: int = 3
    max_exams_per_student: int = 6
    min_room_capacity: int = 30
    max_room_capacity: int = 200
    asegurar_factibilidad_capacidad: bool = True


@dataclass(frozen=True)
class SearchParams:
    """
    Parámetros de búsqueda local.
    """

    max_evaluations: Optional[int] = 20_000
    max_iterations: Optional[int] = None
    include_change_slot: bool = True
    include_swap_slots: bool = True
    first_improvement_seed: int = 42
    track_history: bool = True


@dataclass(frozen=True)
class OutputParams:
    """
    Parámetros de salida.
    """

    output_root: Path = RESULTS_DIR
    save_summary_csv: bool = True
    save_traces_csv: bool = True
    save_plots: bool = True
    show_plots: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuración global de una ejecución experimental.
    """

    instance_params: InstanceParams = field(default_factory=InstanceParams)
    objective_config: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    constructive_config: ConstructiveConfig = field(default_factory=ConstructiveConfig)
    search_params: SearchParams = field(default_factory=SearchParams)
    output_params: OutputParams = field(default_factory=OutputParams)

    # Si no es None, se carga una instancia existente.
    instance_dir: Optional[Path] = None

    verbose: bool = True


# ==============================================================================

# UTILIDADES DE CONFIGURACIÓN ==================================================


def build_constructive_config(
    objective_config: ObjectiveConfig,
    *,
    sort_by_conflicts: bool = True,
    sort_by_size: bool = True,
    prefer_less_loaded_slots: bool = True,
    prefer_best_fit_room: bool = True,
    evaluate_all_feasible_candidates: bool = True,
) -> ConstructiveConfig:
    """
    Construye la configuración de la heurística constructiva reutilizando
    la misma ObjectiveConfig del experimento.
    """
    return ConstructiveConfig(
        objective_config=objective_config,
        sort_by_conflicts=sort_by_conflicts,
        sort_by_size=sort_by_size,
        prefer_less_loaded_slots=prefer_less_loaded_slots,
        prefer_best_fit_room=prefer_best_fit_room,
        evaluate_all_feasible_candidates=evaluate_all_feasible_candidates,
    )


# ==============================================================================

# INSTANCIAS ===================================================================


def resolve_instance_dir(user_value: str) -> Path:
    """
    Resuelve una ruta de instancia.

    Casos aceptados:
    - nombre de directorio dentro de data/instances
    - ruta relativa
    - ruta absoluta
    """
    candidate = Path(user_value)

    if candidate.is_absolute():
        return candidate

    if candidate.exists():
        return candidate.resolve()

    return (BASE_INSTANCES_DIR / user_value).resolve()


def load_or_generate_instance(config: ExperimentConfig) -> ExamInstance:
    """
    Carga una instancia existente o genera una nueva.
    """
    if config.instance_dir is not None:
        return cargar_instancia(config.instance_dir)

    p = config.instance_params
    n_slots = p.n_slots if p.n_slots is not None else max(1, p.n_exams // 2)

    return generar_instancia(
        n_exams=p.n_exams,
        n_students=p.n_students,
        n_rooms=p.n_rooms,
        n_slots=n_slots,
        seed=p.seed,
        min_exams_per_student=p.min_exams_per_student,
        max_exams_per_student=p.max_exams_per_student,
        min_room_capacity=p.min_room_capacity,
        max_room_capacity=p.max_room_capacity,
        asegurar_factibilidad_capacidad=p.asegurar_factibilidad_capacidad,
    )


def build_instance_label(config: ExperimentConfig, instance: ExamInstance) -> str:
    """
    Etiqueta legible para nombrar archivos de salida.
    """
    if config.instance_dir is not None:
        return config.instance_dir.name

    return (
        f"gen_"
        f"E{instance.n_exams}_"
        f"S{instance.n_students}_"
        f"R{instance.n_rooms}_"
        f"T{instance.n_slots}_"
        f"seed{instance.seed}"
    )


# ==============================================================================

# MÉTRICAS =====================================================================


def extract_solution_metrics(
    *,
    algorithm: str,
    solution,
    objective_config: ObjectiveConfig,
    elapsed_time: Optional[float],
    evaluations: Optional[int],
    iterations: Optional[int],
    accepted_moves: Optional[int],
    initial_objective: Optional[float],
    final_objective: Optional[float],
) -> dict[str, Any]:
    """
    Extrae una fila homogénea para la tabla resumen.
    """
    breakdown = objective_breakdown(solution, config=objective_config)

    obj_final = (
        float(final_objective)
        if final_objective is not None
        else float(breakdown["objective"])
    )
    obj_initial = (
        float(initial_objective) if initial_objective is not None else obj_final
    )

    return {
        "algorithm": algorithm,
        "objective": obj_final,
        "initial_objective": obj_initial,
        "final_objective": obj_final,
        "improvement": obj_initial - obj_final,
        "hard_violations": int(breakdown["hard_violations"]),
        "soft_penalty": float(breakdown["soft_penalty"]),
        "student_conflicts": int(breakdown["student_conflicts"]),
        "capacity_violations": int(breakdown["capacity_violations"]),
        "unassigned_exams": int(breakdown["unassigned_exams"]),
        "elapsed_time": float(elapsed_time) if elapsed_time is not None else pd.NA,
        "evaluations": evaluations if evaluations is not None else pd.NA,
        "iterations": iterations if iterations is not None else pd.NA,
        "accepted_moves": accepted_moves if accepted_moves is not None else pd.NA,
    }


def constructive_result_to_row(
    result: ConstructiveResult,
    objective_config: ObjectiveConfig,
    elapsed_time: float,
) -> dict[str, Any]:
    """
    Convierte ConstructiveResult a una fila de resumen.
    """
    return extract_solution_metrics(
        algorithm="constructive",
        solution=result.solution,
        objective_config=objective_config,
        elapsed_time=elapsed_time,
        evaluations=1,
        iterations=None,
        accepted_moves=0,
        initial_objective=result.objective,
        final_objective=result.objective,
    )


def local_search_result_to_row(
    result: LocalSearchResult,
    objective_config: ObjectiveConfig,
) -> dict[str, Any]:
    """
    Convierte LocalSearchResult a una fila de resumen.
    """
    return extract_solution_metrics(
        algorithm=result.algorithm,
        solution=result.solution,
        objective_config=objective_config,
        elapsed_time=result.elapsed_time,
        evaluations=result.evaluations,
        iterations=result.iterations,
        accepted_moves=result.accepted_moves,
        initial_objective=result.initial_objective,
        final_objective=result.final_objective,
    )


# ==============================================================================

# TRAZAS =======================================================================


def constructive_trace_df(result: ConstructiveResult) -> pd.DataFrame:
    """
    Traza mínima para la constructiva.

    Como la constructiva no expone historial incremental, se representa con un
    único punto final.
    """
    return pd.DataFrame(
        [
            {
                "algorithm": "constructive",
                "evaluations": 1,
                "iterations": 0,
                "accepted_moves": 0,
                "objective": float(result.objective),
            }
        ]
    )


def local_search_trace_df(result: LocalSearchResult) -> pd.DataFrame:
    """
    Convierte la historia del resultado de búsqueda local a DataFrame.

    Compatible con:
    - versión ampliada con result.history
    - versión antigua sin history
    """
    history = getattr(result, "history", None)

    if history:
        rows = [
            {
                "algorithm": result.algorithm,
                "evaluations": int(point.evaluations),
                "iterations": int(point.iterations),
                "accepted_moves": int(point.accepted_moves),
                "objective": float(point.objective),
            }
            for point in history
        ]
        return pd.DataFrame(rows)

    # Compatibilidad hacia atrás si no existe history
    if result.initial_objective != result.final_objective:
        return pd.DataFrame(
            [
                {
                    "algorithm": result.algorithm,
                    "evaluations": 1,
                    "iterations": 0,
                    "accepted_moves": 0,
                    "objective": float(result.initial_objective),
                },
                {
                    "algorithm": result.algorithm,
                    "evaluations": int(result.evaluations),
                    "iterations": int(result.iterations),
                    "accepted_moves": int(result.accepted_moves),
                    "objective": float(result.final_objective),
                },
            ]
        )

    return pd.DataFrame(
        [
            {
                "algorithm": result.algorithm,
                "evaluations": int(result.evaluations),
                "iterations": int(result.iterations),
                "accepted_moves": int(result.accepted_moves),
                "objective": float(result.final_objective),
            }
        ]
    )


# ==============================================================================

# GRÁFICAS =====================================================================


def plot_objective_evolution(
    traces_df: pd.DataFrame,
    *,
    title: str,
    x_axis: str,
    output_path: Optional[Path] = None,
    show_plot: bool = False,
) -> None:
    """
    Dibuja la evolución del valor de la función objetivo.

    x_axis permitido:
    - evaluations
    - iterations
    - accepted_moves
    """
    valid_axes = {"evaluations", "iterations", "accepted_moves"}
    if x_axis not in valid_axes:
        raise ValueError(f"x_axis debe ser uno de {sorted(valid_axes)}")

    plt.figure(figsize=(10, 6))

    for algorithm, group in traces_df.groupby("algorithm"):
        group = group.sort_values(x_axis)
        plt.plot(group[x_axis], group["objective"], marker="o", label=algorithm)

    plt.xlabel(x_axis)
    plt.ylabel("objective")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close()


# ==============================================================================

# EJECUCIÓN DE ALGORITMOS ======================================================


def run_constructive(
    instance: ExamInstance,
    constructive_config: ConstructiveConfig,
    objective_config: ObjectiveConfig,
    *,
    verbose: bool,
) -> tuple[ConstructiveResult, dict[str, Any], pd.DataFrame]:
    """
    Ejecuta la heurística constructiva.
    """
    start = perf_counter()
    result = construct_initial_solution(instance, config=constructive_config)
    elapsed = perf_counter() - start

    if verbose:
        print("\n=== CONSTRUCTIVA ===")
        print(summarize_constructive_result(result))

    row = constructive_result_to_row(
        result=result,
        objective_config=objective_config,
        elapsed_time=elapsed,
    )
    trace = constructive_trace_df(result)
    return result, row, trace


def run_first_improvement(
    initial_solution,
    objective_config: ObjectiveConfig,
    search_params: SearchParams,
) -> tuple[LocalSearchResult, dict[str, Any], pd.DataFrame]:
    """
    Ejecuta búsqueda local first improvement.
    """
    result = first_improvement_search(
        initial_solution=initial_solution,
        config=objective_config,
        max_evaluations=search_params.max_evaluations,
        max_iterations=search_params.max_iterations,
        include_change_slot=search_params.include_change_slot,
        include_swap_slots=search_params.include_swap_slots,
        seed=search_params.first_improvement_seed,
        track_history=search_params.track_history,
    )
    row = local_search_result_to_row(result, objective_config)
    trace = local_search_trace_df(result)
    return result, row, trace


def run_best_improvement(
    initial_solution,
    objective_config: ObjectiveConfig,
    search_params: SearchParams,
) -> tuple[LocalSearchResult, dict[str, Any], pd.DataFrame]:
    """
    Ejecuta búsqueda local best improvement.
    """
    result = best_improvement_search(
        initial_solution=initial_solution,
        config=objective_config,
        max_evaluations=search_params.max_evaluations,
        max_iterations=search_params.max_iterations,
        include_change_slot=search_params.include_change_slot,
        include_swap_slots=search_params.include_swap_slots,
        track_history=search_params.track_history,
    )
    row = local_search_result_to_row(result, objective_config)
    trace = local_search_trace_df(result)
    return result, row, trace


# ==============================================================================

# SALIDAS ======================================================================


def save_outputs(
    *,
    config: ExperimentConfig,
    experiment_dir: Path,
    summary_df: pd.DataFrame,
    traces_df: pd.DataFrame,
    instance_label: str,
) -> None:
    """
    Guarda tablas y gráficas del experimento.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if config.output_params.save_summary_csv:
        summary_df.to_csv(experiment_dir / "summary.csv", index=False)

    if config.output_params.save_traces_csv:
        traces_df.to_csv(experiment_dir / "traces.csv", index=False)

    if config.output_params.save_plots:
        plot_objective_evolution(
            traces_df,
            title=f"Evolución del objetivo - {instance_label}",
            x_axis="evaluations",
            output_path=experiment_dir / "objective_by_evaluations.png",
            show_plot=config.output_params.show_plots,
        )
        plot_objective_evolution(
            traces_df,
            title=f"Evolución del objetivo - {instance_label}",
            x_axis="accepted_moves",
            output_path=experiment_dir / "objective_by_accepted_moves.png",
            show_plot=config.output_params.show_plots,
        )


# ==============================================================================

# EXPERIMENTO SIMPLE ===========================================================


def run_single_experiment(
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta un experimento completo:
    - carga/genera instancia
    - constructiva
    - first improvement
    - best improvement
    """
    instance = load_or_generate_instance(config)
    label = build_instance_label(config, instance)
    experiment_dir = config.output_params.output_root / label

    if config.verbose:
        print("=" * 80)
        print(f"INSTANCIA: {label}")
        print("=" * 80)

    constructive_result, constructive_row, constructive_trace = run_constructive(
        instance=instance,
        constructive_config=config.constructive_config,
        objective_config=config.objective_config,
        verbose=config.verbose,
    )

    first_result, first_row, first_trace = run_first_improvement(
        initial_solution=constructive_result.solution,
        objective_config=config.objective_config,
        search_params=config.search_params,
    )

    best_result, best_row, best_trace = run_best_improvement(
        initial_solution=constructive_result.solution,
        objective_config=config.objective_config,
        search_params=config.search_params,
    )

    summary_df = pd.DataFrame([constructive_row, first_row, best_row]).sort_values(
        "objective", ascending=True
    )

    traces_df = pd.concat(
        [constructive_trace, first_trace, best_trace],
        ignore_index=True,
    )

    if config.verbose:
        print("\n=== TABLA RESUMEN ===")
        print(summary_df.to_string(index=False))

    save_outputs(
        config=config,
        experiment_dir=experiment_dir,
        summary_df=summary_df,
        traces_df=traces_df,
        instance_label=label,
    )

    return summary_df, traces_df


# ==============================================================================

# EXPERIMENTO POR LOTES ========================================================


def run_batch_over_seeds(
    base_config: ExperimentConfig,
    seeds: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta varios experimentos cambiando la semilla de la instancia.

    Devuelve:
    - detailed_df: una fila por algoritmo y semilla
    - aggregated_df: medias y desviaciones por algoritmo
    """
    detailed_frames: list[pd.DataFrame] = []

    for seed in seeds:
        instance_params = InstanceParams(
            **{**asdict(base_config.instance_params), "seed": seed}
        )

        config = ExperimentConfig(
            instance_params=instance_params,
            objective_config=base_config.objective_config,
            constructive_config=build_constructive_config(
                base_config.objective_config,
                sort_by_conflicts=base_config.constructive_config.sort_by_conflicts,
                sort_by_size=base_config.constructive_config.sort_by_size,
                prefer_less_loaded_slots=base_config.constructive_config.prefer_less_loaded_slots,
                prefer_best_fit_room=base_config.constructive_config.prefer_best_fit_room,
                evaluate_all_feasible_candidates=base_config.constructive_config.evaluate_all_feasible_candidates,
            ),
            search_params=base_config.search_params,
            output_params=base_config.output_params,
            instance_dir=None,
            verbose=False,
        )

        summary_df, _ = run_single_experiment(config)
        summary_df = summary_df.copy()
        summary_df["seed"] = seed
        detailed_frames.append(summary_df)

    detailed_df = pd.concat(detailed_frames, ignore_index=True)

    aggregated_df = (
        detailed_df.groupby("algorithm", dropna=False)
        .agg(
            objective_mean=("objective", "mean"),
            objective_std=("objective", "std"),
            soft_penalty_mean=("soft_penalty", "mean"),
            hard_violations_mean=("hard_violations", "mean"),
            elapsed_time_mean=("elapsed_time", "mean"),
            evaluations_mean=("evaluations", "mean"),
            iterations_mean=("iterations", "mean"),
            accepted_moves_mean=("accepted_moves", "mean"),
        )
        .reset_index()
        .sort_values("objective_mean", ascending=True)
    )

    batch_dir = base_config.output_params.output_root / "batch"

    if base_config.output_params.save_summary_csv:
        batch_dir.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(batch_dir / "batch_detailed.csv", index=False)
        aggregated_df.to_csv(batch_dir / "batch_aggregated.csv", index=False)

    return detailed_df, aggregated_df


# ==============================================================================

# ARGUMENTOS ===================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Script principal de experimentación para planificación de exámenes."
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--instance-name",
        type=str,
        default=None,
        help=(
            "Nombre del directorio de una instancia dentro de data/instances. "
            "Ejemplo: instancia_100_2000_10_40_seed42"
        ),
    )
    source_group.add_argument(
        "--instance-path",
        type=str,
        default=None,
        help="Ruta completa o relativa a un directorio de instancia guardado.",
    )
    source_group.add_argument(
        "--generate",
        action="store_true",
        help="Generar una nueva instancia.",
    )

    parser.add_argument("--n-exams", type=int, default=100)
    parser.add_argument("--n-students", type=int, default=2000)
    parser.add_argument("--n-rooms", type=int, default=10)
    parser.add_argument("--n-slots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-evaluations", type=int, default=20_000)
    parser.add_argument("--max-iterations", type=int, default=None)

    parser.add_argument("--hard-penalty", type=float, default=1_000_000.0)
    parser.add_argument("--w-consecutive", type=float, default=5.0)
    parser.add_argument("--w-same-day", type=float, default=2.0)
    parser.add_argument("--w-distribution", type=float, default=1.0)
    parser.add_argument("--slots-per-day", type=int, default=4)

    parser.add_argument(
        "--disable-change-slot",
        action="store_true",
        help="Desactiva el operador change_slot.",
    )
    parser.add_argument(
        "--disable-swap-slots",
        action="store_true",
        help="Desactiva el operador swap_slots.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directorio raíz donde guardar resultados.",
    )
    parser.add_argument("--no-save-summary-csv", action="store_true")
    parser.add_argument("--no-save-traces-csv", action="store_true")
    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--batch-seeds",
        type=str,
        default=None,
        help="Lista de semillas separadas por comas. Ejemplo: 1,2,3,4,5",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    objective_config = ObjectiveConfig(
        hard_penalty=args.hard_penalty,
        w_consecutive=args.w_consecutive,
        w_same_day=args.w_same_day,
        w_distribution=args.w_distribution,
        slots_per_day=args.slots_per_day,
    )

    constructive_config = build_constructive_config(objective_config)

    instance_params = InstanceParams(
        n_exams=args.n_exams,
        n_students=args.n_students,
        n_rooms=args.n_rooms,
        n_slots=args.n_slots,
        seed=args.seed,
    )

    search_params = SearchParams(
        max_evaluations=args.max_evaluations,
        max_iterations=args.max_iterations,
        include_change_slot=not args.disable_change_slot,
        include_swap_slots=not args.disable_swap_slots,
        first_improvement_seed=args.seed,
        track_history=True,
    )

    output_params = OutputParams(
        output_root=Path(args.output_dir).resolve(),
        save_summary_csv=not args.no_save_summary_csv,
        save_traces_csv=not args.no_save_traces_csv,
        save_plots=not args.no_save_plots,
        show_plots=args.show_plots,
    )

    instance_dir: Optional[Path] = None
    if args.instance_name is not None:
        instance_dir = resolve_instance_dir(args.instance_name)
    elif args.instance_path is not None:
        instance_dir = resolve_instance_dir(args.instance_path)

    return ExperimentConfig(
        instance_params=instance_params,
        objective_config=objective_config,
        constructive_config=constructive_config,
        search_params=search_params,
        output_params=output_params,
        instance_dir=instance_dir,
        verbose=not args.quiet,
    )


# ==============================================================================

# MAIN =========================================================================


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = config_from_args(args)

    if args.batch_seeds:
        seeds = [int(x.strip()) for x in args.batch_seeds.split(",") if x.strip()]
        detailed_df, aggregated_df = run_batch_over_seeds(config, seeds)

        print("\n=== RESULTADOS DETALLADOS POR SEMILLA ===")
        print(detailed_df.to_string(index=False))

        print("\n=== RESULTADOS AGREGADOS ===")
        print(aggregated_df.to_string(index=False))
    else:
        run_single_experiment(config)


if __name__ == "__main__":
    main()
