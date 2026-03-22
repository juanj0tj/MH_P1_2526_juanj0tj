from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Optional
import os
import platform
import matplotlib.pyplot as plt
import pandas as pd

from instance_manager import (
    BASE_INSTANCES_DIR,
    ExamInstance,
    cargar_instancia,
    generar_instancia,
    guardar_instancia,
    listar_instancias,
    seleccionar_instancia,
    resumir_instancia,
    eliminar_instancia,
    nombre_instancia_por_defecto,
    pedir_entero,
    pedir_si_no,
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

    max_evaluations: Optional[int] = 50_000
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
    seed: Optional[int] = None,
    shuffle_exams: bool = True,
) -> ConstructiveConfig:
    return ConstructiveConfig(
        objective_config=objective_config,
        seed=seed,
        shuffle_exams=shuffle_exams,
    )


# ==============================================================================

# INSTANCIAS ===================================================================


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

# MENÚ - PARÁMETROS DE EJECUCIÓN ===============================================


def pedir_float(
    mensaje: str,
    valor_por_defecto: Optional[float] = None,
    minimo: Optional[float] = None,
) -> float:
    while True:
        texto = input(mensaje).strip()

        if texto == "" and valor_por_defecto is not None:
            valor = valor_por_defecto
        else:
            try:
                valor = float(texto)
            except ValueError:
                print("Entrada no válida. Debes introducir un número.")
                continue

        if minimo is not None and valor < minimo:
            print(f"El valor debe ser >= {minimo}.")
            continue

        return valor


def pedir_entero_o_none(
    mensaje: str,
    valor_por_defecto: Optional[int] = None,
    minimo: Optional[int] = None,
) -> Optional[int]:
    while True:
        texto = input(mensaje).strip()

        if texto == "":
            return valor_por_defecto

        if texto.lower() in {"none", "ninguno", "sin limite", "sin límite"}:
            return None

        try:
            valor = int(texto)
        except ValueError:
            print("Entrada no válida. Debes introducir un entero o dejarlo vacío.")
            continue

        if minimo is not None and valor < minimo:
            print(f"El valor debe ser >= {minimo}.")
            continue

        return valor


def pedir_output_params() -> OutputParams:
    print("\n=== CONFIGURACIÓN DE SALIDA ===")
    guardar_summary = pedir_si_no("¿Guardar summary.csv?", valor_por_defecto=True)
    guardar_traces = pedir_si_no("¿Guardar traces.csv?", valor_por_defecto=True)
    guardar_plots = pedir_si_no("¿Guardar gráficas PNG?", valor_por_defecto=True)
    mostrar_plots = False

    if guardar_plots:
        mostrar_plots = pedir_si_no(
            "¿Mostrar gráficas por pantalla?",
            valor_por_defecto=False,
        )

    return OutputParams(
        output_root=RESULTS_DIR,
        save_summary_csv=guardar_summary,
        save_traces_csv=guardar_traces,
        save_plots=guardar_plots,
        show_plots=mostrar_plots,
    )


def pedir_objective_config() -> ObjectiveConfig:
    print("\n=== CONFIGURACIÓN DE LA FUNCIÓN OBJETIVO ===")
    hard_penalty = pedir_float(
        "Penalización dura [100000.0]: ",
        valor_por_defecto=100000.0,
        minimo=0.0,
    )
    w_consecutive = pedir_float(
        "Peso consecutivos [5.0]: ",
        valor_por_defecto=5.0,
        minimo=0.0,
    )
    w_same_day = pedir_float(
        "Peso mismo día [2.0]: ",
        valor_por_defecto=2.0,
        minimo=0.0,
    )
    w_distribution = pedir_float(
        "Peso distribución [1.0]: ",
        valor_por_defecto=1.0,
        minimo=0.0,
    )
    slots_per_day = pedir_entero(
        "Franjas por día [4]: ",
        valor_por_defecto=4,
        minimo=1,
    )

    return ObjectiveConfig(
        hard_penalty=hard_penalty,
        w_consecutive=w_consecutive,
        w_same_day=w_same_day,
        w_distribution=w_distribution,
        slots_per_day=slots_per_day,
    )


def pedir_search_params() -> SearchParams:
    print("\n=== CONFIGURACIÓN DE BÚSQUEDA LOCAL ===")
    max_evaluations = pedir_entero_o_none(
        "Máximo de evaluaciones [20000] (Enter para 20000): ",
        valor_por_defecto=20_000,
        minimo=1,
    )
    max_iterations = pedir_entero_o_none(
        "Máximo de iteraciones [sin límite] (Enter para None): ",
        valor_por_defecto=None,
        minimo=1,
    )
    include_change_slot = pedir_si_no(
        "¿Activar operador change_slot?",
        valor_por_defecto=True,
    )
    include_swap_slots = pedir_si_no(
        "¿Activar operador swap_slots?",
        valor_por_defecto=True,
    )

    if not include_change_slot and not include_swap_slots:
        print("Debe quedar activado al menos un operador. Se activará change_slot.")
        include_change_slot = True

    first_seed = pedir_entero(
        "Semilla para first_improvement [42]: ",
        valor_por_defecto=42,
        minimo=0,
    )
    track_history = pedir_si_no(
        "¿Registrar historia para trazas y gráficas?",
        valor_por_defecto=True,
    )

    return SearchParams(
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        include_change_slot=include_change_slot,
        include_swap_slots=include_swap_slots,
        first_improvement_seed=first_seed,
        track_history=track_history,
    )


# ==============================================================================

# MENÚ - OPCIONES DE INSTANCIAS ================================================


def opcion_generar_instancia() -> None:
    print("\n=== GENERAR NUEVA INSTANCIA ===")

    n_exams = pedir_entero(
        "Número de exámenes [100]: ",
        valor_por_defecto=100,
        minimo=1,
    )
    n_students = pedir_entero(
        "Número de estudiantes [2000]: ",
        valor_por_defecto=2000,
        minimo=1,
    )
    n_rooms = pedir_entero(
        "Número de aulas [10]: ",
        valor_por_defecto=10,
        minimo=1,
    )

    n_slots_default = max(1, n_exams // 2)
    n_slots = pedir_entero(
        "Número de franjas horarias [n_exams/2]: ",
        valor_por_defecto=n_slots_default,
        minimo=1,
    )
    seed = pedir_entero("Semilla [42]: ", valor_por_defecto=42, minimo=0)

    min_exams_per_student = pedir_entero(
        "Mínimo de exámenes por estudiante [3]: ",
        valor_por_defecto=3,
        minimo=1,
    )
    max_exams_per_student = pedir_entero(
        "Máximo de exámenes por estudiante [6]: ",
        valor_por_defecto=6,
        minimo=min_exams_per_student,
    )
    min_room_capacity = pedir_entero(
        "Capacidad mínima de aula [30]: ",
        valor_por_defecto=30,
        minimo=1,
    )
    max_room_capacity = pedir_entero(
        "Capacidad máxima de aula [200]: ",
        valor_por_defecto=200,
        minimo=min_room_capacity,
    )
    asegurar_factibilidad = pedir_si_no(
        "¿Asegurar que el mayor examen quepa en al menos un aula?",
        valor_por_defecto=True,
    )

    instancia = generar_instancia(
        n_exams=n_exams,
        n_students=n_students,
        n_rooms=n_rooms,
        n_slots=n_slots,
        seed=seed,
        min_exams_per_student=min_exams_per_student,
        max_exams_per_student=max_exams_per_student,
        min_room_capacity=min_room_capacity,
        max_room_capacity=max_room_capacity,
        asegurar_factibilidad_capacidad=asegurar_factibilidad,
    )

    resumir_instancia(instancia)

    nombre_defecto = nombre_instancia_por_defecto(
        n_exams, n_students, n_rooms, n_slots, seed
    )

    nombre_carpeta = input(
        f"Nombre de la carpeta para guardar la instancia [{nombre_defecto}]: "
    ).strip()

    if nombre_carpeta == "":
        nombre_carpeta = nombre_defecto

    ruta_guardado = BASE_INSTANCES_DIR / nombre_carpeta

    if ruta_guardado.exists():
        sobrescribir = pedir_si_no(
            f"La carpeta '{nombre_carpeta}' ya existe. ¿Sobrescribir?",
            valor_por_defecto=False,
        )
        if not sobrescribir:
            print("Operación cancelada.")
            return

    guardar_instancia(instancia, ruta_guardado)
    print(f"\nInstancia guardada en: {ruta_guardado}")


def opcion_resumen_instancia() -> None:
    print("\n=== MOSTRAR RESUMEN DE INSTANCIA ===")

    ruta = seleccionar_instancia()
    if ruta is None:
        print("Operación cancelada.")
        return

    try:
        instancia = cargar_instancia(ruta)
        resumir_instancia(instancia)
    except Exception as e:
        print(f"Error al cargar la instancia: {e}")


def opcion_listar_instancias() -> None:
    listar_instancias()


def opcion_eliminar_instancia() -> None:
    print("\n=== ELIMINAR INSTANCIA ===")

    ruta = seleccionar_instancia()
    if ruta is None:
        print("Operación cancelada.")
        return

    confirmar = pedir_si_no(
        f"¿Seguro que deseas eliminar la instancia '{ruta.name}'?",
        valor_por_defecto=False,
    )

    if not confirmar:
        print("Eliminación cancelada.")
        return

    try:
        eliminar_instancia(ruta)
        print(f"Instancia '{ruta.name}' eliminada correctamente.")
    except Exception as e:
        print(f"Error al eliminar la instancia: {e}")


# ==============================================================================

# MENÚ - EJECUCIÓN DE INSTANCIAS ===============================================


def opcion_ejecutar_instancia() -> None:
    print("\n=== EJECUTAR INSTANCIA EXISTENTE ===")

    ruta = seleccionar_instancia()
    if ruta is None:
        print("Operación cancelada.")
        return

    try:
        instancia = cargar_instancia(ruta)
    except Exception as e:
        print(f"Error al cargar la instancia: {e}")
        return

    resumir_instancia(instancia)

    objective_config = pedir_objective_config()
    search_params = pedir_search_params()
    output_params = pedir_output_params()

    constructive_seed = pedir_entero(
        "Semilla de la constructiva aleatoria [misma semilla de instancia]: ",
        valor_por_defecto=instancia.seed,
        minimo=0,
    )
    shuffle_exams = pedir_si_no(
        "¿Barajar el orden de exámenes en la constructiva?",
        valor_por_defecto=True,
    )
    verbose = pedir_si_no(
        "¿Mostrar resultados detallados por pantalla?",
        valor_por_defecto=True,
    )

    config = ExperimentConfig(
        instance_dir=ruta,
        objective_config=objective_config,
        constructive_config=build_constructive_config(
            objective_config,
            seed=constructive_seed,
            shuffle_exams=shuffle_exams,
        ),
        search_params=search_params,
        output_params=output_params,
        verbose=verbose,
    )

    try:
        summary_df, _ = run_single_experiment(config)

        print("\n=== EJECUCIÓN FINALIZADA ===")
        print(f"Resultados guardados en: {output_params.output_root / ruta.name}")
        print("\nResumen final:")
        print(summary_df.to_string(index=False))
    except Exception as e:
        print(f"Error durante la ejecución: {e}")


# ==============================================================================

# MENÚ - FUNCIONES AUXILIARES ==================================================


def clear_screen() -> None:
    """
    Limpia la terminal en función del sistema operativo.
    """
    os.system("cls" if platform.system() == "Windows" else "clear")


# ==============================================================================

# MENÚ PRINCIPAL ===============================================================


def mostrar_menu() -> None:
    print("\n" + "=" * 60)
    print("     PLANIFICACIÓN DE EXÁMENES - MENÚ PRINCIPAL")
    print("=" * 60)
    print("1. Generar una nueva instancia")
    print("2. Ejecutar una instancia existente")
    print("3. Mostrar resumen de una instancia")
    print("4. Listar instancias disponibles")
    print("5. Eliminar una instancia")
    print("0. Salir")


def main() -> None:
    BASE_INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        mostrar_menu()
        opcion = input("\nSelecciona una opción: ").strip()

        clear_screen()

        if opcion == "1":
            opcion_generar_instancia()
        elif opcion == "2":
            opcion_ejecutar_instancia()
        elif opcion == "3":
            opcion_resumen_instancia()
        elif opcion == "4":
            opcion_listar_instancias()
        elif opcion == "5":
            opcion_eliminar_instancia()
        elif opcion == "0":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")


if __name__ == "__main__":
    main()
