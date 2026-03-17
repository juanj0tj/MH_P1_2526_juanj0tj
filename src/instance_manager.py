import os
import json
import shutil
import random
from dataclasses import dataclass
from typing import Dict, Set, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd


# ==============================================================================

# CONFIGURACIÓN ================================================================


CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent
BASE_INSTANCES_DIR = PROJECT_ROOT / "data" / "instances"
BASE_INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================

# ESTRUCTURA DE LA INSTANCIA ===================================================


@dataclass
class ExamInstance:
    n_exams: int
    n_students: int
    n_rooms: int
    n_slots: int
    seed: int

    student_exam: pd.DataFrame
    exams_df: pd.DataFrame
    rooms_df: pd.DataFrame

    exam_sizes: Dict[int, int]
    room_capacities: Dict[int, int]
    students_by_exam: Dict[int, Set[int]]
    exams_by_student: Dict[int, List[int]]
    conflict_matrix: np.ndarray


# ==============================================================================

# GENERACIÓN DE INSTANCIAS =====================================================


def generar_instancia(
    n_exams: int = 100,
    n_students: int = 2000,
    n_rooms: int = 10,
    n_slots: Optional[int] = None,
    seed: int = 42,
    min_exams_per_student: int = 3,
    max_exams_per_student: int = 6,
    min_room_capacity: int = 30,
    max_room_capacity: int = 200,
    asegurar_factibilidad_capacidad: bool = True,
) -> ExamInstance:
    """
    Genera una instancia sintética reproducible del problema de planificación de exámenes.
    """

    random.seed(seed)
    np.random.seed(seed)

    if n_slots is None:
        n_slots = max(1, n_exams // 2)

    # --------------------------------------------------------
    # 1. Generar relación estudiante-examen
    # --------------------------------------------------------
    student_exam_rows = []

    for student_id in range(n_students):
        k = random.randint(min_exams_per_student, max_exams_per_student)
        k = min(k, n_exams)

        exams = np.random.choice(n_exams, size=k, replace=False)

        for exam_id in exams:
            student_exam_rows.append((student_id, int(exam_id)))

    student_exam = pd.DataFrame(student_exam_rows, columns=["student", "exam"])
    student_exam = student_exam.sort_values(["student", "exam"]).reset_index(drop=True)

    # --------------------------------------------------------
    # 2. Número de estudiantes por examen
    # --------------------------------------------------------
    exam_counts = (
        student_exam.groupby("exam").size().reindex(range(n_exams), fill_value=0)
    )

    exams_df = pd.DataFrame({"exam": range(n_exams), "n_students": exam_counts.values})

    # --------------------------------------------------------
    # 3. Generar aulas y capacidades
    # --------------------------------------------------------
    capacities = np.random.randint(
        min_room_capacity, max_room_capacity + 1, size=n_rooms
    )

    rooms_df = pd.DataFrame({"room": range(n_rooms), "capacity": capacities})

    # --------------------------------------------------------
    # 4. Ajuste opcional de factibilidad por capacidad
    # --------------------------------------------------------
    if asegurar_factibilidad_capacidad:
        max_exam_size = int(exams_df["n_students"].max())
        max_room_cap = int(rooms_df["capacity"].max())

        if max_exam_size > max_room_cap:
            idx_largest_room = rooms_df["capacity"].idxmax()
            rooms_df.loc[idx_largest_room, "capacity"] = max_exam_size

    # --------------------------------------------------------
    # 5. Estructuras auxiliares
    # --------------------------------------------------------
    exam_sizes = dict(zip(exams_df["exam"], exams_df["n_students"]))
    room_capacities = dict(zip(rooms_df["room"], rooms_df["capacity"]))

    students_by_exam_series = student_exam.groupby("exam")["student"].apply(set)
    students_by_exam = {
        exam: set(students_by_exam_series.get(exam, set())) for exam in range(n_exams)
    }

    exams_by_student_series = student_exam.groupby("student")["exam"].apply(list)
    exams_by_student = {
        student: list(exams_by_student_series.get(student, []))
        for student in range(n_students)
    }
    # --------------------------------------------------------
    # 6. Matriz de conflictos
    # --------------------------------------------------------
    conflict_matrix = np.zeros((n_exams, n_exams), dtype=bool)

    for exam_list in exams_by_student.values():
        m = len(exam_list)
        for i in range(m):
            e1 = exam_list[i]
            for j in range(i + 1, m):
                e2 = exam_list[j]
                conflict_matrix[e1, e2] = True
                conflict_matrix[e2, e1] = True

    return ExamInstance(
        n_exams=n_exams,
        n_students=n_students,
        n_rooms=n_rooms,
        n_slots=n_slots,
        seed=seed,
        student_exam=student_exam,
        exams_df=exams_df,
        rooms_df=rooms_df,
        exam_sizes=exam_sizes,
        room_capacities=room_capacities,
        students_by_exam=students_by_exam,
        exams_by_student=exams_by_student,
        conflict_matrix=conflict_matrix,
    )


# ==============================================================================

# GUARDAR / CARGAR INSTANCIAS ==================================================


def guardar_instancia(instancia: ExamInstance, directorio: Path) -> None:
    """
    Guarda la instancia en disco usando CSV + JSON.
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    instancia.student_exam.to_csv(directorio / "student_exam.csv", index=False)
    instancia.exams_df.to_csv(directorio / "exams.csv", index=False)
    instancia.rooms_df.to_csv(directorio / "rooms.csv", index=False)

    metadata = {
        "n_exams": instancia.n_exams,
        "n_students": instancia.n_students,
        "n_rooms": instancia.n_rooms,
        "n_slots": instancia.n_slots,
        "seed": instancia.seed,
    }

    with open(directorio / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def cargar_instancia(directorio: Path) -> ExamInstance:
    """
    Carga una instancia guardada en disco y reconstruye las estructuras auxiliares.
    Esta función no se usa en el menú, pero queda disponible para otros scripts.
    """
    directorio = Path(directorio)

    with open(directorio / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    student_exam = pd.read_csv(directorio / "student_exam.csv")
    exams_df = pd.read_csv(directorio / "exams.csv")
    rooms_df = pd.read_csv(directorio / "rooms.csv")

    n_exams = metadata["n_exams"]
    n_students = metadata["n_students"]

    exam_sizes = dict(zip(exams_df["exam"], exams_df["n_students"]))
    room_capacities = dict(zip(rooms_df["room"], rooms_df["capacity"]))

    students_by_exam_series = student_exam.groupby("exam")["student"].apply(set)
    students_by_exam = {
        exam: set(students_by_exam_series.get(exam, set())) for exam in range(n_exams)
    }

    exams_by_student_series = student_exam.groupby("student")["exam"].apply(list)
    exams_by_student = {
        student: list(exams_by_student_series.get(student, []))
        for student in range(n_students)
    }

    conflict_matrix = np.zeros((n_exams, n_exams), dtype=bool)

    for exam_list in exams_by_student.values():
        m = len(exam_list)
        for i in range(m):
            e1 = exam_list[i]
            for j in range(i + 1, m):
                e2 = exam_list[j]
                conflict_matrix[e1, e2] = True
                conflict_matrix[e2, e1] = True

    return ExamInstance(
        n_exams=metadata["n_exams"],
        n_students=metadata["n_students"],
        n_rooms=metadata["n_rooms"],
        n_slots=metadata["n_slots"],
        seed=metadata["seed"],
        student_exam=student_exam,
        exams_df=exams_df,
        rooms_df=rooms_df,
        exam_sizes=exam_sizes,
        room_capacities=room_capacities,
        students_by_exam=students_by_exam,
        exams_by_student=exams_by_student,
        conflict_matrix=conflict_matrix,
    )


# ==============================================================================

# UTILIDADES DE INSTANCIAS =====================================================


def obtener_instancias_disponibles(base_dir: Path = BASE_INSTANCES_DIR) -> List[str]:
    """
    Devuelve la lista de carpetas de instancias disponibles.
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        return []

    instancias = [p.name for p in base_dir.iterdir() if p.is_dir()]
    instancias.sort()
    return instancias


def listar_instancias(base_dir: Path = BASE_INSTANCES_DIR) -> None:
    """
    Muestra por pantalla las instancias disponibles.
    """
    instancias = obtener_instancias_disponibles(base_dir)

    print("\n=== INSTANCIAS DISPONIBLES ===")
    if not instancias:
        print("No hay instancias guardadas.")
        return

    for i, nombre in enumerate(instancias, start=1):
        print(f"{i}. {nombre}")


def seleccionar_instancia(base_dir: Path = BASE_INSTANCES_DIR) -> Optional[Path]:
    """
    Permite seleccionar una instancia de entre las disponibles.
    Devuelve la ruta completa o None si se cancela.
    """
    base_dir = Path(base_dir)
    instancias = obtener_instancias_disponibles(base_dir)

    if not instancias:
        print("\nNo hay instancias guardadas.")
        return None

    print("\nInstancias disponibles:")
    for i, nombre in enumerate(instancias, start=1):
        print(f"  {i}. {nombre}")

    while True:
        opcion = input(
            "\nSelecciona una instancia por número (Enter para cancelar): "
        ).strip()

        if opcion == "":
            return None

        try:
            indice = int(opcion)
            if 1 <= indice <= len(instancias):
                return base_dir / instancias[indice - 1]
        except ValueError:
            pass

        print("Selección no válida.")


def eliminar_instancia(directorio: Path) -> None:
    """
    Elimina completamente la carpeta de una instancia.
    """
    directorio = Path(directorio)

    if directorio.exists() and directorio.is_dir():
        shutil.rmtree(directorio)
    else:
        raise FileNotFoundError(f"No existe la carpeta: {directorio}")


def nombre_instancia_por_defecto(
    n_exams: int, n_students: int, n_rooms: int, n_slots: int, seed: int
) -> str:
    return f"instancia_{n_exams}_{n_students}_{n_rooms}_{n_slots}_seed{seed}"


# ==============================================================================

# RESUMEN ======================================================================


def resumir_instancia(instancia: ExamInstance) -> None:
    """
    Imprime un resumen de la instancia.
    """
    print("\n=== RESUMEN DE LA INSTANCIA ===")
    print(f"Exámenes:             {instancia.n_exams}")
    print(f"Estudiantes:          {instancia.n_students}")
    print(f"Aulas:                {instancia.n_rooms}")
    print(f"Franjas horarias:     {instancia.n_slots}")
    print(f"Semilla:              {instancia.seed}")
    print()
    print(f"Tamaño medio examen:  {instancia.exams_df['n_students'].mean():.2f}")
    print(f"Capacidad media aula: {instancia.rooms_df['capacity'].mean():.2f}")
    print(f"Mayor examen:         {int(instancia.exams_df['n_students'].max())}")
    print(f"Mayor aula:           {int(instancia.rooms_df['capacity'].max())}")
    print(f"Exámenes vacíos:      {int((instancia.exams_df['n_students'] == 0).sum())}")
    print()


# ==============================================================================

# ENTRADA POR CONSOLA ==========================================================


def pedir_entero(
    mensaje: str, valor_por_defecto: Optional[int] = None, minimo: Optional[int] = None
) -> int:
    while True:
        texto = input(mensaje).strip()

        if texto == "" and valor_por_defecto is not None:
            valor = valor_por_defecto
        else:
            try:
                valor = int(texto)
            except ValueError:
                print("Entrada no válida. Debes introducir un número entero.")
                continue

        if minimo is not None and valor < minimo:
            print(f"El valor debe ser >= {minimo}.")
            continue

        return valor


def pedir_si_no(mensaje: str, valor_por_defecto: bool = True) -> bool:
    while True:
        sufijo = "[S/n]" if valor_por_defecto else "[s/N]"
        texto = input(f"{mensaje} {sufijo}: ").strip().lower()

        if texto == "":
            return valor_por_defecto
        if texto in ("s", "si", "sí", "y", "yes"):
            return True
        if texto in ("n", "no"):
            return False

        print("Respuesta no válida. Escribe 's' o 'n'.")


# ==============================================================================

# OPCIONES DEL MENÚ ============================================================


def opcion_generar_instancia() -> None:
    print("\n=== GENERAR NUEVA INSTANCIA ===")

    n_exams = pedir_entero(
        "Número de exámenes [100]: ", valor_por_defecto=100, minimo=1
    )
    n_students = pedir_entero(
        "Número de estudiantes [2000]: ", valor_por_defecto=2000, minimo=1
    )
    n_rooms = pedir_entero("Número de aulas [10]: ", valor_por_defecto=10, minimo=1)

    n_slots_default = max(1, n_exams // 2)
    n_slots = pedir_entero(
        "Número de franjas horarias [n_exams/2]: ",
        valor_por_defecto=n_slots_default,
        minimo=1,
    )
    seed = pedir_entero("Semilla [42]: ", valor_por_defecto=42, minimo=0)

    min_exams_per_student = pedir_entero(
        "Mínimo de exámenes por estudiante [3]: ", valor_por_defecto=3, minimo=1
    )
    max_exams_per_student = pedir_entero(
        "Máximo de exámenes por estudiante [6]: ",
        valor_por_defecto=6,
        minimo=min_exams_per_student,
    )
    min_room_capacity = pedir_entero(
        "Capacidad mínima de aula [30]: ", valor_por_defecto=30, minimo=1
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

    BASE_INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

    nombre_defecto = nombre_instancia_por_defecto(
        n_exams, n_students, n_rooms, n_slots, seed
    )

    nombre_carpeta = input(
        f"Nombre de la carpeta para guardar la instancia [{nombre_defecto}]: "
    ).strip()

    if nombre_carpeta == "":
        nombre_carpeta = nombre_defecto

    ruta_guardado = BASE_INSTANCES_DIR / nombre_carpeta

    if os.path.exists(ruta_guardado):
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

    nombre = ruta.name

    confirmar = pedir_si_no(
        f"¿Seguro que deseas eliminar la instancia '{nombre}'?", valor_por_defecto=False
    )

    if not confirmar:
        print("Eliminación cancelada.")
        return

    try:
        eliminar_instancia(ruta)
        print(f"Instancia '{nombre}' eliminada correctamente.")
    except Exception as e:
        print(f"Error al eliminar la instancia: {e}")


# ==============================================================================

# MENÚ PRINCIPAL ===============================================================


def mostrar_menu() -> None:
    print("\n" + "=" * 55)
    print("              GESTOR DE INSTANCIAS")
    print("=" * 55)
    print("1. Generar una nueva instancia")
    print("2. Mostrar resumen de una instancia")
    print("3. Listar instancias disponibles")
    print("4. Eliminar una instancia")
    print("0. Salir")


def main() -> None:
    BASE_INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        mostrar_menu()
        opcion = input("\nSelecciona una opción: ").strip()

        if opcion == "1":
            opcion_generar_instancia()
        elif opcion == "2":
            opcion_resumen_instancia()
        elif opcion == "3":
            opcion_listar_instancias()
        elif opcion == "4":
            opcion_eliminar_instancia()
        elif opcion == "0":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")


if __name__ == "__main__":
    main()
