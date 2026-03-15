from __future__ import annotations

from typing import Optional
import numpy as np

from instance_manager import ExamInstance

UNASSIGNED = -1


class ExamSolution:
    """
    Representación de una solución para la planificación de exámenes.

    Estructuras principales:
    - instance: Instancia del problema generada con "instance_manager.py"
    - slot[e]: Franja asignada al examen e.
    - room[e]: Aula asignada al examen e.
    """

    instance: ExamInstance
    slot: np.ndarray
    room: np.ndarray

    def __init__(
        self,
        instance: ExamInstance,
        slot: Optional[np.ndarray] = None,
        room: Optional[np.ndarray] = None,
    ) -> None:
        self.instance = instance

        if (slot is None) != (room is None):
            raise ValueError("slot y room deben proporcionarse ambos o ninguno.")

        if slot is None and room is None:
            self.slot = np.full(instance.n_exams, UNASSIGNED, dtype=np.int32)
            self.room = np.full(instance.n_exams, UNASSIGNED, dtype=np.int32)
        else:
            assert slot is not None
            assert room is not None
            self.slot = slot
            self.room = room

    def copy(self) -> ExamSolution:
        return ExamSolution(
            instance=self.instance,
            slot=self.slot.copy(),
            room=self.room.copy(),
        )

    def num_assigned(self) -> int:
        return int(np.sum((self.slot != UNASSIGNED) & (self.room != UNASSIGNED)))

    def is_complete(self) -> bool:
        return self.num_assigned() == self.instance.n_exams

    def is_assigned(self, exam_id: int) -> bool:
        if not self._valid_exam_id(exam_id):
            return False
        return self.slot[exam_id] != UNASSIGNED and self.room[exam_id] != UNASSIGNED

    def _valid_exam_id(self, exam_id: int) -> bool:
        return 0 <= exam_id < self.instance.n_exams

    def assign_exam(self, exam_id: int, slot: int, room: int) -> None:
        if self._valid_exam_id(exam_id):
            self.slot[exam_id] = slot
            self.room[exam_id] = room

    def unassign_exam(self, exam_id: int) -> None:
        if self._valid_exam_id(exam_id):
            self.slot[exam_id] = UNASSIGNED
            self.room[exam_id] = UNASSIGNED

    def assign_slot(self, exam_id: int, slot: int) -> None:
        if self._valid_exam_id(exam_id):
            self.slot[exam_id] = slot

    def swap_slots(self, exam_id_1: int, exam_id_2: int) -> None:
        if self._valid_exam_id(exam_id_1) and self._valid_exam_id(exam_id_2):
            self.slot[exam_id_1], self.slot[exam_id_2] = (
                self.slot[exam_id_2],
                self.slot[exam_id_1],
            )

    def __str__(self) -> str:
        used_slots = np.unique(self.slot[self.slot != UNASSIGNED])
        used_rooms = np.unique(self.room[self.room != UNASSIGNED])

        n_slots_used = len(used_slots)
        n_rooms_used = len(used_rooms)

        slot_pct = 100 * n_slots_used / self.instance.n_slots
        room_pct = 100 * n_rooms_used / self.instance.n_rooms

        return (
            "\n=== Información de la solución ===\n"
            f"Exámenes totales: {self.instance.n_exams}\n"
            f"Exámenes asignados: {self.num_assigned()}\n"
            f"Solución completa: {self.is_complete()}\n"
            f"Slots usados: {n_slots_used} / {self.instance.n_slots}     ({slot_pct:.0f}%)\n"
            f"Aulas usadas: {n_rooms_used} / {self.instance.n_rooms}     ({room_pct:.0f}%)\n"
        )
