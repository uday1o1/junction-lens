"""Host capability inspection."""

from junctionlens.doctor.models import DoctorReport
from junctionlens.doctor.service import run_doctor

__all__ = ["DoctorReport", "run_doctor"]
