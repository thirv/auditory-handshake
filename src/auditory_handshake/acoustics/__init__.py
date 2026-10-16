"""Acoustic layer — the project's boundary against pyroomacoustics (PLAN.md §11).

Device transfer functions (``devices.py``), room presets (``room.py``) and
channel characterisation (``channel.py``) land in S4/S9/S10. All engine
dependence stays in this subpackage so simulation code never touches
``pyroomacoustics`` directly.
"""

from .channel import (
    ChannelAnalysis,
    analyze_channel,
    compute_rir,
    compute_rir_sweep,
    echo_tail_guard_s,
    sweep_rir_matrix,
)
from .devices import LinkBand, Transducer, device_capacity
from .motion import (
    SPEED_OF_SOUND_M_S,
    doppler_factor,
    measure_doppler_hz,
    moving_source,
    trajectory,
)
from .room import calculate_rt60

__all__ = [
    "ChannelAnalysis",
    "LinkBand",
    "SPEED_OF_SOUND_M_S",
    "Transducer",
    "analyze_channel",
    "calculate_rt60",
    "compute_rir",
    "compute_rir_sweep",
    "device_capacity",
    "echo_tail_guard_s",
    "doppler_factor",
    "measure_doppler_hz",
    "moving_source",
    "sweep_rir_matrix",
    "trajectory",
]
