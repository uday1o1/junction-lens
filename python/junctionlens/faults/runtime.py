"""Bounded test-only runtime fixtures for performance detector faults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AllocatorFixtureResult:
    memory_samples_bytes: tuple[int, ...]
    high_water_bytes: int
    outstanding_bytes: int
    retained_buffers: int


def run_allocator_fixture(
    *,
    frame_count: int,
    buffer_bytes: int,
    leak: bool,
    baseline_bytes: int = 0,
) -> AllocatorFixtureResult:
    """Exercise real owned buffers while bounding total test memory."""
    if not 10 <= frame_count <= 1000:
        raise ValueError("allocator fixture frame count must be within [10, 1000]")
    if not 1024 <= buffer_bytes <= 1024 * 1024:
        raise ValueError("allocator fixture buffer size must be within [1 KiB, 1 MiB]")
    if baseline_bytes < 0:
        raise ValueError("allocator fixture baseline must be nonnegative")
    retained: list[bytearray] = []
    samples = []
    high_water = baseline_bytes
    for _ in range(frame_count):
        buffer = bytearray(buffer_bytes)
        if leak:
            retained.append(buffer)
        outstanding = len(retained) * buffer_bytes
        sample = baseline_bytes + outstanding
        samples.append(sample)
        high_water = max(high_water, sample)
        if not leak:
            del buffer
    return AllocatorFixtureResult(
        memory_samples_bytes=tuple(samples),
        high_water_bytes=high_water,
        outstanding_bytes=len(retained) * buffer_bytes,
        retained_buffers=len(retained),
    )


def inject_bounded_latency(
    samples_ms: tuple[float, ...], *, added_delay_ms: float
) -> tuple[float, ...]:
    """Model one benchmark-only bounded postprocess delay without sleeping the test process."""
    if not samples_ms or not 0.0 < added_delay_ms <= 1000.0:
        raise ValueError("bounded latency injection is invalid")
    return tuple(value + added_delay_ms for value in samples_ms)


__all__ = ["AllocatorFixtureResult", "inject_bounded_latency", "run_allocator_fixture"]
