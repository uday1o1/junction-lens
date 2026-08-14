"""Frozen repository-owned synthetic scene definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

Point = tuple[float, float, float]
Box = tuple[float, float, float, float]


class SceneKind(StrEnum):
    """Mandatory V1 synthetic road-graph shapes."""

    STRAIGHT_CONTROL = "straight-control"
    MERGE = "merge"
    SPLIT = "split"
    INTERSECTION_CROSSWALK = "intersection-crosswalk"


class CorruptionKind(StrEnum):
    """Controlled valid-graph faults emitted beside perfect predictions."""

    DROP_CONTROL = "drop-control"
    BREAK_TOPOLOGY = "break-topology"
    SHIFT_LANE = "shift-lane"


@dataclass(frozen=True, slots=True)
class LaneSpec:
    """One directed lane centerline and its semantic role."""

    key: str
    anchors: tuple[Point, ...]
    intersection_or_connector: bool = False


@dataclass(frozen=True, slots=True)
class ControlSpec:
    """One front-camera control and its governed lane keys."""

    key: str
    normalized_box: Box
    applies_to: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AreaSpec:
    """One road-area geometry with a frozen two-class category index."""

    key: str
    points: tuple[Point, ...]
    category_index: int


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """Declarative synthetic scene used to generate persisted V1 graphs."""

    kind: SceneKind
    lanes: tuple[LaneSpec, ...]
    successors: tuple[tuple[str, str], ...]
    controls: tuple[ControlSpec, ...] = ()
    areas: tuple[AreaSpec, ...] = ()
    frame_count: int = 1
    mandatory_shapes: tuple[str, ...] = ()


def scene_specs() -> tuple[SceneSpec, ...]:
    """Return the ordered frozen V1 synthetic corpus specification."""
    return (
        SceneSpec(
            kind=SceneKind.STRAIGHT_CONTROL,
            lanes=(LaneSpec("main", ((5.0, 0.0, 0.0), (35.0, 0.0, 0.0))),),
            successors=(),
            controls=(ControlSpec("signal", (0.45, 0.15, 0.55, 0.30), ("main",)),),
            frame_count=2,
            mandatory_shapes=("road", "control", "temporal-ego-motion"),
        ),
        SceneSpec(
            kind=SceneKind.MERGE,
            lanes=(
                LaneSpec("upper-in", ((5.0, 3.0, 0.0), (20.0, 0.75, 0.0))),
                LaneSpec("lower-in", ((5.0, -3.0, 0.0), (20.0, -0.75, 0.0))),
                LaneSpec("out", ((20.0, 0.0, 0.0), (42.0, 0.0, 0.0))),
            ),
            successors=(("upper-in", "out"), ("lower-in", "out")),
            mandatory_shapes=("road", "merge"),
        ),
        SceneSpec(
            kind=SceneKind.SPLIT,
            lanes=(
                LaneSpec("in", ((5.0, 0.0, 0.0), (20.0, 0.0, 0.0))),
                LaneSpec("upper-out", ((20.0, 0.75, 0.0), (42.0, 7.0, 0.0))),
                LaneSpec("lower-out", ((20.0, -0.75, 0.0), (42.0, -7.0, 0.0))),
            ),
            successors=(("in", "upper-out"), ("in", "lower-out")),
            mandatory_shapes=("road", "split"),
        ),
        SceneSpec(
            kind=SceneKind.INTERSECTION_CROSSWALK,
            lanes=(
                LaneSpec("eastbound", ((5.0, 0.0, 0.0), (45.0, 0.0, 0.0)), True),
                LaneSpec("northbound", ((25.0, -20.0, 0.0), (25.0, 20.0, 0.0)), True),
                LaneSpec(
                    "left-turn",
                    ((12.0, 0.0, 0.0), (22.0, 1.5, 0.0), (25.0, 10.0, 0.0)),
                    True,
                ),
            ),
            successors=(("eastbound", "left-turn"),),
            controls=(
                ControlSpec("intersection-signal", (0.42, 0.12, 0.50, 0.28), ("eastbound",)),
            ),
            areas=(
                AreaSpec(
                    "crosswalk",
                    (
                        (21.5, -6.0, 0.0),
                        (24.0, -6.0, 0.0),
                        (24.0, 6.0, 0.0),
                        (21.5, 6.0, 0.0),
                        (21.5, -6.0, 0.0),
                    ),
                    0,
                ),
            ),
            mandatory_shapes=("road", "intersection", "control", "crosswalk"),
        ),
    )
