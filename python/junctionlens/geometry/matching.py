"""Deterministic rectangular Hungarian assignment."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from junctionlens.data.geometry import GeometryError


def _rows_to_columns(cost: npt.NDArray[np.float64]) -> list[int]:
    rows, columns = cost.shape
    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    column_row = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row in range(1, rows + 1):
        column_row[0] = row
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        column = 0
        while True:
            used[column] = True
            active_row = column_row[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = (
                    float(cost[active_row - 1, candidate - 1])
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta or (
                    minimum[candidate] == delta and candidate < next_column
                ):
                    delta = minimum[candidate]
                    next_column = candidate
            if not math.isfinite(delta):
                raise GeometryError("assignment has no finite augmenting path")
            for candidate in range(columns + 1):
                if used[candidate]:
                    row_potential[column_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if column_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            column_row[column] = column_row[previous]
            column = previous
            if column == 0:
                break
    result = [-1] * rows
    for column in range(1, columns + 1):
        if column_row[column] != 0:
            result[column_row[column] - 1] = column - 1
    return result


def deterministic_hungarian(cost_matrix: object) -> tuple[tuple[int, int], ...]:
    """Return minimum-cost pairs with stable row and column tie traversal."""
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise GeometryError("cost matrix must be two-dimensional")
    if not np.isfinite(cost).all():
        raise GeometryError("cost matrix contains a nonfinite value")
    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        return ()
    if rows <= columns:
        assignments = tuple((row, column) for row, column in enumerate(_rows_to_columns(cost)))
    else:
        transposed = _rows_to_columns(cost.T)
        assignments = tuple(sorted((row, column) for column, row in enumerate(transposed)))
    return assignments
