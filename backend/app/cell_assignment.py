"""Centroid-in-polygon assignment — geometric only, no statistics."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
from shapely.geometry import Point, shape


def cells_inside_polygon(
    cell_ids: Iterable[str],
    xs: np.ndarray,
    ys: np.ndarray,
    geometry_geojson: dict,
) -> list[str]:
    geom = shape(geometry_geojson)
    if not geom.is_valid:
        geom = geom.buffer(0)
    out: list[str] = []
    for cid, x, y in zip(cell_ids, xs, ys):
        if geom.contains(Point(float(x), float(y))) or geom.touches(Point(float(x), float(y))):
            out.append(str(cid))
    return out


def geometry_to_json(geom: dict) -> str:
    return json.dumps(geom)
