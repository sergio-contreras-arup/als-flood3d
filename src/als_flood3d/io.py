"""Point cloud I/O: LAS/LAZ (via laspy) and plain ASCII (x y z [class])."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

LAS_SUFFIXES = {".las", ".laz"}


@dataclass
class PointCloud:
    xyz: np.ndarray  # (N, 3) float64
    classification: np.ndarray | None  # (N,) ASPRS classes, if available
    las: laspy.LasData | None = None  # original LAS, kept to preserve all attributes on write


def read_point_cloud(path: str | Path) -> PointCloud:
    path = Path(path)
    if path.suffix.lower() in LAS_SUFFIXES:
        las = laspy.read(path)
        xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        return PointCloud(xyz=xyz, classification=np.asarray(las.classification, dtype=np.int64), las=las)

    data = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError(f"{path}: expected at least 3 columns (x y z [classification]).")
    cls = data[:, 3].astype(np.int64) if data.shape[1] >= 4 else None
    return PointCloud(xyz=data[:, :3].astype(np.float64), classification=cls)


def voxel_downsample(xyz: np.ndarray, size: float) -> np.ndarray:
    """Indices of one point (the first found) per ``size``-metre voxel, in original order."""
    ijk = np.floor((xyz - xyz.min(axis=0)) / size).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def _subset(cloud: PointCloud, idx: np.ndarray) -> PointCloud:
    return PointCloud(
        xyz=cloud.xyz[idx],
        classification=None if cloud.classification is None else cloud.classification[idx],
        las=None if cloud.las is None else cloud.las[idx],
    )


def read_point_clouds(paths: list[str | Path], voxel: float | None = None) -> PointCloud:
    """Read and merge several tiles, optionally voxel-thinning each one first.

    Merging adjacent tiles before computing avoids edge effects: the 100 m
    accumulation neighbourhood of a point near a tile border spans the
    neighbouring tile.
    """
    clouds = []
    for p in paths:
        c = read_point_cloud(p)
        if voxel:
            c = _subset(c, voxel_downsample(c.xyz, voxel))
        clouds.append(c)
    if len(clouds) == 1:
        return clouds[0]

    xyz = np.vstack([c.xyz for c in clouds])
    cls = None if any(c.classification is None for c in clouds) else np.concatenate([c.classification for c in clouds])
    las = None
    if all(c.las is not None for c in clouds) and len({c.las.point_format.id for c in clouds}) == 1:
        first = clouds[0].las
        header = laspy.LasHeader(point_format=first.header.point_format, version=first.header.version)
        header.offsets = xyz.min(axis=0)
        header.scales = first.header.scales
        for vlr in first.header.vlrs:  # keep CRS (GeoTIFF keys / WKT)
            header.vlrs.append(vlr)
        las = laspy.LasData(header)
        n = len(xyz)
        las.points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
        start = 0
        for c in clouds:
            sl = slice(start, start + len(c.xyz))
            for dim in first.point_format.dimension_names:
                if dim not in ("X", "Y", "Z"):
                    las[dim][sl] = c.las[dim]
            start = sl.stop
        las.x, las.y, las.z = xyz.T
    return PointCloud(xyz=xyz, classification=cls, las=las)


def rgb_of(cloud: PointCloud) -> np.ndarray | None:
    """(N, 3) uint8 colours if the LAS has RGB (16-bit scaled down), else None."""
    las = cloud.las
    if las is None or "red" not in las.point_format.dimension_names:
        return None
    rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.uint32)
    if rgb.max() > 255:
        rgb >>= 8
    return rgb.astype(np.uint8)


def write_enriched(path: str | Path, cloud: PointCloud, fields: dict[str, np.ndarray]) -> None:
    """Write the cloud plus one float32 scalar field per entry of ``fields``.

    LAS/LAZ output stores them as extra dimensions (readable in CloudCompare,
    QGIS, PDAL...). Any other suffix writes a space-delimited text file.
    """
    path = Path(path)
    if path.suffix.lower() in LAS_SUFFIXES:
        if cloud.las is not None:
            las = cloud.las
        else:
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.offsets = cloud.xyz.min(axis=0)
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x, las.y, las.z = cloud.xyz.T
            if cloud.classification is not None:
                las.classification = cloud.classification.astype(np.uint8)
        existing = set(las.point_format.dimension_names)
        new = [laspy.ExtraBytesParams(name=k, type=np.float32) for k in fields if k not in existing]
        if new:
            las.add_extra_dims(new)
        for k, v in fields.items():
            las[k] = np.asarray(v, dtype=np.float32)
        las.write(path)
        return

    cols = [cloud.xyz]
    names = ["x", "y", "z"]
    if cloud.classification is not None:
        cols.append(cloud.classification[:, None])
        names.append("classification")
    cols.extend(np.asarray(v)[:, None] for v in fields.values())
    names.extend(fields)
    np.savetxt(path, np.hstack(cols), fmt="%.8g", header=" ".join(names), comments="//")
