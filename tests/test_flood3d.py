import numpy as np
import pytest

from als_flood3d import (
    FloodRiskConfig,
    compute_flood_risk,
    geometric_features,
    ground_grid_subsample,
    runoff_potential,
    water_accumulation,
)
from als_flood3d.cli import main
from als_flood3d.io import read_point_cloud
from als_flood3d.synthetic import make_scene


@pytest.fixture(scope="module")
def scene():
    return make_scene(size=400.0, density=1.0, seed=1)


@pytest.fixture(scope="module")
def result(scene):
    return compute_flood_risk(scene.xyz, scene.classification, FloodRiskConfig(r1=100.0, r2=4.0))


# --- Eq. 1 -------------------------------------------------------------------

def test_acu_matches_brute_force_on_subsampled_ground():
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(0, 300, 3000), rng.uniform(0, 300, 3000), rng.normal(size=3000)])
    pts = np.column_stack([rng.uniform(0, 300, 200), rng.uniform(0, 300, 200), rng.normal(5, 1, 200)])
    acu = water_accumulation(pts, ground, r1=50.0, grid_cell=0, method="kdtree", clip_negative=False)
    d = np.hypot(pts[:, None, 0] - ground[None, :, 0], pts[:, None, 1] - ground[None, :, 1])
    expected = pts[:, 2] - np.where(d <= 50.0, ground[None, :, 2], np.inf).min(axis=1)
    np.testing.assert_allclose(acu, expected)


def test_grid_method_close_to_kdtree(scene):
    g = scene.xyz[scene.classification == 2]
    a_grid = water_accumulation(scene.xyz, g, method="grid")
    a_tree = water_accumulation(scene.xyz, g, method="kdtree")
    # The raster approximation shifts the neighbourhood by at most one cell.
    assert np.nanmedian(np.abs(a_grid - a_tree)) < 0.5


def test_ground_subsample_keeps_lowest_point_per_cell():
    g = np.array([[1, 1, 5.0], [2, 2, 3.0], [15, 1, 7.0], [16, 2, 9.0]])
    sub = ground_grid_subsample(g, cell=10.0)
    assert sorted(sub[:, 2].tolist()) == [3.0, 7.0]


def test_acu_lowest_in_depression(scene, result):
    cx, cy = scene.DEPRESSION_CENTER
    ground = scene.label == "ground"
    r = np.hypot(scene.xyz[:, 0] - cx, scene.xyz[:, 1] - cy)
    bottom = result.acu[ground & (r < 5)]
    rim = result.acu[ground & (r > 60) & (r < 70)]
    assert np.median(bottom) < 0.5
    assert np.median(rim) > 3.0
    assert np.median(result.acu[scene.label == "flat_roof"]) > 14.0  # roofs are "safe"


def test_acu_nan_without_ground_in_radius():
    acu = water_accumulation(np.array([[1000.0, 1000.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]), r1=100.0, method="kdtree")
    assert np.isnan(acu[0])


# --- Eq. 2-6 -----------------------------------------------------------------

def test_features_on_ideal_shapes():
    rng = np.random.default_rng(0)
    plane = np.column_stack([rng.uniform(0, 20, 4000), rng.uniform(0, 20, 4000), np.zeros(4000)])
    wall = plane[:, [0, 2, 1]]
    blob = rng.normal(size=(4000, 3))
    fp, fw, fb = (geometric_features(p, radius=2.0) for p in (plane, wall, blob))
    assert np.nanmedian(fp.verticality) < 1e-6 and np.nanmedian(fp.planarity) > 0.7
    assert np.nanmedian(fw.verticality) > 0.99
    assert np.nanmedian(fb.scattering) > 0.5 and np.nanmedian(fb.planarity) < 0.4


def test_features_match_numpy_covariance():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(500, 3))
    f = geometric_features(pts, radius=1.0)
    i = int(np.argmin(np.linalg.norm(pts, axis=1)))  # densest neighbourhood
    nb = pts[np.linalg.norm(pts - pts[i], axis=1) <= 1.0]
    w = np.linalg.eigvalsh(np.cov(nb.T, bias=True))
    np.testing.assert_allclose(f.eigenvalues[i], w, atol=1e-10)


# --- Eq. 7 -------------------------------------------------------------------

def test_runoff_formula():
    v, p, s = np.array([0.5]), np.array([0.8]), np.array([0.1])
    np.testing.assert_allclose(runoff_potential(v, p, s), 0.5**0.1 * 0.8 * 0.9)
    assert runoff_potential(np.array([0.0]), np.array([1.0]), np.array([0.0]))[0] == 0.0


def test_runoff_ranking_on_scene(scene, result):
    med = {k: np.nanmedian(result.run[scene.label == k]) for k in ("gable_roof", "flat_roof", "tree")}
    assert med["gable_roof"] > 0.5
    assert med["gable_roof"] > med["flat_roof"]
    assert med["gable_roof"] > med["tree"]


# --- CLI / IO ----------------------------------------------------------------

def test_cli_roundtrip_las(tmp_path, scene):
    import laspy

    src = tmp_path / "in.txt"
    np.savetxt(src, np.column_stack([scene.xyz, scene.classification]))
    out = tmp_path / "out.laz"
    main([str(src), str(out), "--r2", "4"])
    las = laspy.read(out)
    for dim in ("ACU", "RUN", "verticality", "planarity", "scattering"):
        assert dim in las.point_format.dimension_names
    assert len(las.points) == len(scene.xyz)
    back = read_point_cloud(out)
    np.testing.assert_array_equal(back.classification, scene.classification)


def test_viewer_export_roundtrip(tmp_path, scene, result):
    import base64
    import json
    import re

    from als_flood3d.viewer import export_html

    out = tmp_path / "v.html"
    n = export_html(out, scene.xyz, result.acu, result.run, scene.classification, max_points=5000)
    html = out.read_text()
    meta = json.loads(re.search(r'id="meta" type="application/json">(.*?)</script>', html).group(1))
    blob = base64.b64decode(re.search(r'id="data" type="application/octet-stream">(.*?)</script>', html, re.S).group(1))
    assert n == meta["n"] == 5000
    assert len(blob) == 13 * n
    assert "__DATA__" not in html and "__META__" not in html


def test_voxel_downsample_one_point_per_voxel():
    from als_flood3d.io import voxel_downsample

    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.5, 0.1, 0.1], [0.1, 0.1, 1.5]])
    assert voxel_downsample(pts, 1.0).tolist() == [0, 2, 3]
