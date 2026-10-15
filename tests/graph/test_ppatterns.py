from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pandas.testing import assert_frame_equal

from squidpy._constants._pkg_constants import Key
from squidpy.gr import co_occurrence, spatial_autocorr
from squidpy.gr._ppatterns import _co_occurrence_helper, _find_min_max, _occur_count

MORAN_K = "moranI"
GEARY_C = "gearyC"


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_seq_par(dummy_adata: AnnData, mode: str):
    """Check whether spatial autocorr results are the same for seq. and parallel computation."""
    spatial_autocorr(dummy_adata, mode=mode)
    dummy_adata.var["highly_variable"] = np.random.choice([True, False], size=dummy_adata.var_names.shape)
    df = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=1, rng=np.random.default_rng(42), n_perms=50)
    df_parallel = spatial_autocorr(
        dummy_adata, mode=mode, copy=True, n_jobs=2, rng=np.random.default_rng(42), n_perms=50
    )

    idx_df = df.index.values
    idx_adata = dummy_adata[:, dummy_adata.var.highly_variable.values].var_names.values

    if mode == "moran":
        UNS_KEY = MORAN_K
    elif mode == "geary":
        UNS_KEY = GEARY_C
    assert UNS_KEY in dummy_adata.uns.keys()
    assert "pval_sim_fdr_bh" in df
    assert "pval_norm_fdr_bh" in dummy_adata.uns[UNS_KEY]
    assert dummy_adata.uns[UNS_KEY].columns.shape == (4,)
    assert df.columns.shape == (9,)
    # test pval_norm same
    # will need to increase the tolerance because numba parallel computations might not be exactly the same
    # these pval_norms don't use the seed anyway so the difference is not due to the seed
    # see https://github.com/scverse/squidpy/issues/1030 for more details
    np.testing.assert_allclose(df["pval_norm"].values, df_parallel["pval_norm"].values, atol=1e-12)
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    # each permutation now gets its own seed (spawned from a SeedSequence), so the
    # simulated p-values no longer depend on how the permutations are split across jobs
    np.testing.assert_allclose(df["pval_sim"].values, df_parallel["pval_sim"].values, atol=1e-12)
    np.testing.assert_allclose(df["pval_z_sim"].values, df_parallel["pval_z_sim"].values, atol=1e-12)
    np.testing.assert_allclose(df["var_sim"].values, df_parallel["var_sim"].values, atol=1e-12)


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_spatial_autocorr_reproducibility(dummy_adata: AnnData, n_jobs: int, mode: str):
    """Check spatial autocorr reproducibility results."""
    rng = np.random.RandomState(42)
    spatial_autocorr(dummy_adata, mode=mode)
    dummy_adata.var["highly_variable"] = rng.choice([True, False], size=dummy_adata.var_names.shape)
    # seed will work only when multiprocessing/loky
    df_1 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=n_jobs, rng=np.random.default_rng(42), n_perms=50)
    df_2 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=n_jobs, rng=np.random.default_rng(42), n_perms=50)

    idx_df = df_1.index.values
    idx_adata = dummy_adata[:, dummy_adata.var["highly_variable"].values].var_names.values

    if mode == "moran":
        UNS_KEY = MORAN_K
    elif mode == "geary":
        UNS_KEY = GEARY_C
    assert UNS_KEY in dummy_adata.uns.keys()
    # assert fdr correction in adata.uns
    assert "pval_sim_fdr_bh" in df_1
    assert "pval_norm_fdr_bh" in dummy_adata.uns[UNS_KEY]
    # test pval_norm same
    # will need to increase the tolerance because numba parallel computations might not be exactly the same
    # see https://github.com/scverse/squidpy/issues/1030 for more details about the tolerance
    # these pval_norms don't use the seed anyway so the difference is not due to the seed
    np.testing.assert_allclose(df_1["pval_norm"].values, df_2["pval_norm"].values, atol=1e-12)
    np.testing.assert_allclose(df_1["var_norm"].values, df_2["var_norm"].values, atol=1e-12)
    assert dummy_adata.uns[UNS_KEY].columns.shape == (4,)
    assert df_2.columns.shape == (9,)
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df_1.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    assert_frame_equal(df_1, df_2)


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_n_jobs_invariance(dummy_adata: AnnData, mode: str):
    """The number of workers must not change the permutation-based results (seed spawned per permutation)."""
    kw = {"mode": mode, "copy": True, "rng": 42, "n_perms": 50}
    df_serial = spatial_autocorr(dummy_adata, n_jobs=1, **kw)
    df_parallel = spatial_autocorr(dummy_adata, n_jobs=2, **kw)

    # align on the gene index in case the stat-based sort order ties differently
    df_parallel = df_parallel.loc[df_serial.index]
    for col in ["pval_sim", "pval_z_sim", "var_sim"]:
        np.testing.assert_allclose(df_serial[col].values, df_parallel[col].values, atol=1e-12)


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_var_norm_formula(dummy_adata: AnnData, mode: str):
    """Analytic ``var_norm`` must use the variance matching the chosen statistic.

    Regression test for #1183: Geary's C and Moran's I have different sampling
    variances under the normality assumption (Cliff & Ord 1981). Reusing Moran's
    variance for Geary's C produced a miscalibrated analytic p-value.
    """
    from sklearn.preprocessing import normalize

    from squidpy.gr._ppatterns import _g_moments

    uns_key = MORAN_K if mode == "moran" else GEARY_C
    spatial_autocorr(dummy_adata, mode=mode, transformation=True, n_perms=None, rng=np.random.default_rng(0))
    var_norm = float(dummy_adata.uns[uns_key]["var_norm"].iloc[0])

    # Reconstruct the exact (row-standardised) weight matrix the routine used.
    g = dummy_adata.obsp["spatial_connectivities"].copy()
    normalize(g, norm="l1", axis=1, copy=False)
    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
    moran_var = (n * n * s1 - n * s2 + 3 * s02) / ((n - 1) * (n + 1) * s02) - (1.0 / (n - 1)) ** 2
    geary_var = ((2 * s1 + s2) * (n - 1) - 4 * s02) / (2 * (n + 1) * s02)

    expected = moran_var if mode == "moran" else geary_var
    np.testing.assert_allclose(var_norm, expected, rtol=1e-10)
    if mode == "geary":
        # the two formulas differ here, so the test would fail if Moran's were reused
        assert not np.isclose(geary_var, moran_var, rtol=1e-3)


@pytest.mark.parametrize(
    "attr,layer,genes",
    [
        ("X", None, None),
        ("obs", None, None),
        ("obs", None, "foo"),
        ("obsm", "spatial", None),
        ("obsm", "spatial", [1, 0]),
    ],
)
def test_spatial_autocorr_attr(dummy_adata: AnnData, attr: Literal["X", "obs", "obsm"], layer: str, genes: Any):
    if attr == "obs":
        if isinstance(genes, str):
            dummy_adata.obs[genes] = np.random.RandomState(42).normal(size=(dummy_adata.n_obs,))
            index = [genes]
        else:
            index = dummy_adata.obs.select_dtypes(include=np.number).columns
    elif attr == "X":
        index = dummy_adata.var_names if genes is None else genes
    elif attr == "obsm":
        index = np.arange(dummy_adata.obsm[layer].shape[1]) if genes is None else genes

    spatial_autocorr(dummy_adata, attr=attr, mode="moran", layer=layer, genes=genes)

    df = dummy_adata.uns[MORAN_K]
    np.testing.assert_array_equal(np.isfinite(df), True)
    np.testing.assert_array_equal(sorted(df.index), sorted(index))


def test_co_occurrence(adata: AnnData):
    """
    check co_occurrence score and shape
    """
    co_occurrence(adata, cluster_key="leiden")

    # assert occurrence in adata.uns
    assert "leiden_co_occurrence" in adata.uns.keys()
    assert "occ" in adata.uns["leiden_co_occurrence"].keys()
    assert "interval" in adata.uns["leiden_co_occurrence"].keys()

    # assert shapes
    arr = adata.uns["leiden_co_occurrence"]["occ"]
    assert arr.ndim == 3
    assert arr.shape[2] == 49
    assert arr.shape[1] == arr.shape[0] == adata.obs["leiden"].unique().shape[0]


def test_co_occurrence_reproducibility(adata: AnnData):
    """Check co_occurrence reproducibility results."""
    arr_1, interval_1 = co_occurrence(adata, cluster_key="leiden", copy=True)
    arr_2, interval_2 = co_occurrence(adata, cluster_key="leiden", copy=True)

    np.testing.assert_array_equal(sorted(interval_1), sorted(interval_2))
    np.testing.assert_allclose(arr_1, arr_2)


def test_co_occurrence_missing_labels_raise(adata: AnnData):
    # a missing label has code -1, which the kernel would count as a phantom cluster
    adata.obs["leiden"] = adata.obs["leiden"].copy()
    adata.obs.loc[adata.obs_names[:3], "leiden"] = np.nan
    with pytest.raises(ValueError, match=r"contains missing values"):
        co_occurrence(adata, cluster_key="leiden", copy=True)


def _occur_count_reference(
    x: np.ndarray, y: np.ndarray, thresholds: np.ndarray, labs: np.ndarray, k: int
) -> np.ndarray:
    # per-threshold scan over all ordered pairs of distinct points, in IEEE float32 like the kernel
    counts = np.zeros((k, k, len(thresholds)), dtype=np.int64)
    for i in range(len(x)):
        dx = (x[i] - x).astype(np.float32)
        dy = (y[i] - y).astype(np.float32)
        d2 = (dx * dx + dy * dy).astype(np.float32)
        within = d2[:, None] <= thresholds[None, :]  # (n, l_val)
        within[i] = False
        for b in range(k):
            counts[labs[i], b] += within[labs == b].sum(axis=0)
    return counts


@pytest.mark.parametrize("layout", ["uniform", "lattice"])
@pytest.mark.parametrize("n_chunks", [1, 7])
def test_occur_count_matches_pairwise_scan(layout: str, n_chunks: int):
    rng = np.random.default_rng(0)
    n, k = 600, 5
    if layout == "lattice":  # integer coordinates give many exactly tied distances
        coords = np.array([(i * 69, j * 120) for i in range(25) for j in range(24)], dtype=np.float32)
    else:
        coords = rng.uniform(0, 1000, size=(n, 2)).astype(np.float32)
    x, y = coords[:, 0], coords[:, 1]  # strided views, as co_occurrence passes them
    labs = rng.integers(0, k, size=n).astype(np.int32)
    labs[:50] = 3  # label 4 stays rare, label 3 common
    d_min, d_max = _find_min_max(coords)
    pair_d2 = (x[:40, None] - x[None, 40:80]) ** 2 + (y[:40, None] - y[None, 40:80]) ** 2
    thresholds_sets = [
        np.linspace(d_min, d_max, 50, dtype=np.float32)[1:] ** 2,
        np.linspace(d_min, d_max, 3, dtype=np.float32)[1:] ** 2,
        np.sort(pair_d2.ravel()[:64]).astype(np.float32),  # thresholds equal to actual squared distances
        np.array([69.0, 120.0, 120.0, 138.0, 1e6], dtype=np.float32) ** 2,  # repeated thresholds
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([100.0**2, np.inf], dtype=np.float32),
    ]
    for thresholds in thresholds_sets:
        expected = _occur_count_reference(x, y, thresholds, labs, k)
        actual = _occur_count(x, y, thresholds, labs, k, n_chunks)
        np.testing.assert_array_equal(actual, expected)


def test_co_occurrence_exact_at_threshold():
    # the squared distance of this pair equals the squared threshold in float32; a fused multiply-add
    # would round it above the threshold and drop the pair from the first bin
    adata = AnnData(obs=pd.DataFrame({"ct": pd.Categorical(["a", "b"])}, index=["c0", "c1"]))
    adata.obsm["spatial"] = np.array([[285.46524, 257.36044], [204.76054, 35.444046]], dtype=np.float32)
    occ, _ = co_occurrence(adata, cluster_key="ct", copy=True, interval=np.array([0.0, 236.13583, 472.27167]))

    assert occ.shape == (2, 2, 2)
    assert occ[0, 1, 1] > 0
    np.testing.assert_array_equal(occ[..., 0], occ[..., 1])


def test_co_occurrence_helper_threshold_order():
    # _find_min_max can return min > max, which gives a descending interval; counts must follow its order
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 500, size=(300, 2)).astype(np.float32)
    labs = rng.integers(0, 3, size=300).astype(np.int32)
    ascending = np.array([0, 50, 120, 200, 400], dtype=np.float32)  # the first value is not a threshold
    descending = np.array([1000, 400, 200, 120, 50], dtype=np.float32)
    occ_asc = _co_occurrence_helper(coords[:, 0], coords[:, 1], ascending, labs, 3, 2)
    occ_desc = _co_occurrence_helper(coords[:, 0], coords[:, 1], descending, labs, 3, 2)

    assert occ_asc[0, 1, 0] != occ_asc[0, 1, -1]
    np.testing.assert_array_equal(occ_desc[..., ::-1], occ_asc)


def test_co_occurrence_unused_category(adata: AnnData):
    # the result is indexed by the categories, so an unused category gets an all-zero row and column
    expected, _ = co_occurrence(adata, cluster_key="leiden", copy=True)
    categories = list(adata.obs["leiden"].cat.categories)
    adata.obs["leiden"] = adata.obs["leiden"].cat.add_categories(["unused"])
    adata.obs["leiden"] = adata.obs["leiden"].cat.reorder_categories(["unused", *categories])
    actual, _ = co_occurrence(adata, cluster_key="leiden", copy=True)

    assert actual.shape == (expected.shape[0] + 1, expected.shape[1] + 1, expected.shape[2])
    np.testing.assert_array_equal(actual[0], 0)
    np.testing.assert_array_equal(actual[:, 0], 0)
    np.testing.assert_array_equal(actual[1:, 1:], expected)


def test_co_occurrence_nan_interval_raises(adata: AnnData):
    with pytest.raises(ValueError, match=r"NaN"):
        co_occurrence(adata, cluster_key="leiden", copy=True, interval=np.array([0.0, np.nan, 200.0]))


@pytest.mark.parametrize("size", [1, 3])
def test_co_occurrence_explicit_interval(adata: AnnData, size: int):
    minn, maxx = _find_min_max(adata.obsm[Key.obsm.spatial])
    interval = np.linspace(minn, maxx, size)
    if size == 1:
        with pytest.raises(ValueError, match=r"Expected interval to be of length"):
            _ = co_occurrence(adata, cluster_key="leiden", copy=True, interval=interval)
    else:
        _, interval_1 = co_occurrence(adata, cluster_key="leiden", copy=True, interval=interval)

        assert interval is not interval_1
        np.testing.assert_allclose(interval, interval_1)  # allclose because in the func, we use f32


def test_use_raw(dummy_adata: AnnData):
    var_names = [str(i) for i in range(10)]
    raw = dummy_adata[:, dummy_adata.var_names[: len(var_names)]].copy()
    raw.var_names = var_names
    dummy_adata.raw = raw

    df = spatial_autocorr(dummy_adata, use_raw=True, copy=True)

    np.testing.assert_equal(sorted(df.index), sorted(var_names))
