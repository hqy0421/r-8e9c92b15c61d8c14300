# This file is part of Patsy
# Copyright (C) 2025
# See file LICENSE.txt for license information.

# Tests for weighted ridge regression with hard scenario constraints.

import numpy as np
import pytest

from patsy import (
    ConstraintStatus,
    InfeasibleConstraintsError,
    PatsyError,
    ScenarioConstraints,
    build_design_matrices,
    constrained_ridge,
    demo_data,
    dmatrices,
    dmatrix,
)
from patsy.ridge import _MaskRecordingNAAction, _PredictionArray
from patsy.util import have_pandas

if have_pandas:
    import pandas


def _weighted_ridge_closed_form(X, y, w, lam):
    Xw = X * np.sqrt(w)[:, None]
    H = Xw.T @ Xw + lam * np.eye(X.shape[1])
    return np.linalg.solve(H, Xw.T @ (y * np.sqrt(w)))


def test_unconstrained_matches_weighted_ridge_closed_form():
    rng = np.random.RandomState(0)
    n = 50
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 - 2.0 * x1 + 0.5 * x2 + 0.1 * rng.normal(size=n)
    w = rng.uniform(0.5, 3.0, size=n)
    data = {"x1": x1, "x2": x2, "y": y}
    res = constrained_ridge("y ~ x1 + x2", data, lam=1.7, weights=w)

    y_dm, X_dm = dmatrices("y ~ x1 + x2", data)
    beta = _weighted_ridge_closed_form(np.asarray(X_dm), np.asarray(y_dm)[:, 0], w, 1.7)
    assert np.allclose(res.coefs, beta)
    assert res.feasible
    assert res.result is None
    assert res.constraint_status == []
    assert res.scenario_predictions.shape == (0,)
    assert res.constraint_values.shape == (0,)
    # The reported objective is the un-scaled weighted SSE plus ridge term.
    X = np.asarray(X_dm)
    expected_obj = np.sum(w * (y - X @ beta) ** 2) + 1.7 * beta @ beta
    assert np.isclose(res.objective, expected_obj)
    assert res.lam == 1.7
    assert res.n_obs == n


def test_row_filtering_missing_and_zero_weights():
    rng = np.random.RandomState(1)
    n = 20
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    y[2] = np.nan  # missing response
    x2 = x.copy()
    x2[3] = np.nan  # missing feature
    w = np.ones(n)
    w[0] = 0.0  # zero weight
    w[1] = np.nan  # missing weight
    data = {"x": x2, "y": y}
    res = constrained_ridge("y ~ x", data, lam=0.5, weights=w)
    excluded = {0, 1, 2, 3}
    assert set(res.fitted_rows.tolist()) == set(range(n)) - excluded
    assert res.n_obs == n - 4
    assert res.weights.shape == (n - 4,)
    assert np.all(res.weights > 0)
    # Hand-check the coefficients against the surviving rows only.
    keep = np.array([i not in excluded for i in range(n)])
    X = np.column_stack([np.ones(n), x])[keep]
    beta = _weighted_ridge_closed_form(X, y[keep], np.ones(keep.sum()), 0.5)
    assert np.allclose(res.coefs, beta)


def test_all_rows_excluded_raises():
    data = {"x": [1.0, 2.0], "y": [1.0, 2.0]}
    with pytest.raises(PatsyError):
        constrained_ridge("y ~ x", data, lam=1.0, weights=[0.0, 0.0])
    with pytest.raises(PatsyError):
        constrained_ridge("y ~ x", {"x": [1.0, 2.0], "y": [np.nan, np.nan]}, 1.0)


def test_lambda_must_be_positive():
    data = {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
    for bad in (0, 0.0, -1.0, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            constrained_ridge("y ~ x", data, lam=bad)


def test_weight_validation():
    data = {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
    with pytest.raises(ValueError):
        constrained_ridge("y ~ x", data, lam=1.0, weights=[1.0, -2.0, 1.0])
    with pytest.raises(PatsyError):
        constrained_ridge("y ~ x", data, lam=1.0, weights=[1.0, 1.0])
    # 2-d response is rejected
    yy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(PatsyError):
        constrained_ridge(("y ~ x"), {"x": [1, 2, 3], "y": yy}, 1.0)


def test_equality_constraint_is_exact():
    rng = np.random.RandomState(2)
    n = 80
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    data = {"x": x, "y": y}
    scenario = {"x": [1.0, 2.0, 3.0]}
    res = constrained_ridge(
        "y ~ x",
        data,
        0.2,
        scenario_data=scenario,
        scenario_lower=[7.0],
        scenario_upper=[7.0],
        scenario_linear_combinations=np.array([[1.0, 1.0, 1.0]]),
    )
    assert len(res.constraint_status) == 1
    status = res.constraint_status[0]
    assert status.is_equality
    assert status.satisfied
    assert np.isclose(status.value, 7.0, atol=1e-7)
    assert np.isclose(res.scenario_predictions.sum(), 7.0, atol=1e-7)
    # Actual scenario predictions correspond to the three scenario rows.
    assert res.scenario_predictions.shape == (3,)


def test_inequality_bounds_and_binding():
    rng = np.random.RandomState(3)
    n = 100
    x = rng.normal(size=n)
    y = x + 0.05 * rng.normal(size=n)
    data = {"x": x, "y": y}
    scenario = {"x": [1.0, 2.0, 3.0]}
    res = constrained_ridge(
        "y ~ x",
        data,
        0.1,
        scenario_data=scenario,
        scenario_lower=[100.0],
        scenario_upper=[200.0],
        scenario_linear_combinations=np.array([[1.0, 0.0, 0.0]]),
    )
    status = res.constraint_status[0]
    assert not status.is_equality
    assert status.satisfied
    assert status.value >= 100.0 - 1e-8
    # It binds at the lower boundary (objective would prefer a small value).
    assert np.isclose(status.value, 100.0, atol=1e-6)

    # A non-binding wide box simply contains the unconstrained prediction.
    free = constrained_ridge("y ~ x", data, 0.1)
    boxed = constrained_ridge(
        "y ~ x",
        data,
        0.1,
        scenario_data=scenario,
        scenario_lower=[-1000.0] * 3,
        scenario_upper=[1000.0] * 3,
    )
    assert np.allclose(boxed.coefs, free.coefs)


def test_identity_combination_is_default():
    data = {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.1, 2.9, 4.2]}
    scenario = {"x": [1.0, 2.0]}
    res = constrained_ridge(
        "y ~ x",
        data,
        0.1,
        scenario_data=scenario,
        scenario_lower=[0.0, 0.0],
        scenario_upper=[5.0, 5.0],
    )
    assert np.all(res.scenario_predictions >= 0.0)
    assert np.all(res.scenario_predictions <= 5.0)
    assert np.allclose(res.constraint_values, res.scenario_predictions)


def test_infeasible_constraints_raise_and_are_not_softened():
    rng = np.random.RandomState(4)
    n = 30
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    data = {"x": x, "y": y}
    scenario = {"x": [0.0, 1.0]}

    # 0 * beta cannot equal 5.
    with pytest.raises(InfeasibleConstraintsError) as excinfo:
        constrained_ridge(
            "y ~ x",
            data,
            1.0,
            scenario_data=scenario,
            scenario_lower=[5.0],
            scenario_upper=[5.0],
            scenario_linear_combinations=np.zeros((1, 2)),
        )
    assert excinfo.value.linprog_status == 2

    # Two contradictory equalities on the same linear combination.
    with pytest.raises(InfeasibleConstraintsError):
        constrained_ridge(
            "y ~ x",
            data,
            1.0,
            scenario_data=scenario,
            scenario_lower=[1.0, 2.0],
            scenario_upper=[1.0, 2.0],
            scenario_linear_combinations=np.array([[1.0, 0.0], [1.0, 0.0]]),
        )


def test_scenario_constraints_validation():
    scenario = {"x": [1.0, 2.0]}
    with pytest.raises(ValueError):
        ScenarioConstraints(scenario, [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        ScenarioConstraints(scenario, [5.0], [1.0])
    with pytest.raises(ValueError):
        ScenarioConstraints(scenario, [np.nan], [1.0])
    with pytest.raises(ValueError):
        ScenarioConstraints(scenario, [], [])
    with pytest.raises(ValueError):
        ScenarioConstraints(scenario, [1.0], [1.0], linear_combinations=np.ones((2, 2)))
    with pytest.raises(ValueError):
        ScenarioConstraints(
            scenario, [1.0], [1.0], linear_combinations=np.array([1.0, 1.0])
        )
    sc = ScenarioConstraints(scenario, [1.0], [2.0])
    assert sc.equality_mask == [False]
    with pytest.raises(ValueError):
        sc.coef_matrix  # noqa: B018  not built yet


def test_scenario_uses_fitted_design_info_only():
    rng = np.random.RandomState(5)
    n = 100
    x = rng.normal(size=n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)
    data = {"x": x, "y": y}
    baseline = constrained_ridge("y ~ center(x)", data, 1.0)
    training_mean = x.mean()

    # Scenario values are wildly outside the training range; they must not
    # change the memorized centering mean.
    scenario = {"x": [1000.0, -1000.0]}
    res = constrained_ridge(
        "y ~ center(x)",
        data,
        1.0,
        scenario_data=scenario,
        scenario_lower=[-1e9, -1e9],
        scenario_upper=[1e9, 1e9],
    )
    assert res.design_info.column_names == baseline.design_info.column_names
    expected_X = np.column_stack(
        [np.ones(2), np.array([1000.0, -1000.0]) - training_mean]
    )
    assert np.allclose(expected_X @ res.coefs, res.scenario_predictions)

    # An unknown categorical level in the scenario is an error, not a newly
    # learned category.
    g = rng.choice(["a", "b"], size=n)
    cat_data = {"x": x, "g": g, "y": y}
    with pytest.raises(PatsyError):
        constrained_ridge(
            "y ~ x + C(g)",
            cat_data,
            1.0,
            scenario_data={"x": [0.0], "g": ["zzz"]},
            scenario_lower=[0.0],
            scenario_upper=[1.0],
        )

    # Missing scenario values are rejected; scenarios never drive NA dropping.
    with pytest.raises(PatsyError):
        constrained_ridge(
            "y ~ center(x)",
            data,
            1.0,
            scenario_data={"x": [np.nan]},
            scenario_lower=[0.0],
            scenario_upper=[1.0],
        )


def test_design_info_is_retained_for_prediction():
    rng = np.random.RandomState(6)
    n = 40
    x = rng.normal(size=n)
    g = rng.choice(["a", "b"], size=n)
    y = rng.normal(size=n)
    data = {"x": x, "g": g, "y": y}
    res = constrained_ridge("y ~ x + C(g)", data, 0.4)
    # A fresh build yields an equivalent design; prediction reuses the exact
    # DesignInfo captured at fit time.
    standalone = dmatrices("y ~ x + C(g)", data)[1].design_info
    assert res.design_info.column_names == standalone.column_names
    assert res.design_info.term_names == standalone.term_names
    new_data = {"x": np.array([1.0, 2.0]), "g": ["a", "b"]}
    (X_new,) = build_design_matrices([res.design_info], new_data)
    pred = res.predict(new_data)
    assert pred.design_info is res.design_info
    assert np.allclose(pred, np.asarray(X_new) @ res.coefs)


def test_predict_passthrough_preserves_na_order_and_length():
    rng = np.random.RandomState(7)
    data = {
        "x": rng.normal(size=25),
        "g": rng.choice(["a", "b"], size=25),
        "y": rng.normal(size=25),
    }
    res = constrained_ridge("y ~ x + C(g)", data, 0.3)
    new_data = {
        "x": np.array([1.0, np.nan, 0.5, np.nan, -1.0]),
        "g": ["a", "b", "a", "a", "b"],
    }
    pred = res.predict(new_data)
    assert isinstance(pred, np.ndarray)
    assert pred.design_info is res.design_info
    assert pred.shape == (5,)
    assert np.isnan(pred[1]) and np.isnan(pred[3])
    assert np.all(np.isfinite(pred[[0, 2, 4]]))
    # The finite entries match a build on the complete rows, in original order.
    kept = build_design_matrices(
        [res.design_info],
        {"x": np.array([1.0, 0.5, -1.0]), "g": ["a", "a", "b"]},
    )[0]
    assert np.allclose(pred[[0, 2, 4]], np.asarray(kept) @ res.coefs)

    # drop and raise modes behave like standard patsy.
    dropped = res.predict(new_data, NA_action="drop")
    assert dropped.shape == (3,)
    assert np.allclose(dropped, pred[[0, 2, 4]])
    with pytest.raises(PatsyError):
        res.predict(new_data, NA_action="raise")

    # Categorical missingness is also passed through as NaN.
    cat_na = {"x": np.array([1.0, 2.0]), "g": np.array(["a", None], dtype=object)}
    pred_cat = res.predict(cat_na)
    assert pred_cat.shape == (2,)
    assert np.isfinite(pred_cat[0]) and np.isnan(pred_cat[1])


def test_predict_intercept_only_formula():
    data = {"y": np.arange(10.0)}
    res = constrained_ridge("y ~ 1", data, 1.0)
    if have_pandas:
        # A data frame lets patsy infer the number of rows even though the
        # formula has no data-dependent factors.
        new = pandas.DataFrame({"unused": [0.0, 1.0, 2.0]})
        pred = res.predict(new)
        assert pred.shape == (3,)
        assert np.allclose(pred, res.coefs[0])
    # An empty mapping cannot determine a row count, exactly like raw patsy.
    with pytest.raises(PatsyError):
        res.predict({})


if have_pandas:

    def test_predict_dataframe_index_and_duplicates():
        rng = np.random.RandomState(8)
        n = 20
        df = pandas.DataFrame(
            {
                "x": rng.normal(size=n),
                "g": rng.choice(["a", "b"], size=n),
                "y": rng.normal(size=n),
            },
            index=[i % 5 for i in range(n)],  # duplicated index labels
        )
        res = constrained_ridge("y ~ x + C(g)", df, 0.3)
        new = pandas.DataFrame(
            {"x": [1.0, np.nan, 0.5], "g": ["a", "b", "a"]},
            index=[0, 0, 0],
        )
        out = res.predict(new, return_type="dataframe")
        assert isinstance(out, pandas.Series)
        assert len(out) == 3
        assert list(out.index) == [0, 0, 0]
        assert np.isnan(out.iloc[1])
        assert out.design_info is res.design_info

        dropped = res.predict(new, NA_action="drop", return_type="dataframe")
        assert list(dropped.index) == [0, 0]
        assert len(dropped) == 2

        # Matrix mode on a frame preserves the full number of rows.
        mat = res.predict(new)
        assert isinstance(mat, _PredictionArray)
        assert mat.shape == (3,)

    def test_pandas_weights_aligned_positionally():
        df = pandas.DataFrame(
            {"x": np.arange(10.0), "y": np.arange(10.0) + 1.0},
            index=[i % 4 for i in range(10)],
        )
        w = pandas.Series(np.ones(10), index=df.index)
        w.iloc[0] = 0.0
        res = constrained_ridge("y ~ x", df, 0.2, weights=w)
        assert 0 not in res.fitted_rows
        assert res.n_obs == 9


def test_mask_recording_na_action():
    action = _MaskRecordingNAAction()
    values = [np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])]
    is_na = [np.array([False, True, False]), np.array([False, False, False])]
    out = action.handle_NA(values, is_na, [None, None])
    assert np.array_equal(action.kept_mask, [True, False, True])
    assert np.array_equal(out[0], [1.0, 3.0])


def test_constraint_status_repr_and_residual():
    s = ConstraintStatus(0, True, 2.0, 2.0, 2.0 + 1e-9)
    assert s.satisfied
    assert "equality" in repr(s)
    violated = ConstraintStatus(1, False, 0.0, 1.0, 1.5)
    assert not violated.satisfied
    assert violated.residual < 0


def test_scenario_constraints_object_can_be_passed_directly():
    rng = np.random.RandomState(9)
    n = 40
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    data = {"x": x, "y": y}
    scenario = {"x": [0.0, 1.0]}
    sc = ScenarioConstraints(scenario, [0.0, 0.0], [3.0, 3.0])
    res = constrained_ridge("y ~ x", data, 0.5, scenario_constraints=sc)
    assert np.all(res.scenario_predictions <= 3.0 + 1e-8)
    assert np.all(res.scenario_predictions >= -1e-8)
    # A already-built object carries a coefficient-space matrix A with the
    # right width.
    assert sc.coef_matrix.shape == (2, res.coefs.shape[0])


def test_categorical_formula_design_info_unchanged_by_fit():
    # The fit must not alter ordinary formula behavior: the returned design
    # info is the same one dmatrix produces standalone.
    data = demo_data("a", "x", "y", min_rows=12)
    res = constrained_ridge("y ~ C(a) + x", data, 0.8)
    standalone = dmatrix("C(a) + x", data)
    assert res.design_info.column_names == standalone.design_info.column_names
    new = demo_data("a", "x", min_rows=4)
    pred = res.predict(new)
    (X_new,) = build_design_matrices([res.design_info], new)
    assert np.allclose(pred, np.asarray(X_new) @ res.coefs)
