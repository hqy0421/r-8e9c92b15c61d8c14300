# This file is part of Patsy
# Copyright (C) 2025
# See file LICENSE.txt for license information.

# Weighted ridge regression with hard linear constraints on scenario
# predictions.
#
# The problem solved here is, for coefficient vector beta (p coefficients),
# n training observations (X, y) with non-negative weights w:
#
#     min_beta  sum_i w_i * (y_i - X_i beta)^2 + lam * ||beta||^2
#
# subject to hard linear constraints on *scenario* predictions. If the
# scenario design matrix is X_s (m scenarios by p columns, built using the
# same fitted DesignInfo as the training design), and each scenario
# constraint is a linear combination of the scenario predictions with row
# vector c_j, then:
#
#     lb_j <= c_j @ (X_s beta) <= ub_j
#
# Rows where lb_j == ub_j are equality constraints. These are exact,
# non-softened constraints -- no slack variables, no penalty approximation.
# Infeasibility is reported explicitly rather than silently relaxed.

# These are made available in the patsy.* namespace:
__all__ = [
    "ConstrainedRidgeResults",
    "ConstraintStatus",
    "InfeasibleConstraintsError",
    "ScenarioConstraints",
    "constrained_ridge",
]

import numpy as np

from patsy import PatsyError
from patsy.build import build_design_matrices
from patsy.highlevel import dmatrices
from patsy.missing import NAAction
from patsy.util import have_pandas

if have_pandas:
    import pandas


# Numerical tolerances. The constraints are *hard*; these tolerances are only
# used for deciding whether a returned solution actually satisfies them, not
# for softening them during optimization.
_FEAS_TOL = 1e-8
# SLSQP becomes unreliable (spurious line-search failures at the boundary)
# when asked for a tighter stopping tolerance than roughly 1e-9. We try a
# ladder of tolerances and keep the best successful result.
_SOLVER_TOLS = (1e-9, 1e-7, 1e-6)


def _require_scipy():
    # scipy is an optional dependency of patsy itself; the constrained ridge
    # machinery needs a real QP solver, so we only require it when used.
    try:
        from scipy.optimize import linprog, minimize
    except ImportError as exc:  # pragma: no cover
        raise PatsyError(
            "constrained ridge regression requires scipy "
            "(install the 'scipy' package to use this feature)"
        ) from exc
    return linprog, minimize


class _MaskRecordingNAAction(NAAction):
    """An NAAction that behaves exactly like the standard 'drop' action, but
    also records a boolean mask of which *original* rows were kept, so callers
    can align externally-supplied vectors (weights, responses) with the rows
    that patsy actually used."""

    def __init__(self):
        NAAction.__init__(self, on_NA="drop")
        self.kept_mask = None

    def handle_NA(self, values, is_NAs, origins):
        if len(values) == 0:
            # No factors at all (e.g. an intercept-only formula): there is
            # nothing that could be missing, so every row is kept.
            self.kept_mask = None
            return NAAction.handle_NA(self, values, is_NAs, origins)
        total_mask = np.zeros(is_NAs[0].shape[0], dtype=bool)
        for is_NA in is_NAs:
            total_mask |= is_NA
        self.kept_mask = ~total_mask
        return NAAction.handle_NA(self, values, is_NAs, origins)


def _as_1d_float(values, name, n_rows):
    """Coerce a user-supplied 1-d vector (y or weights) to a float ndarray,
    validating its shape against the design-matrix row count."""
    if have_pandas and isinstance(values, pandas.Series):
        values = values.values
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            raise PatsyError(
                "%s must be 1-dimensional (got %d columns)" % (name, arr.shape[1])
            )
    if arr.ndim != 1:
        raise PatsyError("%s must be 1-dimensional" % (name,))
    if arr.shape[0] != n_rows:
        raise PatsyError(
            "%s has %d rows, but the data has %d rows" % (name, arr.shape[0], n_rows)
        )
    return arr


class InfeasibleConstraintsError(PatsyError):
    """Raised when the requested scenario constraints cannot be satisfied by
    any coefficient vector.

    The constraints are never deleted or turned into soft penalties, so an
    infeasible system is reported by raising this exception rather than by
    returning a fitted model. The exception message describes the feasibility
    check (an independent linear program) that detected the conflict.

    .. attribute:: linprog_status

       The status code returned by the feasibility linear program.

    .. attribute:: linprog_message

       The status message returned by the feasibility linear program.
    """

    def __init__(self, message, linprog_result=None):
        PatsyError.__init__(self, message)
        self.linprog_status = None
        self.linprog_message = None
        if linprog_result is not None:
            self.linprog_status = linprog_result.status
            self.linprog_message = linprog_result.message


class ConstraintStatus(object):
    """Per-constraint verification information for a fitted model.

    Attributes:

    .. attribute:: index

       The row index of this constraint in the original lower/upper bound
       arrays (constraint order is preserved, including equalities).

    .. attribute:: is_equality

       True if the lower and upper bound were equal, i.e. this is an equality
       constraint.

    .. attribute:: lower

       The requested lower bound.

    .. attribute:: upper

       The requested upper bound.

    .. attribute:: value

       The actual value of the constrained linear combination, evaluated at
       the fitted coefficients (``c_j @ X_s @ beta``).

    .. attribute:: residual

       Signed distance from the feasible region. Negative means the bound was
       violated; zero (within tolerance) means satisfied. For an equality this
       is ``min(value - lower, upper - value)``.

    .. attribute:: satisfied

       True if both bounds are respected within the feasibility tolerance.
    """

    def __init__(self, index, is_equality, lower, upper, value, tol=_FEAS_TOL):
        self.index = index
        self.is_equality = bool(is_equality)
        self.lower = float(lower)
        self.upper = float(upper)
        self.value = float(value)
        if is_equality:
            self.residual = -abs(value - lower)
        else:
            self.residual = min(value - lower, upper - value)
        self.satisfied = bool(self.residual >= -tol)

    def __repr__(self):
        kind = "equality" if self.is_equality else "inequality"
        return (
            "ConstraintStatus(index=%r, kind=%s, bound=[%r, %r], value=%r, "
            "satisfied=%r)"
            % (self.index, kind, self.lower, self.upper, self.value, self.satisfied)
        )


class ScenarioConstraints(object):
    """Hard linear constraints on scenario predictions.

    A scenario constraint bounds a *linear combination* of the predictions
    made on rows of a separate scenario data table. If :math:`X_s` is the
    scenario design matrix (built from the same :class:`DesignInfo` as the
    training design) and row :math:`c_j` of `linear_combinations` gives the
    weights of the combination, then constraint :math:`j` is

    .. math::

       lb_j \\le c_j^\\top X_s \\beta \\le ub_j

    A row with ``lb_j == ub_j`` is treated as an **equality** constraint.

    The scenario data is only ever evaluated through a *fitted* design (via
    :func:`build_design_matrices` on an existing :class:`DesignInfo`). It is
    never passed to :func:`design_matrix_builders`, so scenario rows cannot
    influence learned categorical levels, stateful-transform means, spline
    knots, or any other part of the fitted formula scheme.

    :arg scenario_data: A dict-like data object (e.g. a
      :class:`pandas.DataFrame`) providing the variables named in the fitted
      formula. It must not contain missing values.
    :arg lower: 1-d array of lower bounds, one entry per constraint.
    :arg upper: 1-d array of upper bounds, one entry per constraint. Must have
      the same length as `lower`. Entries equal to the corresponding lower
      bound define equality constraints; entries with ``upper < lower`` are an
      error.
    :arg linear_combinations: Optional 2-d array with one row per constraint
      and one column per scenario row. Row ``j`` gives the coefficients of the
      linear combination over scenario predictions. If omitted, the identity
      matrix is used, i.e. each constraint directly bounds one scenario
      prediction (so there must be exactly as many constraints as scenarios).
    """

    def __init__(self, scenario_data, lower, upper, linear_combinations=None):
        self.scenario_data = scenario_data
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("scenario bounds must be 1-dimensional")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper bounds must have the same length")
        if lower.size == 0:
            raise ValueError("at least one scenario constraint is required")
        if np.any(upper < lower):
            raise ValueError(
                "upper bounds must be greater than or equal to lower bounds"
            )
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("scenario bounds must all be finite")
        self.lower = lower
        self.upper = upper
        self.n_constraints = lower.shape[0]
        if linear_combinations is None:
            linear_combinations = None
        else:
            C = np.asarray(linear_combinations, dtype=float)
            if C.ndim != 2:
                raise ValueError("linear_combinations must be 2-dimensional")
            if C.shape[0] != self.n_constraints:
                raise ValueError(
                    "linear_combinations has %d rows but there are %d constraints"
                    % (C.shape[0], self.n_constraints)
                )
            if np.any(~np.isfinite(C)):
                raise ValueError("linear_combinations must contain only finite values")
            linear_combinations = C
        self.linear_combinations = linear_combinations
        # Filled in by .build():
        self._coef_matrix = None
        self._equality_mask = self.lower == self.upper

    def build(self, design_info):
        """Evaluate the scenario data against an already-fitted
        :class:`DesignInfo` and translate the bounds into coefficient-space
        constraints ``A beta in [lower, upper]``.

        This never fits anything: categorical levels, means and knots all come
        from `design_info`, which was learned on the training data only.
        """
        (scenario_matrix,) = build_design_matrices(
            [design_info], self.scenario_data, NA_action="raise"
        )
        Xs = np.asarray(scenario_matrix, dtype=float)
        if self.linear_combinations is None:
            if self.n_constraints != Xs.shape[0]:
                raise ValueError(
                    "without linear_combinations there must be one constraint "
                    "per scenario row (got %d constraints and %d scenario rows)"
                    % (self.n_constraints, Xs.shape[0])
                )
            C = np.eye(Xs.shape[0])
        else:
            if self.linear_combinations.shape[1] != Xs.shape[0]:
                raise ValueError(
                    "linear_combinations has %d columns but there are %d "
                    "scenario rows" % (self.linear_combinations.shape[1], Xs.shape[0])
                )
            C = self.linear_combinations
        # A beta = C @ Xs @ beta are the constrained linear combinations.
        self._coef_matrix = C @ Xs
        self.scenario_design_matrix = scenario_matrix
        return self

    @property
    def coef_matrix(self):
        """The matrix A such that each constraint is ``lb <= A_j beta <= ub``.
        Available after :meth:`build` has been called."""
        if self._coef_matrix is None:
            raise ValueError("ScenarioConstraints.build() has not been called yet")
        return self._coef_matrix

    @property
    def equality_mask(self):
        """Boolean array marking which constraints are equalities
        (``lower == upper``)."""
        return self._equality_mask


def _check_feasible(A, lower, upper, equality_mask, linprog):
    """Independent feasibility check via a linear program. Constraints are
    never softened here either: we ask whether *any* beta exists at all that
    satisfies them exactly. Returns (feasible, linprog_result)."""
    p = A.shape[1]
    # Variables: beta (p), plus non-negative surplus/slack variables for the
    # inequality rows. Equalities get no slack.
    ineq_idx = np.where(~equality_mask)[0]
    eq_idx = np.where(equality_mask)[0]
    n_ineq = ineq_idx.size
    n_vars = p + 2 * n_ineq

    c = np.zeros(n_vars)
    # Minimize the total slack, so a zero-objective optimum proves feasibility.
    c[p:] = 1.0

    rows = []
    lbs = []
    ubs = []
    # Equality rows: A_eq beta = b_eq exactly.
    if eq_idx.size:
        rows.append(np.hstack([A[eq_idx], np.zeros((eq_idx.size, 2 * n_ineq))]))
        lbs.append(lower[eq_idx])
        ubs.append(upper[eq_idx])
    # Inequality rows: A_j beta - s_j + t_j in [lb_j, ub_j], s_j, t_j >= 0.
    if n_ineq:
        slack_block = np.zeros((n_ineq, 2 * n_ineq))
        for k in range(n_ineq):
            slack_block[k, 2 * k] = -1.0
            slack_block[k, 2 * k + 1] = 1.0
        rows.append(np.hstack([A[ineq_idx], slack_block]))
        lbs.append(lower[ineq_idx])
        ubs.append(upper[ineq_idx])

    A_bounds = np.vstack(rows)
    lb_all = np.concatenate(lbs)
    ub_all = np.concatenate(ubs)
    # Unbounded beta entries, non-negative surplus/slack entries.
    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n_ineq)
    # linprog's A_eq is exact equality, but we have ranged rows
    # lb <= A_bounds z <= ub. Encode them with two one-sided inequalities;
    # the slack variables alone are minimized, so a zero objective proves a
    # feasible beta exists and a strictly positive one proves infeasibility.
    result = linprog(
        c,
        A_ub=np.vstack([-A_bounds, A_bounds]),
        b_ub=np.concatenate([-lb_all, ub_all]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        # Numerical failure is not the same as a proven infeasibility.
        if result.status == 2:
            return False, result
        raise PatsyError(
            "feasibility check failed (status %s): %s" % (result.status, result.message)
        )
    feasible = result.fun <= max(_FEAS_TOL, 1e-7 * (1.0 + np.abs(lb_all).sum()))
    return feasible, result


class ConstrainedRidgeResults(object):
    """The fitted weighted ridge model and everything needed to audit it.

    Attributes:

    .. attribute:: coefs

       1-d ndarray of fitted coefficients, in the column order of
       :attr:`design_info`.

    .. attribute:: design_info

       The :class:`DesignInfo` learned on the training data. This is the
       *real* design info produced by the formula machinery; it is reused
       unchanged for every prediction.

    .. attribute:: lambda

       The (strictly positive) ridge penalty that was used.

    .. attribute:: weights

       The 1-d weights as actually used in the fit, aligned to the rows that
       survived missing-value and zero-weight filtering.

    .. attribute:: fitted_rows

       Integer ndarray of original-row indices that participated in learning
       (non-missing response/features/weights and non-zero weight).

    .. attribute:: n_obs

       Number of rows that actually contributed to the objective.

    .. attribute:: scenario_predictions

       1-d ndarray of the actual predictions on every scenario row, evaluated
       at the fitted coefficients.

    .. attribute:: constraint_values

       1-d ndarray of the actual constrained linear combinations (one per
       constraint), in constraint order.

    .. attribute:: constraint_status

       List of :class:`ConstraintStatus` objects, one per constraint, giving
       bounds, achieved values, residuals and satisfaction flags.

    .. attribute:: feasible

       True if every constraint is satisfied within the feasibility tolerance.

    .. attribute:: objective

       The achieved value of ``sum w (y - X beta)^2 + lambda ||beta||^2``.

    .. attribute:: result

       The raw :class:`scipy.optimize.OptimizeResult` from the solver.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def predict(self, new_data, NA_action="passthrough", return_type="matrix"):
        """Predict on new data using the same fitted formula scheme.

        :arg new_data: Dict-like data object with the variables named in the
          fitted formula.
        :arg NA_action: By default ``"passthrough"``: rows whose formula
          features are missing are kept in their original positions and their
          predictions are returned as ``NaN``. Pass ``"drop"`` to remove such
          rows instead (the standard patsy behavior), or ``"raise"`` to error
          out.
        :arg return_type: ``"matrix"`` (default) returns an ndarray (with a
          ``.design_info`` reference when a :class:`DesignMatrix` is produced);
          ``"dataframe"`` returns a :class:`pandas.Series` indexed like the
          input.
        :returns: Predictions in the original row order. With
          ``NA_action="passthrough"`` the output always has the same number of
          rows as the input, including rows with missing inputs (NaNs) and
          preserving duplicate index entries.
        """
        if return_type not in ("matrix", "dataframe"):
            raise PatsyError("return_type must be 'matrix' or 'dataframe'")
        if return_type == "dataframe" and not have_pandas:
            raise PatsyError(
                "return_type='dataframe' was requested, but pandas is not installed"
            )
        matrix_type = "dataframe" if return_type == "dataframe" else "matrix"
        if NA_action == "passthrough":
            recorder = _MaskRecordingNAAction()
            (X_new,) = build_design_matrices(
                [self.design_info],
                new_data,
                NA_action=recorder,
                return_type=matrix_type,
            )
            kept = recorder.kept_mask
            preds_kept = np.asarray(X_new) @ self.coefs
            if kept is None:
                # No factors (e.g. an intercept-only formula): nothing can be
                # missing, so every produced row is a real prediction.
                kept = np.ones(preds_kept.shape[0], dtype=bool)
            if return_type == "dataframe":
                # Rebuild on the full input index by position rather than
                # label, so duplicate index entries stay correctly aligned
                # and dropped rows reappear as NaN.
                out = pandas.Series(np.nan, index=_input_index(new_data, kept.size))
                out.values[kept] = preds_kept
                out.design_info = self.design_info
                return out
            out = np.full(kept.shape[0], np.nan, dtype=float)
            out[kept] = preds_kept
            return _PredictionArray(out, self.design_info)
        (X_new,) = build_design_matrices(
            [self.design_info],
            new_data,
            NA_action=NA_action,
            return_type=matrix_type,
        )
        preds = np.asarray(X_new) @ self.coefs
        if return_type == "dataframe":
            out = pandas.Series(preds, index=X_new.index)
            out.design_info = self.design_info
            return out
        return _PredictionArray(preds, self.design_info)


def _input_index(data, n_rows):
    if have_pandas and isinstance(data, pandas.DataFrame):
        return data.index
    return np.arange(n_rows)


class _PredictionArray(np.ndarray):
    """A 1-d ndarray of predictions that remembers the DesignInfo that
    produced them. Behaves as an ordinary ndarray otherwise."""

    def __new__(cls, input_array, design_info):
        obj = np.asarray(input_array, dtype=float).view(cls)
        obj.design_info = design_info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.design_info = getattr(obj, "design_info", None)


def constrained_ridge(
    formula_like,
    data,
    lam,
    *,
    weights=None,
    scenario_constraints=None,
    scenario_data=None,
    scenario_lower=None,
    scenario_upper=None,
    scenario_linear_combinations=None,
    eval_env=0,
):
    """Fit a weighted ridge regression with hard constraints on scenario
    predictions.

    The objective minimized is

    .. math::

       \\sum_i w_i (y_i - X_i \\beta)^2 + \\lambda \\|\\beta\\|^2

    over the rows that participate in learning. A row is excluded when its
    response, any formula feature, or its weight is missing; rows with zero
    weight are also excluded (they do not contribute to the objective). All
    filtering happens *after* the :class:`DesignInfo` has been learned, so the
    real design info -- categorical levels, stateful-transform means, spline
    knots and so on -- is always retained and reused for prediction.

    :arg formula_like: A formula specification accepted by
      :func:`dmatrices` (e.g. the string ``"y ~ x1 + x2"``). It must define a
      single response column.
    :arg data: A dict-like data object holding the training variables.
    :arg lam: Ridge penalty :math:`\\lambda`. Must be explicitly positive
      (``lam > 0``); zero, negative or non-finite values are rejected.
    :arg weights: Optional 1-d vector of non-negative row weights, aligned to
      the rows of `data`. Missing weights exclude the row from learning; zero
      weights do the same. If omitted, every (complete) row has weight 1.
    :arg scenario_constraints: A ready-made
      :class:`ScenarioConstraints` object. Alternatively the pieces can be
      passed directly via `scenario_data`, `scenario_lower`,
      `scenario_upper` and `scenario_linear_combinations`. Exactly one of the
      two styles must be used when constraints are wanted. Fitting without any
      scenario constraints is also allowed and reduces to ordinary weighted
      ridge regression.
    :arg scenario_data: Scenario table used only when
      `scenario_constraints` is not given.
    :arg scenario_lower: Lower bounds on the scenario prediction combinations.
    :arg scenario_upper: Upper bounds; equal lower/upper entries are
      equalities.
    :arg scenario_linear_combinations: Optional combination matrix; see
      :class:`ScenarioConstraints`.
    :arg eval_env: Environment depth for formula variable resolution, as in
      :func:`dmatrix`.
    :returns: A :class:`ConstrainedRidgeResults` object.
    :raises InfeasibleConstraintsError: if the constraints cannot be met by
      any coefficient vector. The constraints are never dropped or softened
      to make the problem feasible.
    """
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be a strictly positive finite number")
    linprog, minimize = _require_scipy()

    # Resolve the constraint specification into exactly one object (or None).
    inline_given = scenario_data is not None or scenario_lower is not None
    if scenario_constraints is not None and inline_given:
        raise ValueError(
            "pass either scenario_constraints= or the individual scenario_* "
            "arguments, not both"
        )
    if scenario_constraints is None and inline_given:
        if scenario_data is None or scenario_lower is None or scenario_upper is None:
            raise ValueError(
                "scenario_data, scenario_lower and scenario_upper must all be "
                "provided together"
            )
        scenario_constraints = ScenarioConstraints(
            scenario_data,
            scenario_lower,
            scenario_upper,
            linear_combinations=scenario_linear_combinations,
        )
    if scenario_linear_combinations is not None and scenario_constraints is None:
        raise ValueError(
            "scenario_linear_combinations requires scenario_data/scenario_lower/"
            "scenario_upper as well"
        )

    # Build the training design the ordinary way. dmatrices performs the full
    # stateful fitting pass (category sniffing, means, knots) on the training
    # data and returns the real DesignInfo. The recording NA action drops rows
    # with missing responses or formula features, and remembers which original
    # rows survived, so externally supplied weights can be aligned to them.
    recorder = _MaskRecordingNAAction()
    y_dm, X_dm = dmatrices(formula_like, data, eval_env=eval_env, NA_action=recorder)
    design_info = X_dm.design_info
    if y_dm.shape[1] != 1:
        raise PatsyError(
            "constrained ridge expects a single response column, got %d"
            % (y_dm.shape[1],)
        )
    feature_kept = recorder.kept_mask
    if feature_kept is None:
        # No evaluators at all (e.g. intercept-only formula without a data
        # frame): nothing could be missing, so every row was kept.
        n_orig = X_dm.shape[0]
        feature_kept = np.ones(n_orig, dtype=bool)
    else:
        n_orig = feature_kept.shape[0]
    y_kept = _as_1d_float(np.asarray(y_dm), "response", X_dm.shape[0])
    X_kept = np.asarray(X_dm, dtype=float)

    if weights is None:
        w_full = np.ones(n_orig, dtype=float)
    else:
        w_full = _as_1d_float(weights, "weights", n_orig)
        if np.any(w_full < 0):
            raise ValueError("weights must be non-negative")

    # Rows with missing response or formula features were already removed by
    # the design build. Among the survivors, missing or zero weights also keep
    # a row out of the learning objective.
    w_kept = w_full[feature_kept]
    learn_in_kept = (~np.isnan(w_kept)) & (w_kept > 0.0)
    fitted_rows = np.where(feature_kept)[0][learn_in_kept]
    if fitted_rows.size == 0:
        raise PatsyError(
            "no rows remain after removing missing and zero-weight rows; "
            "cannot fit the model"
        )
    X = X_kept[learn_in_kept]
    y = y_kept[learn_in_kept]
    w = w_kept[learn_in_kept]

    # Translate the scenario constraints into coefficient-space constraints
    # using the *fitted* design info only.
    if scenario_constraints is not None:
        scenario_constraints.build(design_info)
        A = scenario_constraints.coef_matrix
        lower = scenario_constraints.lower
        upper = scenario_constraints.upper
        equality_mask = scenario_constraints.equality_mask
        if A.shape[1] != X.shape[1]:
            raise PatsyError(
                "scenario constraint matrix has %d columns but the design has "
                "%d coefficients" % (A.shape[1], X.shape[1])
            )
        feasible, lp_result = _check_feasible(A, lower, upper, equality_mask, linprog)
        if not feasible:
            raise InfeasibleConstraintsError(
                "the scenario constraints are infeasible: no coefficient "
                "vector can satisfy all of the lower/upper bounds (the "
                "independent feasibility linear program returned status %r: "
                "%s). No constraints were removed or relaxed."
                % (lp_result.status, lp_result.message),
                linprog_result=lp_result,
            )
    else:
        A = lower = upper = equality_mask = None

    p = X.shape[1]
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    # Objective in beta: 0.5 * ||Xw beta - yw||^2 + 0.5 * lam ||beta||^2,
    # which equals half the stated weighted SSE plus ridge term.
    H = Xw.T @ Xw + lam * np.eye(p)
    g = Xw.T @ yw

    def objective(beta):
        r = Xw @ beta - yw
        return 0.5 * (r @ r) + 0.5 * lam * (beta @ beta)

    def gradient(beta):
        return H @ beta - g

    # A strictly positive lambda makes H positive definite, so the objective
    # is strictly convex and any local optimum is the unique global optimum.
    unconstrained = np.linalg.solve(H, g)

    solver_constraints = []
    if scenario_constraints is not None:
        for j in range(A.shape[0]):
            if equality_mask[j]:
                solver_constraints.append(
                    {"type": "eq", "fun": lambda b, j=j: A[j] @ b - lower[j]}
                )
            else:
                solver_constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda b, j=j: A[j] @ b - lower[j],
                    }
                )
                solver_constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda b, j=j: upper[j] - A[j] @ b,
                    }
                )

    if not solver_constraints:
        beta = unconstrained
        raw_result = None
    else:
        # Solve the strictly convex quadratic program with SLSQP. We try a
        # ladder of stopping tolerances from the unconstrained solution, and
        # if needed restart from a provably feasible point supplied by the
        # feasibility LP. The best successful result (lowest objective) wins.
        # Constraints are never relaxed in any of these attempts.
        starts = [unconstrained]
        attempts = [(x0, ftol) for x0 in starts for ftol in _SOLVER_TOLS]
        best = None
        last_result = None
        for x0, ftol in attempts:
            raw_result = minimize(
                objective,
                x0,
                jac=gradient,
                method="SLSQP",
                constraints=solver_constraints,
                options={"ftol": ftol, "maxiter": 1000, "disp": False},
            )
            last_result = raw_result
            if raw_result.success and (best is None or raw_result.fun < best.fun):
                best = raw_result
        if best is None:
            # Every attempt from the unconstrained start failed. Ask the
            # feasibility LP for a feasible starting point and retry.
            feasible, lp_result = _check_feasible(
                A, lower, upper, equality_mask, linprog
            )
            if feasible:
                x0 = lp_result.x[:p]
                for ftol in _SOLVER_TOLS:
                    raw_result = minimize(
                        objective,
                        x0,
                        jac=gradient,
                        method="SLSQP",
                        constraints=solver_constraints,
                        options={"ftol": ftol, "maxiter": 1000, "disp": False},
                    )
                    last_result = raw_result
                    if raw_result.success and (
                        best is None or raw_result.fun < best.fun
                    ):
                        best = raw_result
        if best is None:
            raise PatsyError(
                "constrained ridge solver failed to converge: %s"
                % (last_result.message,)
            )
        raw_result = best
        beta = best.x

    # Independently audit the achieved solution against every hard bound.
    if scenario_constraints is not None:
        values = A @ beta
        statuses = [
            ConstraintStatus(j, equality_mask[j], lower[j], upper[j], values[j])
            for j in range(A.shape[0])
        ]
        all_satisfied = all(s.satisfied for s in statuses)
        if not all_satisfied:
            violated = [s for s in statuses if not s.satisfied]
            detail = "; ".join(
                "constraint %d: value=%r bound=[%r, %r]"
                % (s.index, s.value, s.lower, s.upper)
                for s in violated
            )
            raise PatsyError(
                "solver returned a solution violating hard constraints "
                "(%s); the bounds were not relaxed" % (detail,)
            )
        Xs = np.asarray(scenario_constraints.scenario_design_matrix, dtype=float)
        scenario_predictions = Xs @ beta
    else:
        values = np.empty(0)
        statuses = []
        all_satisfied = True
        scenario_predictions = np.empty(0)

    residual = Xw @ beta - yw
    achieved_objective = residual @ residual + lam * (beta @ beta)

    return ConstrainedRidgeResults(
        coefs=beta,
        design_info=design_info,
        lam=lam,
        weights=w,
        fitted_rows=fitted_rows,
        n_obs=fitted_rows.size,
        scenario_predictions=scenario_predictions,
        constraint_values=values,
        constraint_status=statuses,
        feasible=all_satisfied,
        objective=achieved_objective,
        result=raw_result,
    )
