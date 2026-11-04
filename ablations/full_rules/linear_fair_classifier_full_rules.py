import logging
import time

import numpy as np
import pyomo.environ as pyo

from sklearn.linear_model import LogisticRegression

# ignore assert warnings
# trunk-ignore-all(bandit/B101)


logger = logging.getLogger(__name__)


class SubgroupBudgetExceeded(RuntimeError):
    """Raised when exhaustive rule-subgroup enumeration exceeds the configured budget."""


class LinearFairClassifierFullRules:
    """Variant of LinearFairClassifier that replaces lazy-cut generation with an upfront,
    exhaustive enumeration of every "rule" (conjunction) subgroup.

    LinearFairClassifier (see linear_fair_classifier.py) enforces gamma-fairness against an
    exponential family of subgroups by lazily generating the single most-violating cut at
    each incumbent solution (a Gurobi callback that itself solves a MIO to find that cut).
    This class instead enumerates every subgroup once, upfront, and adds *all* of the
    corresponding fairness constraints to the model before solving - no callback needed.

    This is only tractable for subgroup_type="rule": a rule subgroup is a conjunction over a
    finite set of boolean protected features, so the set of *distinct, non-empty* subgroups is
    finite and can be enumerated with anti-monotone (Apriori-style) support pruning - a
    conjunction can only match a sample if every one of its literals does, so the moment a
    partial conjunction has zero support, every extension of it does too, and the whole branch
    can be dropped. "linear" subgroups are a continuous family (real-valued coefficients and
    threshold) and have no such finite enumeration, so this class does not support them.

    Warning: even with anti-monotone pruning, the number of realizable conjunctions can grow
    very quickly with the number of protected features (empirically, hundreds of thousands of
    subgroups even for a couple dozen protected columns with only per-group minimum-support
    pruning) - see `n_min_group` and `max_subgroups` on `find_classifier`.
    """

    def __init__(self, gamma: float, fairness_type: str = "", subgroup_type: str = "rule", sparsity_term: float = 0) -> None:
        """
        Initializes the classifier with the specified fairness constraint.
        Args:
            gamma (float): Threshold for fairness.
            fairness_type (str): Type of fairness constraint to apply.
                One of 'MSD', 'MSD0', 'SPSF', 'FPSF' or an empty string ("").
        Fairness Types:
            - MSD: Maximum Subgroup Discrepancy
            - MSD0: Maximum Subgroup Discrepancy on y=0 into FPSF
            - SPSF: Statistical Parity Subgroup Fairness
            - FPSF: False Positive Subgroup Fairness
        Group Types:
            - rule (the only type this class supports - see class docstring)
        """
        if subgroup_type != "rule":
            raise NotImplementedError(
                f"LinearFairClassifierFullRules only supports subgroup_type='rule', got {subgroup_type!r}. "
                "Exhaustive enumeration requires a finite subgroup family; 'linear' subgroups "
                "(real-valued coefficients/threshold) are continuous and cannot be enumerated - "
                "use LinearFairClassifier (lazy cuts) for that case instead."
            )
        self._gamma = gamma
        self.fairness_def = fairness_type
        self.subgroup_type = subgroup_type
        self.sparsity_term = sparsity_term

    def _make_linear_classif(
        self,
        X: np.ndarray[float],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        # trunk-ignore(ruff/B006)
        feat_init: dict[int, int] = {},
        epsilon: float = 1e-6,
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal linear classifier using 0-1 loss.

        Identical to LinearFairClassifier._make_linear_classif: sets up the base 0-1-loss MIO
        (and, for MSD, the p_pos/p_neg proportion variables used by fairness constraints), but
        without registering any lazy-constraint machinery - all fairness constraints are added
        directly to `model.fair_cuts` by the caller before the model is solved.

        Args:
            X (np.ndarray[bool]): input matrix
            y (np.ndarray[bool]): target labels
            feat_init (dict[int, int], optional): Initialization of the conjunction.
                A dictionary containing feature indices as keys and 0/1 values of whether they are used. Defaults to {}.

        Returns:
            pyo.ConcreteModel: The MIO model containing the formulation
        """
        n, d = X.shape

        model = pyo.ConcreteModel()
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        init_coef, init_intercept = None, None
        if feat_init is not None:
            if "coeffs" in feat_init:
                init_coef = feat_init["coeffs"]
            if "thresh" in feat_init:
                init_intercept = -feat_init["thresh"]

        model.coefs = pyo.Var(
            model.feat_i,
            domain=pyo.Reals,
            initialize=init_coef,
            bounds=(-1, 1),
        )
        model.intercept = pyo.Var(
            domain=pyo.Reals, initialize=init_intercept, bounds=(-d, d)
        )
        model.y_hat = pyo.Var(model.all_i, domain=pyo.Binary)

        if init_coef is not None and init_intercept is not None:
            coeffs = np.array([init_coef[i] for i in model.feat_i])
            pred = X @ coeffs + init_intercept >= 0
            init_yhat = {i: int(v) for i, v in enumerate(pred)}
            model.y_hat.set_values(init_yhat)

        bigM = 2*d + epsilon
        model.positive = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                sum(m.coefs[j] * X[i, j] for j in model.feat_i) + m.intercept
                >= (m.y_hat[i] - 1) * bigM
            ),
        )
        model.negative = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                sum(m.coefs[j] * X[i, j] for j in model.feat_i) + m.intercept
                <= m.y_hat[i] * bigM - epsilon
            ),
        )

        if self.sparsity_term != 0:
            model.non_zero = pyo.Var(model.feat_i, domain=pyo.Binary)
            model.from_below = pyo.Constraint(model.feat_i, rule=lambda m, j: m.coefs[j] >= -m.non_zero[j])
            model.from_above = pyo.Constraint(model.feat_i, rule=lambda m, j: m.coefs[j] <= m.non_zero[j])

            model.obj = pyo.Objective(
                expr=sum(model.y_hat[i]*weights[i] for i in model.neg_i)
                + sum((1 - model.y_hat[i])*weights[i] for i in model.pos_i)
                + sum(model.non_zero[j] for j in model.feat_i) * (self.sparsity_term/d),
                sense=pyo.minimize,
            )
        else:
            model.obj = pyo.Objective(
                expr=sum(model.y_hat[i]*weights[i] for i in model.neg_i)
                + sum((1 - model.y_hat[i])*weights[i] for i in model.pos_i),
                sense=pyo.minimize,
            )

        model.fair_cuts = pyo.ConstraintList()

        if self.fairness_def == "MSD":
            model.p_pos = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.p_pos_ref = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.pos_sum = pyo.Constraint(expr=sum(model.p_pos[i] for i in model.all_i) == 1)
            # positive and negative are flipped - CNF mapping
            model.pos_0 = pyo.Constraint(model.all_i, rule=lambda m, i: m.p_pos[i] <= m.y_hat[i])
            model.pos_ref = pyo.Constraint(model.all_i, rule=lambda m, i: m.p_pos[i] <= m.p_pos_ref)
            model.pos_ref_up = pyo.Constraint(model.all_i, rule=lambda m, i: m.p_pos[i] >= m.p_pos_ref - (1 - m.y_hat[i]))

            model.p_neg = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.p_neg_ref = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.neg_sum = pyo.Constraint(expr=sum(model.p_neg[i] for i in model.all_i) == 1)
            # positive and negative are flipped - CNF mapping
            model.neg_0 = pyo.Constraint(model.all_i, rule=lambda m, i: m.p_neg[i] <= 1 - m.y_hat[i])
            model.neg_ref = pyo.Constraint(model.all_i, rule=lambda m, i: m.p_neg[i] <= m.p_neg_ref)
            model.neg_ref_up = pyo.Constraint(model.all_i, rule=lambda m, i: m.p_neg[i] >= m.p_neg_ref - m.y_hat[i])

        return model

    def _enumerate_rule_subgroups(
        self,
        X_prot: np.ndarray[bool],
        n_min: int = 1,
        max_subgroups: int | None = None,
    ) -> list[tuple[tuple[int, ...], np.ndarray[bool]]]:
        """Enumerate every conjunction ("rule") of protected boolean features with support >= n_min.

        A rule subgroup is defined by a (non-empty) subset S of the columns of X_prot: a sample
        is in-group iff X_prot[i, j] is True for every j in S. Support is anti-monotone in S
        (adding a literal can only shrink or keep the matching set the same size), so subgroups
        are grown level-by-level (by itemset size) the way Apriori mines frequent itemsets: a
        candidate of size k+1 is only formed by joining two known-frequent size-k itemsets that
        share their first k-1 elements, and it is only kept if ALL of its size-k sub-itemsets
        were themselves frequent and its own support is still >= n_min. This prunes the search
        the moment a branch becomes infeasible, instead of ever materializing all 2^d subsets.

        Args:
            X_prot (np.ndarray[bool]): protected-attribute indicator matrix, shape (n, d)
            n_min (int): minimum number of matching samples for a subgroup to be kept. Using 1
                keeps this an *exact* enumeration of every non-empty subgroup (any subgroup with
                zero support trivially satisfies every fairness constraint anyway, so dropping it
                changes nothing). Raising it prunes rare, less statistically meaningful subgroups
                in exchange for tractability - see the class docstring's tractability warning.
            max_subgroups (int | None): if given, raise SubgroupBudgetExceeded as soon as the
                running total would exceed this, instead of silently continuing to enumerate
                (and potentially exhausting memory/time). None means no limit.

        Returns:
            list[tuple[tuple[int, ...], np.ndarray[bool]]]: list of (feature-index-tuple, in-group
                boolean mask over all n samples) pairs, one per distinct subgroup found.
        """
        n, d = X_prot.shape

        def check_budget(count: int, where: str) -> None:
            if max_subgroups is not None and count > max_subgroups:
                raise SubgroupBudgetExceeded(
                    f"Exhaustive rule-subgroup enumeration exceeded max_subgroups={max_subgroups} "
                    f"({count} subgroups found so far, while building {where}). "
                    "Raise n_min_group (fewer, more common subgroups) or max_subgroups, "
                    "or fall back to the lazy-cut LinearFairClassifier for this scenario."
                )

        level: dict[tuple[int, ...], np.ndarray] = {
            (j,): X_prot[:, j] for j in range(d) if X_prot[:, j].sum() >= n_min
        }
        all_groups: dict[tuple[int, ...], np.ndarray] = dict(level)
        check_budget(len(all_groups), "level 1 (single features)")
        level_size = 1

        while level:
            by_prefix: dict[tuple[int, ...], list[int]] = {}
            for itemset in level:
                by_prefix.setdefault(itemset[:-1], []).append(itemset[-1])

            next_level: dict[tuple[int, ...], np.ndarray] = {}
            for prefix, lasts in by_prefix.items():
                lasts.sort()
                for a in range(len(lasts)):
                    for b in range(a + 1, len(lasts)):
                        candidate = prefix + (lasts[a], lasts[b])
                        # Apriori pruning: every (k)-subset of a (k+1)-candidate must already be
                        # a known frequent itemset, or the candidate cannot be frequent either.
                        if not all(
                            candidate[:i] + candidate[i + 1:] in level
                            for i in range(len(candidate))
                        ):
                            continue
                        mask = level[prefix + (lasts[a],)] & level[prefix + (lasts[b],)]
                        if mask.sum() >= n_min:
                            next_level[candidate] = mask
                            check_budget(len(all_groups) + len(next_level), f"level {level_size + 1}")

            all_groups.update(next_level)
            level = next_level
            level_size += 1
            if next_level:
                logger.info(
                    f"Rule enumeration level {level_size}: {len(next_level)} new subgroups "
                    f"({len(all_groups)} total so far)"
                )

        return list(all_groups.items())

    def _add_group_constraints(self, ingroup_i: np.ndarray[int]) -> None:
        """Add BOTH direction fairness constraints for one subgroup directly to the model.

        Unlike LinearFairClassifier._add_cut (called lazily, only once a specific violation
        direction has been observed at an incumbent solution), here the model is built once,
        upfront, before solving - so both directions of the |...| <= gamma constraint are added
        unconditionally for every enumerated subgroup.
        """
        if ingroup_i.shape[0] == 0:
            # Zero-support subgroup: both proportions are 0, constraint is trivially satisfied.
            return

        neg_i = self.model.neg_i
        self.n_cuts += 2

        if self.fairness_def == "MSD":
            pos_prop = sum(self.model.p_pos[i] for i in ingroup_i)
            neg_prop = sum(self.model.p_neg[i] for i in ingroup_i)
            self.model.fair_cuts.add(pos_prop - neg_prop <= self._gamma)
            self.model.fair_cuts.add(neg_prop - pos_prop <= self._gamma)
            return

        if self.fairness_def == "SPSF":
            p_base = sum(self.model.y_hat[i] for i in self.model.all_i) / self.n_samples
        elif self.fairness_def in ["FPSF", "MSD0"]:
            p_base = sum(self.model.y_hat[i] for i in neg_i) / len(neg_i)
        else:
            raise ValueError(f"Unknown fairness definition {self.fairness_def}")

        # for FPSF/MSD0, ingroup_i is already restricted to indices where y_true == 0
        p_subgroup = ingroup_i.shape[0] / self.n_samples
        p_joint = sum(self.model.y_hat[i] for i in ingroup_i) / self.n_samples
        self.model.fair_cuts.add(p_base * p_subgroup - p_joint <= self._gamma)
        self.model.fair_cuts.add(p_joint - p_base * p_subgroup <= self._gamma)

    @property
    def n_samples(self):
        return self.X.shape[0]

    def find_classifier(
        self,
        X: np.ndarray[float],
        X_prot: np.ndarray[bool],
        y: np.ndarray[bool],
        time_limit: int = 120,
        epsilon: float = 1e-6,
        warmstart: bool = False,
        verbose: bool = False,
        n_min_group: int = 1,
        max_subgroups: int | None = None,
        solver_name: str = "gurobi",
    ) -> tuple[list[float], float]:
        """Find a single linear classifier with lowest 0-1 error, subject to gamma-fairness
        against EVERY rule (conjunction) subgroup, added as regular constraints upfront.

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            X_prot (np.ndarray[bool]): Input data only for protected attributes (boolean values), shape (n, d2)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            warmstart (bool, optional): If true, an approximate solution will be created first to warmstart the MIO.
                Defaults to False.
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.
            n_min_group (int): minimum in-group sample count for a rule subgroup to be
                enumerated (see `_enumerate_rule_subgroups`). Defaults to 1 (exact/exhaustive).
            max_subgroups (int | None): fail fast (SubgroupBudgetExceeded) if enumeration would
                exceed this many subgroups, instead of exhausting memory/time. Defaults to None
                (no limit) - set this for large protected dimensions.
            solver_name (str): pyomo solver name. Defaults to "gurobi" (no callback/lazy-constraint
                API is used here, so unlike LinearFairClassifier this also works with any other
                MIP solver pyomo supports, e.g. "glpk"/"appsi_highs", which is mainly useful for
                testing without a Gurobi license).

        Returns:
            tuple[list[float], float]: Tuple of linear combination coefficients nad a threshold value
        """
        assert y.shape == (X.shape[0],) and y.shape == (X_prot.shape[0],)
        assert y.dtype == bool
        assert X_prot.dtype == bool

        self.X = X
        self.X_prot = X_prot
        self.true_y = y
        self.verbose = verbose

        w = np.ones_like(y, dtype=float)
        size1 = np.sum(y)
        w[y] = 1 / size1
        w[~y] = 1 / (y.shape[0] - size1)
        init = {}
        if warmstart:
            lin_classif = LogisticRegression(random_state=0, class_weight={True: 1 / size1, False: 1 / (y.shape[0] - size1)})
            lin_classif.fit(X, y)
            coeffs = lin_classif.coef_[0]
            thresh = -lin_classif.intercept_[0]
            c_max = np.abs(coeffs).max()
            coeffs = coeffs / c_max
            thresh = thresh / c_max
            pred = X @ coeffs - thresh
            if np.any(pred >= 0):
                min_pos = np.min(pred[pred >= 0])
                thresh += min_pos
            init["coeffs"] = {i: coeffs[i] for i in range(len(coeffs))}
            init["thresh"] = thresh
        int_model = self._make_linear_classif(X, y, weights=w, epsilon=epsilon, feat_init=init)
        self.model = int_model

        self.n_cuts = 0
        self.n_callbacks = 0
        self.n_subgroups = 0
        t_enum_start = time.perf_counter()
        if self.fairness_def != "":
            subgroups = self._enumerate_rule_subgroups(X_prot, n_min=n_min_group, max_subgroups=max_subgroups)
            self.n_subgroups = len(subgroups)
            restrict_to_negatives = self.fairness_def in ("FPSF", "MSD0")
            for _feat_idx, mask in subgroups:
                group_mask = (mask & ~y) if restrict_to_negatives else mask
                self._add_group_constraints(np.flatnonzero(group_mask))
            logger.info(
                f"Enumerated {self.n_subgroups} rule subgroups -> {self.n_cuts} constraints "
                f"in {time.perf_counter() - t_enum_start:.1f}s"
            )
        self.enum_time = time.perf_counter() - t_enum_start

        if solver_name == "gurobi":
            opt = pyo.SolverFactory("gurobi", solver_io="python")
            opt.options["TimeLimit"] = time_limit
            opt.options["FeasibilityTol"] = 1e-7
            opt.options["IntFeasTol"] = 1e-7
        else:
            opt = pyo.SolverFactory(solver_name)
            time_limit_option = {"glpk": "tmlim"}.get(solver_name, "time_limit")
            try:
                opt.options[time_limit_option] = time_limit
            except Exception:
                logger.warning(
                    f"Could not set a time limit ({time_limit_option}) for solver {solver_name!r}; "
                    "it may run until full convergence."
                )

        t_start_solve = time.perf_counter()
        result = opt.solve(int_model, tee=verbose)
        t_diff = time.perf_counter() - t_start_solve
        logger.info(
            f"Estim. Train Accuracy: {1 - (sum(int_model.y_hat[i].value for i in int_model.neg_i) + sum(1 - int_model.y_hat[i].value for i in int_model.pos_i)) / self.n_samples}"
        )
        # No callback is used in this variant (all constraints are added upfront), kept for
        # output-reporting compatibility with LinearFairClassifier / experiment_train.py.
        self.callback_time_proportion = 0.0
        self.callback_time = 0.0
        self.total_time = t_diff
        self.mio_result = result

        coefs = np.array(
            [
                int_model.coefs[i].value if int_model.coefs[i].value is not None else 0
                for i in int_model.feat_i
            ],
            dtype=float,
        )
        threshold = -int_model.intercept.value

        return coefs, threshold
