import logging
import time

import numpy as np
import pyomo.environ as pyo

from ._exact_backend import ExactDetector
from ._mio import MIODetector
from ._gurobi import import_grb
from .measures import fpsf as eval_fpsf, msd as eval_msd, spsf as eval_spsf

from sklearn.linear_model import LogisticRegression



logger = logging.getLogger(__name__)


class LinearFairClassifier:
    """MIO formulation for finding a linear classifier with the lowest (balanced) 0-1 error, constrained to gamma-fairness."""

    def __init__(self, gamma: float, fairness_type: str = "", subgroup_type: str = "rule", sparsity_term: float = 0, detection_backend: str = "exact") -> None:
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
            detection_backend (str): "exact" (Rust search, conjunction subgroups only)
                or "mio" (MIO solved by Gurobi). Linear subgroups always use "mio".
        Group Types:
            - rule (default)
            - linear
        """
        self._gamma = gamma
        self.fairness_def = fairness_type
        self.subgroup_type = subgroup_type
        self.sparsity_term = sparsity_term
        self.detection_backend = detection_backend

    def _make_detector(self):
        """Subgroup search used to find violating subgroups (the exact search supports only conjunctions)."""
        if self.detection_backend == "exact" and self.subgroup_type == "rule":
            return ExactDetector()
        return MIODetector()


    def _make_linear_classif(
        self,
        X: np.ndarray[float],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        feat_init: dict[int, int] = {},
        epsilon: float = 1e-6,
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal linear classifier using 0-1 loss.

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

    def _add_cut(self, ingroup_i: np.ndarray[int], positive_direction: bool):
        neg_i = self.model.neg_i

        self.n_cuts += 1
        self.cut_subgroups.append(self._last_group_def)
        # ingroup contains only those indices where y_true == 0!

        if self.fairness_def == "MSD":
            pos_prop = sum(self.model.p_pos[i] for i in ingroup_i)
            neg_prop = sum(self.model.p_neg[i] for i in ingroup_i)
            logger.debug(f"ADDING MSD CUT ingroup_i={ingroup_i}")
            if positive_direction:
                return self.model.fair_cuts.add(pos_prop - neg_prop <= self._gamma)
            else:
                return self.model.fair_cuts.add(neg_prop - pos_prop <= self._gamma)
        if self.fairness_def == "SPSF":
            p_base = sum(self.model.y_hat[i] for i in self.model.all_i) / self.n_samples
        elif self.fairness_def in ["FPSF", "MSD0"]:
            p_base = sum(self.model.y_hat[i] for i in neg_i) / len(neg_i)
        # for FPSF ingroup contains only those indices where y_true == 0!
        p_subgroup = ingroup_i.shape[0] / self.n_samples
        p_joint = sum(self.model.y_hat[i] for i in ingroup_i) / self.n_samples
        logger.debug(f"ADDING CUT ingroup_i={ingroup_i} p={p_subgroup}")
        if positive_direction:
            return self.model.fair_cuts.add(
                p_base * p_subgroup - p_joint <= self._gamma
            )
        else:
            return self.model.fair_cuts.add(
                p_joint - p_base * p_subgroup <= self._gamma
            )

    def _find_subgroup(
        self,
        y_hat: np.ndarray[bool],
    ) -> tuple[np.ndarray[int], float]:
        self.n_callbacks += 1
        mio = self._make_detector()
        if self.fairness_def in ["FPSF", "MSD0"]:
            mask = self.true_y == 0
            if self.fairness_def == "FPSF":
                gamma = self._gamma / mask.mean() # SPSF(on y=0) * n_0 / n is FPSF - we divide the gamma by the factor
                group_def = mio.find_subgroup(self.X_prot[mask], y_hat[mask], verbose=self.verbose, bias_type="spsf", group_type=self.subgroup_type, gamma=gamma)
            else:
                gamma = self._gamma / (y_hat[mask].mean() * (~y_hat[mask]).mean()) # SPSF = p1 * p0 * MSD, thus we divide by both ps
                group_def = mio.find_subgroup(self.X_prot[mask], y_hat[mask], verbose=self.verbose, bias_type="msd", group_type=self.subgroup_type, gamma=gamma)

            if group_def is None:
                logger.info("Did NOT find a SUBGROUP.")
                return None, 0, True
            if self.subgroup_type == "rule":
                ingroup = np.ones_like(self.true_y, dtype=bool)
                for feat_i in group_def:
                    ingroup &= self.X_prot[:, feat_i]
            elif self.subgroup_type == "linear":
                coeffs, thresh = group_def[:-1], group_def[-1]
                ingroup = self.X_prot @ coeffs >= thresh
            # MSD0 only uses SD to find the subgroup; the constraint (cut) is always on FPSF
            violation, direction = eval_fpsf(self.true_y, y_hat, ingroup, get_direction=True)
            ingroup_i = np.flatnonzero(ingroup & ~self.true_y)
        else:
            group_def, group = mio.find_subgroup_with_ingroup(self.X_prot, y_hat, verbose=self.verbose, bias_type=self.fairness_def.lower(), group_type=self.subgroup_type, gamma=self._gamma)
            if group_def is None:
                logger.info("Did NOT find a SUBGROUP.")
                return None, 0, True
            if self.fairness_def == "SPSF":
                violation, direction = eval_spsf(y_hat, group, get_direction=True)
            elif self.fairness_def == "MSD":
                violation, direction = eval_msd(y_hat, group, get_direction=True)
            else:
                raise ValueError(f"Unknown fairness definition {self.fairness_def}")
            ingroup_i = np.flatnonzero(group)
        self._last_group_def = group_def
        logger.info(
            f"FOUND SUBGROUP {group_def} with violation {violation} (MIO:{mio.model.o.value if hasattr(mio, 'model') else '--'}) in {'positive' if direction else 'negative'} direction"
        )
        return ingroup_i, violation, direction

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
    ) -> tuple[list[float], float]:
        """Find a single conjunction with lowest 0-1 error

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            X_prot (np.ndarray[bool]): Input data only for protected attributes (boolean values), shape (n, d2)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            warmstart (bool, optional): If true, an approximate solution will be created first to warmstart the MIO.
                Defaults to False.
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.

        Returns:
            tuple[list[float], float]: Tuple of linear combination coefficients nad a threshold value
        """
        assert y.shape == (X.shape[0],) and y.shape == (X_prot.shape[0],)
        assert y.dtype == bool
        assert self.subgroup_type != "rule" or X_prot.dtype == bool

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
        GRB = import_grb()
        int_model = self._make_linear_classif(X, y, weights=w, epsilon=epsilon, feat_init=init)
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.options["TimeLimit"] = time_limit
        opt.set_instance(int_model)
        self.model = int_model

        def callback(cb_m, cb_opt, cb_where):
            try:
                if cb_where == GRB.Callback.MIPSOL:
                    t_start = time.perf_counter()
                    obj_value = cb_opt.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_value > self.__best_obj:
                        return
                    cb_opt.cbGetSolution(
                        vars=[self.model.y_hat[i] for i in self.model.all_i]
                    )
                    y_hat = np.array(
                        [self.model.y_hat[i].value > 1e-4 for i in self.model.all_i]
                    )
                    group_i, violation, pos_direction = self._find_subgroup(y_hat)
                    if violation > self._gamma:
                        logger.info(f"Solution with objective {obj_value} was rejected.")
                        cb_opt.cbLazy(self._add_cut(group_i, pos_direction))
                    else:
                        logger.info(f"Solution with objective {obj_value} passed the check.")
                        self.__best_obj = obj_value
                    self.__callback_time += time.perf_counter() - t_start
            except Exception as e:
                logger.exception(f"Error in callback: {e}")

        opt.set_gurobi_param("FeasibilityTol", 1e-7)
        opt.set_gurobi_param("IntFeasTol", 1e-7)

        if self.fairness_def != "":
            opt.set_gurobi_param("PreCrush", 1)
            opt.set_gurobi_param("LazyConstraints", 1)
            opt.set_callback(callback)
            self.__best_obj = np.inf
        self.n_cuts = 0
        self.cut_subgroups = []  # descriptions of the subgroups added as constraints
        self.n_callbacks = 0
        self.__callback_time = 0
        t_start_solve = time.perf_counter()
        result = opt.solve(tee=verbose)
        t_diff = time.perf_counter() - t_start_solve
        logger.info(f"Time in callbacks: {self.__callback_time}")
        logger.info(
            f"Estim. Train Accuracy: {1 - (sum(int_model.y_hat[i].value for i in int_model.neg_i) + sum(1 - int_model.y_hat[i].value for i in int_model.pos_i)) / self.n_samples}"
        )
        self.callback_time_proportion = self.__callback_time / t_diff
        self.callback_time = self.__callback_time
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
