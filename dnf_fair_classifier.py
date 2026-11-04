import logging
import time

import numpy as np
import pyomo.environ as pyo
from gurobipy import GRB

# from one_rule import OneRule
# from one_rule_lp import OneRule
# from spsf_mio import SPSF
# from msd import MSD
from bias_detect_mio import BiasDetect
from utils import eval_fpsf, eval_spsf, eval_msd

# ignore assert warnings
# trunk-ignore-all(bandit/B101)


logger = logging.getLogger(__name__)


class DNFFairClassifier:
    """Implementation of a MIO formulation for finding an optimal DNF with the lowest 0-1 error constrained to gamma-fairness.
    The formulation is based on an implementation of eq. (10) in https://krvarshney.github.io/pubs/SuWVM_mlsp2016.pdf
    """

    def __init__(self, gamma: float, fairness_type: str = "", subgroup_type: str = "rule") -> None:
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
            - rule (default)
            - linear
        """
        self._gamma = gamma
        self.fairness_def = fairness_type
        self.subgroup_type = subgroup_type

    # TODO implement hamming loss
    def _make_int_cnf(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        n_clauses: int,
        # trunk-ignore(ruff/B006)
        feat_init: dict[int, int] = {},
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal conjunction using 0-1 loss.

        Args:
            X (np.ndarray[bool]): input matrix
            y (np.ndarray[bool]): target labels
            feat_init (dict[int, int], optional): Initialization of the conjunction.
                A dictionary containing feature indices as keys and 0/1 values of whether they are used. Defaults to {}.

        Returns:
            pyo.ConcreteModel: The MIO model containing the formulation
        """
        n, d = X.shape
        Xint = np.zeros_like(X, dtype=int)
        Xint[X] = 1

        model = pyo.ConcreteModel()
        model.clause_i = pyo.Set(initialize=np.arange(n_clauses))
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        model.use_feat = pyo.Var(
            model.feat_i,
            model.clause_i,
            domain=pyo.Binary,
            initialize=feat_init,
        )
        # model.error = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.error = pyo.Var(model.all_i, domain=pyo.Binary, bounds=(0, 1))

        # model.pos = pyo.Constraint(
        #     model.pos_i,
        #     model.clause_i,
        #     rule=lambda m, i, r: (
        #         m.error[i] >= 1 - sum(Xint[i, j] * m.use_feat[j, r] for j in m.feat_i)
        #     ),
        # )

        model.clause_res = pyo.Var(
            model.all_i, model.clause_i, domain=pyo.NonNegativeReals, bounds=(0, 1)
        )
        model.clause1 = pyo.Constraint(
            model.all_i,
            model.feat_i,
            model.clause_i,
            rule=lambda m, i, j, r: m.clause_res[i, r] >= Xint[i, j] * m.use_feat[j, r],
        )
        model.clause2 = pyo.Constraint(
            model.all_i,
            model.clause_i,
            rule=lambda m, i, r: m.clause_res[i, r] <= sum(Xint[i, j] * m.use_feat[j, r] for j in m.feat_i),
        )
        model.neg1 = pyo.Constraint(
            model.neg_i,
            rule=lambda m, i: (
                m.error[i]
                >= sum(m.clause_res[i, r] for r in m.clause_i) - (n_clauses - 1)
            ),
        )
        model.neg2 = pyo.Constraint(
            model.neg_i,
            rule=lambda m, i: (
                m.error[i]
                <= sum(m.clause_res[i, r] for r in m.clause_i) / n_clauses
            ),
        )
        model.pos1 = pyo.Constraint(
            model.pos_i,
            rule=lambda m, i: (
                m.error[i]
                >= 1 - sum(m.clause_res[i, r] for r in m.clause_i) / n_clauses
            ),
        )
        model.pos2 = pyo.Constraint(
            model.pos_i,
            rule=lambda m, i: (
                m.error[i]
                <= n_clauses - sum(m.clause_res[i, r] for r in m.clause_i)
            ),
        )

        model.obj = pyo.Objective(
            expr=sum(model.error[i] * weights[i] for i in model.all_i),
            sense=pyo.minimize,
        )

        model.fair_cuts = pyo.ConstraintList()

        if self.fairness_def == "MSD":
            model.p_pos = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.p_pos_ref = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.pos_sum = pyo.Constraint(expr=sum(model.p_pos[i] for i in model.all_i) == 1)
            # positive and negative are flipped - CNF mapping
            model.pos_0_onpos = pyo.Constraint(model.neg_i, rule=lambda m, i: m.p_pos[i] <= 1 - m.error[i])
            model.pos_0_onneg = pyo.Constraint(model.pos_i, rule=lambda m, i: m.p_pos[i] <= m.error[i])
            model.pos_ref = pyo.Constraint(model.all_i, rule=lambda m, i: m.p_pos[i] <= m.p_pos_ref)
            model.pos_ref_up_onpos = pyo.Constraint(model.neg_i, rule=lambda m, i: m.p_pos[i] >= m.p_pos_ref - m.error[i])
            model.pos_ref_up_onneg = pyo.Constraint(model.pos_i, rule=lambda m, i: m.p_pos[i] >= m.p_pos_ref - (1 - m.error[i]))

            model.p_neg = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.p_neg_ref = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.neg_sum = pyo.Constraint(expr=sum(model.p_neg[i] for i in model.all_i) == 1)
            # positive and negative are flipped - CNF mapping
            model.neg_0_onpos = pyo.Constraint(model.neg_i, rule=lambda m, i: m.p_neg[i] <= m.error[i])
            model.neg_0_onneg = pyo.Constraint(model.pos_i, rule=lambda m, i: m.p_neg[i] <= 1 - m.error[i])
            model.neg_ref = pyo.Constraint(model.all_i, rule=lambda m, i: m.p_neg[i] <= m.p_neg_ref)
            model.neg_ref_up_onpos = pyo.Constraint(model.neg_i, rule=lambda m, i: m.p_neg[i] >= m.p_neg_ref - (1 - m.error[i]))
            model.neg_ref_up_onneg = pyo.Constraint(model.pos_i, rule=lambda m, i: m.p_neg[i] >= m.p_neg_ref - m.error[i])

        return model

    def _add_cut(self, ingroup_i: np.ndarray[int], positive_direction: bool):
        # indices of negative and positive classifications are swapped,
        #   since we learn a CNF equivalent to sought DNF (because of convexity)
        # here we swap them to the correct values as used for FPSF
        neg_i = self.model.pos_i

        self.n_cuts += 1
        # for FPSF ingroup contains only those indices where y_true == 0!
        # p_group = ingroup_i.shape[0] / self.n_samples
        # sum_y_hat = sum(self.model.error[i] for i in neg_i)
        # sum_group_y_hat = sum(self.model.error[i] for i in ingroup_i)
        # logger.debug(f"ADDING CUT ingroup_i={ingroup_i} p={p_group}")
        # if positive_direction:
        #     return self.model.fair_cuts.add(
        #         (p_group / len(neg_i)) * sum_y_hat
        #         - (1 / self.n_samples) * sum_group_y_hat
        #         <= self._gamma
        #     )
        # else:
        #     return self.model.fair_cuts.add(
        #         (1 / self.n_samples) * sum_group_y_hat
        #         - (p_group / len(neg_i)) * sum_y_hat
        #         <= self._gamma
        #     )

        if self.fairness_def == "MSD":
            pos_prop = sum(self.model.p_pos[i] for i in ingroup_i)
            neg_prop = sum(self.model.p_neg[i] for i in ingroup_i)
            logger.debug(f"ADDING MSD CUT ingroup_i={ingroup_i}")
            if positive_direction:
                return self.model.fair_cuts.add(pos_prop - neg_prop <= self._gamma)
            else:
                return self.model.fair_cuts.add(neg_prop - pos_prop <= self._gamma)
        if self.fairness_def == "SPSF":
            p_base = sum(1 - self.model.error[i] if self.true_y[i] else self.model.error[i] for i in self.model.all_i) / self.n_samples
        elif self.fairness_def in ["FPSF", "MSD0"]:
            p_base = sum(self.model.error[i] for i in neg_i) / len(neg_i)
        # for FPSF ingroup contains only those indices where y_true == 0!
        p_subgroup = ingroup_i.shape[0] / self.n_samples
        p_joint = sum(1 - self.model.error[i] if self.true_y[i] else self.model.error[i] for i in ingroup_i) / self.n_samples
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
        errors: np.ndarray[bool],
    ) -> tuple[np.ndarray[int], float]:
        self.n_callbacks += 1
        y_hat = np.logical_xor(self.true_y, errors)
        mio = BiasDetect()
        if self.fairness_def in ["FPSF", "MSD0"]:
            mask = self.true_y == 0
            if self.fairness_def == "FPSF":
                gamma = self._gamma / mask.mean() # SPSF(on y=0) * n_0 / n is FPSF - we divide the gamma by the factor
                group_def = mio.find_subgroup(self.X_prot[mask], y_hat[mask], verbose=False, bias_type="spsf", group_type=self.subgroup_type, gamma=gamma)
            else:
                group_def = mio.find_subgroup(self.X_prot[mask], y_hat[mask], verbose=self.verbose, bias_type="msd", group_type=self.subgroup_type)

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
            # even for msd0, use fpsf - maybe it correlates?
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
        logger.info(
            f"FOUND SUBGROUP {group_def} with violation {violation} (MIO:{mio.model.o.value if hasattr(mio, 'model') else '--'}) in {'positive' if direction else 'negative'} direction"
        )
        return ingroup_i, violation, direction
        # if self.fairness_def == "FPSF":
        #     spsf_mio = SPSF()
        #     mask = self.true_y == 0
        #     rule = spsf_mio.find_rule(self.X_prot[mask], y_hat[mask], verbose=self.verbose)
        #     group = np.ones_like(self.true_y, dtype=bool)
        #     for conj in rule:
        #         group &= self.X_prot[:, conj]
        #     violation, direction = eval_fpsf(self.true_y, y_hat, group, get_direction=True)
        # elif self.fairness_def == "SPSF":
        #     spsf_mio = SPSF()
        #     rule = spsf_mio.find_rule(self.X_prot, y_hat, verbose=self.verbose)
        #     group = np.ones_like(self.true_y, dtype=bool)
        #     for conj in rule:
        #         group &= self.X_prot[:, conj]
        #     violation, direction = eval_spsf(y_hat, group, get_direction=True)
        # elif self.fairness_def == "MSD":
        #     msd = MSD()
        #     rule = msd.find_rule(self.X_prot, y_hat, verbose=self.verbose)
        #     group = np.ones_like(self.true_y, dtype=bool)
        #     for conj in rule:
        #         group &= self.X_prot[:, conj]
        #     violation, direction = eval_msd(y_hat, group, get_direction=True)
        # elif self.fairness_def == "MSD0":
        #     mio = MSD()
        #     mask = self.true_y == 0
        #     rule = mio.find_rule(self.X_prot[mask], y_hat[mask], verbose=self.verbose)
        #     group = np.ones_like(self.true_y, dtype=bool)
        #     for conj in rule:
        #         group &= self.X_prot[:, conj]
        #     violation, direction = eval_fpsf(y_hat, group, get_direction=True)
        # else:
        #     raise ValueError(f"Unknown fairness definition {self.fairness_def}")
        # logger.debug(
        #     f"FOUND SUBGROUP {rule} with violation {violation} in {'positive' if direction else 'negative'} direction"
        # )
        # ingroup_i = np.where(group & ~self.true_y)[0]
        # return ingroup_i, violation, direction

    @property
    def n_samples(self):
        return self.X.shape[0]

    def find_dnf(
        self,
        X: np.ndarray[bool],
        X_prot: np.ndarray[bool],
        y: np.ndarray[bool],
        n_terms: int,
        time_limit: int = 120,
        warmstart: bool = False,
        verbose: bool = False,
    ) -> list[list[int]]:
        """Find a single conjunction with lowest 0-1 error

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            X_prot (np.ndarray[bool]): Input data only for protected attributes (boolean values), shape (n, d2)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            warmstart (bool, optional): If true, an approximate solution will be created first to warmstart the MIO.
                Defaults to False.
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.

        Returns:
            list[int]: List of indices of the literals in the final conjunction
        """
        assert y.shape == (X.shape[0],) and y.shape == (X_prot.shape[0],)
        assert X.dtype == bool and y.dtype == bool
        assert self.subgroup_type != "rule" or X_prot.dtype == bool

        if warmstart:
            print("No warmstart available")

        self.X = X
        self.X_prot = X_prot
        self.true_y = y
        self.verbose = verbose

        w = np.ones_like(y, dtype=float)
        size1 = np.sum(y)
        w[y] = 1 / size1
        w[~y] = 1 / (y.shape[0] - size1)
        # it finds a CNF -> negate inputs and outputs
        int_model = self._make_int_cnf(~X, ~y, weights=w, n_clauses=n_terms)
        # opt = pyo.SolverFactory("gurobi", solver_io="python")
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.options["TimeLimit"] = time_limit
        opt.set_instance(int_model)
        opt.set_gurobi_param("PreCrush", 1)
        opt.set_gurobi_param("LazyConstraints", 1)
        self.model = int_model

        def callback(cb_m, cb_opt, cb_where):
            if cb_where == GRB.Callback.MIPSOL and self.fairness_def != "":
                t_start = time.perf_counter()
                cb_opt.cbGetSolution(
                    vars=[self.model.error[i] for i in self.model.all_i]
                )
                errors = np.array(
                    [self.model.error[i].value > 1e-4 for i in self.model.all_i]
                )
                group_i, violation, pos_direction = self._find_subgroup(errors)
                if violation > self._gamma:
                    cb_opt.cbLazy(self._add_cut(group_i, pos_direction))
                self.__callback_time += time.perf_counter() - t_start

        if self.fairness_def != "":
            opt.set_gurobi_param("PreCrush", 1)
            opt.set_gurobi_param("LazyConstraints", 1)
            opt.set_callback(callback)

        self.n_cuts = 0
        self.n_callbacks = 0
        self.__callback_time = 0
        t_start_solve = time.perf_counter()
        result = opt.solve(tee=verbose)
        t_diff = time.perf_counter() - t_start_solve
        print(f"~Time~ in callbacks: {self.__callback_time}")
        self.callback_time_proportion = self.__callback_time / t_diff
        self.mio_result = result

        dnf = [
            [i for i in int_model.feat_i if int_model.use_feat[i, r].value != 0]
            for r in int_model.clause_i
        ]

        return dnf
