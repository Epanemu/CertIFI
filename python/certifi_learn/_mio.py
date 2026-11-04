import inspect
import logging

import numpy as np
import pyomo.environ as pyo

from sklearn.linear_model import LogisticRegression



logger = logging.getLogger(__name__)


class MIODetector:
    """Implementation of a MIO formulation for finding most disparate subgroup
    The formulation is inspired by 1Rule method from http://proceedings.mlr.press/v28/malioutov13.pdf
    """
    EPS = 1e-6

    def __init__(self) -> None:
        pass

    @staticmethod
    def _extract_gap(opt: pyo.SolverFactory, solver: str) -> tuple[float, float, float]:
        """Extract the (relative) MIP gap, best objective bound, and incumbent objective
        value from a solved model, using the solver-specific low-level API.
        """
        solver_model = opt._solver_model
        if solver == "gurobi":
            return solver_model.MIPGap, solver_model.ObjBound, solver_model.ObjVal
        elif solver == "highs":
            info = solver_model.getInfo()
            return info.mip_gap, info.mip_dual_bound, info.objective_function_value
        else:
            raise ValueError("Incorrect solver, choose one of ['gurobi', 'highs']")

    def _make_rule_model(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        n_min: int,
        feat_init: dict[int, int] = {},
        counts: list[int] = [],
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal conjunction.

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
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        model.use_feat = pyo.Var(model.feat_i, domain=pyo.Binary, initialize=feat_init)
        model.ingroup = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))

        model.force_0 = pyo.Constraint(
            model.all_i,
            model.feat_i,
            rule=lambda m, i, j: (
                m.ingroup[i] <= 1 - (m.use_feat[j] - Xint[i, j] * m.use_feat[j])
            ),
        )
        model.force_1 = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                m.ingroup[i]
                >= 1 - sum(m.use_feat[j] - Xint[i, j] * m.use_feat[j] for j in m.feat_i)
            ),
        )

        if n_min > 0:
            model.minimum = pyo.Constraint(
                expr=sum(model.ingroup[i] * counts[i] for i in model.all_i) >= n_min
            )

        model.o = pyo.Var(domain=pyo.NonNegativeReals)
        model.b = pyo.Var(domain=pyo.Binary)
        term1 = sum(model.ingroup[i] * weights[i] for i in model.pos_i)
        term2 = sum(model.ingroup[i] * weights[i] for i in model.neg_i)
        model.abs_obj_u1 = pyo.Constraint(expr=model.o <= term1 - term2 + 2 * model.b)
        model.abs_obj_u2 = pyo.Constraint(
            expr=model.o <= term2 - term1 + 2 * (1 - model.b)
        )
        model.obj = pyo.Objective(
            expr=model.o,
            sense=pyo.maximize,
        )

        return model

    def _make_linear_model(
        self,
        X: np.ndarray[float],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        n_min: int,
        feat_init: dict[int, int] = {},
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal linear model.

        Args:
            X (np.ndarray[float]): input matrix
            y (np.ndarray[bool]): target labels
            feat_init (dict[int, int], optional): Initialization of the linear model.
                A dictionary containing feature indices as keys and 0/1 values of whether they are used. Defaults to {}.

        Returns:
            pyo.ConcreteModel: The MIO model containing the formulation
        """
        n, d = X.shape
        X = X.copy().astype(float)


        if feat_init != {}:
            init_coef, init_thresh = feat_init["coeffs"], feat_init["thresh"]
        else:
            init_coef, init_thresh = None, None

        model = pyo.ConcreteModel()
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        model.coeffs = pyo.Var(model.feat_i, domain=pyo.Reals, initialize=init_coef, bounds=(-1,1))
        model.thresh = pyo.Var(domain=pyo.Reals, initialize=init_thresh, bounds=(-d, d))
        model.ingroup = pyo.Var(model.all_i, domain=pyo.Binary)

        if init_coef is not None:
            coeffs = np.array([init_coef[i] for i in model.feat_i])
            ingroup_pred = X @ coeffs - init_thresh >= 0
            init_ingroup = {i: int(v) for i, v in enumerate(ingroup_pred)}
            model.ingroup.set_values(init_ingroup)

        bigM = 2*d

        model.force_0 = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                sum(X[i, j] * m.coeffs[j] for j in m.feat_i) - m.thresh
                >= -(1 - m.ingroup[i]) * bigM
            ),
        )
        model.force_1 = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                sum(X[i, j] * m.coeffs[j] for j in m.feat_i) - m.thresh
                <= m.ingroup[i] * bigM - self.EPS
            ),
        )

        if n_min > 0:
            model.minimum = pyo.Constraint(
                expr=sum(model.ingroup[i] for i in model.all_i) >= n_min
            )

        M_obj = float(np.sum(np.abs(weights)))

        model.o = pyo.Var(domain=pyo.NonNegativeReals)
        model.b = pyo.Var(domain=pyo.Binary)
        term1 = sum(model.ingroup[i] * weights[i] for i in model.pos_i)
        term2 = sum(model.ingroup[i] * weights[i] for i in model.neg_i)
        model.abs_obj_u1 = pyo.Constraint(expr=model.o <= term1 - term2 + M_obj * model.b)
        model.abs_obj_u2 = pyo.Constraint(
            expr=model.o <= term2 - term1 + M_obj * (1 - model.b)
        )
        model.obj = pyo.Objective(
            expr=model.o,
            sense=pyo.maximize,
        )

        return model

    def find_subgroup(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        verbose: bool = False,
        n_min: int = 0,
        time_limit: int = 300,
        bias_type: str = "msd", # or "spsf"
        group_type: str = "rule", # or "linear"
        probabilistic: bool = False,
        return_opt_flag: bool = False,
        return_gap: bool = False,
        warmstart_model: dict[str] | None = None,
        warmstart: bool = True,
        gamma: float | None = None,
        solver: str = "gurobi", # or "highs"
    ) -> list[int]:
        """Find a subgroup

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            y (np.ndarray[bool]): Target (boolean values or probability values), shape (n,)
            warmstart (bool, optional): If true, an approximate solution will be created first to warmstart the MIO.
                Defaults to False.
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.
            solver (str, optional): Which MIO solver to use, one of "gurobi" or "highs". Defaults to "gurobi".

        Returns:
            list[int]: List of indices of the literals in the final conjunction or a X.shape[1]+1 coefficients (a,b) of a linear threshold function aTx >= b (the last element is the threshold b)
        """
        assert y.shape == (X.shape[0],)
        if group_type == "rule":
            assert X.dtype == bool
        if not probabilistic:
            assert y.dtype == bool


        n = y.shape[0]
        if not probabilistic:
            size1 = np.sum(y)
            if size1 == 0 or size1 == n:
                if group_type == "rule":
                    if return_opt_flag:
                        return [], True
                    return []
                elif group_type == "linear":
                    if return_opt_flag:
                        return [0] * (X.shape[1]+1), True
                    return [0] * (X.shape[1]+1)
                else:
                    raise ValueError("Incorrect group type, choose one of ['rule', 'linear']")

            size0 = n - size1
            p1 = size1 / n
            p0 = size0 / n
        else:
            raise NotImplementedError("Under construction")

        X0, counts0 = np.unique(X[~y], return_counts=True, axis=0)
        X1, counts1 = np.unique(X[y], return_counts=True, axis=0)
        X_pruned = np.concatenate([X0, X1], axis=0)
        if not probabilistic:
            y_pruned = np.zeros(X_pruned.shape[0], dtype=bool)
            y_pruned[X0.shape[0] :] = True
        else:
            raise NotImplementedError("Under construction")
        if bias_type == "msd":
            w = np.concatenate([counts0 / size0, counts1 / size1], axis=0)
        elif bias_type == "spsf":
            w = np.concatenate([counts0 * (p1 / n), counts1 * (p0 / n)], axis=0)
        else:
            raise ValueError("Incorrect group type, choose one of ['rule', 'linear']")

        if group_type == "rule":
            int_model = self._make_rule_model(X_pruned, y_pruned, weights=w, n_min=n_min, counts=np.concatenate([counts0, counts1]))
        elif group_type == "linear":
            init = {}
            if warmstart and warmstart_model is None:
                Classifier = LogisticRegression
                if bias_type == "msd":
                    lin_classif = Classifier(random_state=0, class_weight={True: 1/size1, False: 1/size0})
                elif bias_type == "spsf":
                    lin_classif = Classifier(random_state=0, class_weight={True: 1/(p0 / n), False: 1/(p1 / n)})
                lin_classif.fit(X, y)
                coeffs = lin_classif.coef_[0]
                thresh = -lin_classif.intercept_[0]
                c_max = np.abs(coeffs).max()
                coeffs = coeffs / c_max
                thresh = thresh / c_max
                pred = X_pruned @ coeffs - thresh
                if np.any(pred >= 0):
                    min_pos = np.min(pred[pred >= 0])
                    thresh += min_pos
                init["coeffs"] = {i: coeffs[i] for i in range(len(coeffs))}
                init["thresh"] = thresh
            elif warmstart:
                init["coeffs"] = {i: c for i, c in enumerate(warmstart_model["coeffs"])}
                init["thresh"] = warmstart_model["thresh"]
            int_model = self._make_linear_model(X_pruned, y_pruned, weights=w, n_min=n_min, feat_init=init)
        else:
            raise ValueError("Incorrect group type, choose one of ['rule', 'linear']")
        # set a lower bound on the objective, we are interested only in subgroups above a certain gamma
        if gamma is not None:
            int_model.gamma_lower = pyo.Constraint(expr=int_model.o >= gamma)
        self.model = int_model
        if solver == "gurobi":
            opt = pyo.SolverFactory("gurobi", solver_io="python")
            opt.options["TimeLimit"] = time_limit
            opt.options["FeasibilityTol"] = 1e-8
            opt.options["IntFeasTol"] = 1e-8
            opt.options["OutputFlag"] = 1 if verbose else 0
        elif solver == "highs":
            # appsi_highs is Pyomo's HiGHS interface (wraps highspy). Option
            # names below are HiGHS' closest equivalents to the Gurobi ones
            # above: FeasibilityTol -> primal/dual_feasibility_tolerance,
            # IntFeasTol -> mip_feasibility_tolerance, OutputFlag -> output_flag.
            opt = pyo.SolverFactory("appsi_highs")
            opt.options["time_limit"] = time_limit
            opt.options["primal_feasibility_tolerance"] = 1e-8
            opt.options["dual_feasibility_tolerance"] = 1e-8
            opt.options["mip_feasibility_tolerance"] = 1e-8
        else:
            raise ValueError("Incorrect solver, choose one of ['gurobi', 'highs']")
        use_warmstart = group_type == "linear"
        solve_params = inspect.signature(opt.solve).parameters
        if "warmstart" in solve_params:
            # older/appsi-legacy signature: warmstart is a solve() kwarg
            result = opt.solve(int_model, tee=verbose, warmstart=use_warmstart)
        else:
            # newer appsi signature dropped the solve() kwarg in favor of the
            # config object; set it there instead so HiGHS still gets the
            # initialized variable values (feat_init) as a MIP start.
            if use_warmstart and hasattr(opt, "config") and "warmstart" in opt.config:
                opt.config.warmstart = True
            elif use_warmstart:
                logger.warning(
                    "Solver '%s' does not expose a warmstart option in this pyomo "
                    "version; proceeding without warmstart.", solver
                )
            result = opt.solve(int_model)#, tee=verbose)
        optim = True
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            logger.info("Solver did not find an optimal solution")
            optim = False

        if int_model.o.value is None:
            # No feasible solution found
            if return_opt_flag:
                return None, result.solver.termination_condition in [pyo.TerminationCondition.infeasible, pyo.TerminationCondition.infeasibleOrUnbounded]
            else:
                return None

        if group_type == "rule":
            vals = [int_model.use_feat[i].value for i in int_model.feat_i]
            rule = [i for i in int_model.feat_i if vals[i] is not None and vals[i] >= 0.5]
            if return_opt_flag:
                return rule, optim
            if return_gap:
                gap, obj_bound, obj_val = self._extract_gap(opt, solver)
                return rule, gap, obj_bound, obj_val
            return rule
        elif group_type == "linear":
            vals = [int_model.coeffs[i].value for i in int_model.feat_i]
            vals.append(int_model.thresh.value)
            model_in = np.array([int_model.ingroup[i].value > 0.5 for i in int_model.all_i])
            min_pred = np.min(X_pruned[model_in] @ vals[:-1] - vals[-1])
            if min_pred < 0:
                vals[-1] -= min(-min_pred, self.EPS/2)
            if return_opt_flag:
                return vals, optim
            if return_gap:
                gap, obj_bound, obj_val = self._extract_gap(opt, solver)
                return vals, gap, obj_bound, obj_val
            return vals

    def find_subgroup_with_ingroup(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        verbose: bool = False,
        n_min: int = 0,
        time_limit: int = 300,
        bias_type: str = "msd", # or "spsf"
        group_type: str = "rule", # or "linear"
        probabilistic: bool = False,
        return_opt_flag: bool = False,
        warmstart_model: dict | None = None,
        gamma: int | None = None,
        solver: str = "gurobi", # or "highs"
    ) -> list[int]:
        group_desc, opt = self.find_subgroup(
            X,
            y,
            verbose = verbose,
            n_min = n_min,
            time_limit = time_limit,
            bias_type = bias_type,
            group_type = group_type,
            probabilistic = probabilistic,
            return_opt_flag = True,
            warmstart_model = warmstart_model,
            gamma = gamma,
            solver = solver,
        )
        if group_desc is None:
            ingroup = None
        elif group_type == "rule":
            rule = group_desc
            ingroup = np.ones_like(y, dtype=bool)
            for feat_i in rule:
                ingroup &= X[:, feat_i]
        elif group_type == "linear":
            coeffs, thresh = group_desc[:-1], group_desc[-1]
            ingroup = X @ coeffs >= thresh
        else:
            raise ValueError(f"Unsupported group type {group_type}")

        if return_opt_flag:
            return group_desc, ingroup, opt
        else:
            return group_desc, ingroup


