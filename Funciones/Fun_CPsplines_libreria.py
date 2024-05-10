import itertools
import logging
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mosek.fusion
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy.spatial import Delaunay
from statsmodels.genmod.families.family import Binomial, Family, Gaussian, Poisson

from cpsplines.mosek_functions.interval_constraints import IntConstraints
from cpsplines.mosek_functions.obj_function import ObjectiveFunction
from cpsplines.mosek_functions.pdf_constraints import PDFConstraint
from cpsplines.mosek_functions.point_constraints import PointConstraints
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from cpsplines.utils.box_product import box_product
from cpsplines.utils.fast_kron import matrix_by_transpose
from cpsplines.utils.gcv import GCV
from cpsplines.utils.normalize_data import DataNormalizer
from cpsplines.utils.rearrange_data import RearrangingError, scatter_to_grid
from cpsplines.utils.simulator_grid_search import print_grid_search_results
from cpsplines.utils.simulator_optimize import Simulator
from cpsplines.utils.timer import timer
from cpsplines.utils.weighted_b import get_idx_fitting_region
from Fun_BsplineBasis import Fun_BsplineBasis


class Fun_CPsplines:
    
    def __init__(
        self,
        deg: Iterable[int] = (3,),
        ord_d: Iterable[int] = (2,),
        n_int: Iterable[int] = (40,),
        B: np.ndarray = None,
        x_range: Optional[Dict[str, Tuple[Union[int, float]]]] = None,
        sp_method: str = "optimizer",
        sp_args: Optional[Dict[str, Any]] = None,
        family: str = "gaussian",
        int_constraints: Optional[
            Dict[str, Dict[int, Dict[str, Union[int, float]]]]
        ] = None,
        pt_constraints: Optional[Dict[Tuple[int], Dict[str, pd.DataFrame]]] = None,
        pdf_constraint: bool = False,
    ):
        self.deg = deg
        self.ord_d = ord_d
        self.n_int = n_int
        self.x_range = x_range
        self.sp_method = sp_method
        self.sp_args = sp_args
        self.family = self._get_family(family)
        self.int_constraints = int_constraints
        self.pt_constraints = pt_constraints
        self.pdf_constraint = pdf_constraint
        self.B = B
        
    @staticmethod
    def _get_family(family: str) -> Family:
        
        if family == "gaussian":
            family_statsmodels = Gaussian()
        else:
            raise ValueError(f"Family {family} is not implemented.")
        return family_statsmodels
    
    
    def _get_bspline_bases(self, x: Iterable[np.ndarray]) -> List[Fun_BsplineBasis]:
    
        bspline_bases = []
        if self.x_range is None:
            self.x_range = {}
        for deg, xsample, n_int, name in zip(
            self.deg, x, self.n_int, self.feature_names
        ):
            # Get the maximum and minimum of the fitting regions
            x_min, x_max = np.min(xsample), np.max(xsample)
            prediction_dict = {}
            if name in self.x_range:
                # If the values in `x_range` are outside the fitting region,
                # include them in the `prediction` argument of the BsplineBasis
                pred_min, pred_max = min(self.x_range[name]), max(self.x_range[name])
                if pred_max > x_max:
                    prediction_dict["forward"] = pred_max
                if pred_min < x_min:
                    prediction_dict["backwards"] = pred_min
            bsp = Fun_BsplineBasis(
                deg=deg, xsample=xsample, n_int=n_int, prediction=prediction_dict, B= self.B
            )
            # Generate the design matrix of the B-spline basis
            bsp.get_matrix_B()
            if self.int_constraints is not None or self.pdf_constraint:
                bsp.get_matrices_S()
            bspline_bases.append(bsp)
        return bspline_bases
    
    def _fill_sp_args(self):
        """
        Fill the `sp_args` dictionary by default parameters on the case they are
        not provided.
        """
        if self.sp_args is None:
            self.sp_args = {}
        if self.sp_method == "grid_search":
            self.sp_args["grid"] = self.sp_args.get(
                "grid", [(0.01, 0.1, 1, 10) for _ in range(len(self.deg))]
            )
            self.sp_args["parallel"] = self.sp_args.get("parallel", False)
            self.sp_args["n_jobs"] = self.sp_args.get("n_jobs", -2)
            self.sp_args["top_n"] = self.sp_args.get("top_n", None)
        else:
            self.sp_args["verbose"] = self.sp_args.get("verbose", False)
            self.sp_args["x0"] = self.sp_args.get("x0", np.ones(len(self.deg)))
            self.sp_args["method"] = self.sp_args.get("method", "SLSQP")
            self.sp_args["bounds"] = self.sp_args.get(
                "bounds", [(1e-10, 1e16) for _ in range(len(self.deg))]
            )
        return None
    
    def _get_obj_func_arrays(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        obj_matrices = {}
        obj_matrices["B"] = []
        obj_matrices["D"] = []
        obj_matrices["D_mul"] = []
        
        B = self.B
        obj_matrices["B"].append(B)
        n = np.shape(B)[1]
        D = np.diag(np.ones(n-1), k=-1) + np.diag(-2 * np.ones(n), k=0) + np.diag(np.ones(n-1), k=1)
        D = D[1:-1, :]
        P = P = np.transpose(D) @ D
        # P_2 = np.zeros((n+1, n+1))
        # P_2[1:,1:] = P
        # n_2 = n + 1
        # D_2 = np.diag(np.ones(n_2-1), k=-1) + np.diag(-2 * np.ones(n_2), k=0) + np.diag(np.ones(n_2-1), k=1)
        # D_2 = D_2[1:-1, :]

        obj_matrices["D"].append(D)
        obj_matrices["D_mul"].append(P)
        obj_matrices["y"] = y.copy()
        return obj_matrices
    
        # The extended response variable sample dimensions can be obtained as
        # the number of rows of the design matrix B
        # indexes_fit = get_idx_fitting_region(self.bspline_bases)
        # for bsp, ord_d, idx in zip(self.bspline_bases, self.ord_d, indexes_fit):
            #B = bsp.matrixB
            #obj_matrices["B"].append(B[idx])
            #penaltymat = PenaltyMatrix(bspline=bsp)
            #P = penaltymat.get_penalty_matrix(**{"ord_d": ord_d})
            #obj_matrices["D"].append(penaltymat.matrixD)
            #obj_matrices["D_mul"].append(P)

        #obj_matrices["y"] = y.copy()
        #return obj_matrices
    
    def _initialize_model(
        self,
        obj_matrices: Union[np.ndarray, Iterable[np.ndarray]],
        y_col: str,
        data_normalizer: Optional[DataNormalizer] = None,
    ) -> mosek.fusion.Model:
    
        M = mosek.fusion.Model()
        # Create the variables of the optimization problem
        mos_obj_f = ObjectiveFunction(bspline=self.bspline_bases, model=M)
        # For each axis, a smoothing parameter is needed
        sp = [M.parameter(f"sp_{i}", 1) for i, _ in enumerate(self.deg)]
        # Build the objective function of the problem
        mos_obj_f.create_obj_function(
            obj_matrices=obj_matrices,
            sp=sp,
            family=self.family,
            data_arrangement=self.data_arrangement,
        )

        if self.int_constraints is not None:
            max_deriv = max([max(v.keys()) for v in self.int_constraints.values()])
            matrices_S = {
                name: bsp.matrices_S
                for name, bsp in zip(self.feature_names, self.bspline_bases)
            }
            # Iterate for every variable with constraints and for every
            # derivative order
            for var_name in self.int_constraints.keys():
                for deriv, constraints in self.int_constraints[var_name].items():
                    if list(constraints.values())[0] != 0 and not isinstance(
                        self.family, Gaussian
                    ):
                        raise ValueError(
                            "No threshold is allowed in the shape constraints for non Gaussian data."
                        )
                    # Scale the integer constraints thresholds in the case the
                    # data is scaled
                    if data_normalizer is not None:
                        derivative = True if deriv != 0 else False
                        constraints = {
                            k: data_normalizer.transform(y=v, derivative=derivative)
                            for k, v in constraints.items()
                        }
                    matrices_S_copy = matrices_S.copy()
                    # Build the interval constraints
                    cons = IntConstraints(
                        bspline={
                            name: bsp
                            for name, bsp in zip(self.feature_names, self.bspline_bases)
                        },
                        var_name=var_name,
                        derivative=deriv,
                        constraints=constraints,
                    )
                    cons.interval_cons(
                        var_dict=mos_obj_f.var_dict, model=M, matrices_S=matrices_S_copy
                    )
        else:
            self.int_constraints = {}

        if self.pt_constraints is not None:

            # Iterate for every combination of the derivative orders where
            # constraints must be enforced
            for deriv, dict_deriv in self.pt_constraints.items():
                for sense, data in dict_deriv.items():
                    # Scale the point constraints thresholds in the case the data is
                    # scaled
                    if data_normalizer is not None:
                        derivative = any(v != 0 for v in deriv)
                        data = data.assign(
                            y=data_normalizer.transform(
                                y=data[y_col], derivative=derivative
                            )
                        )
                        if "tol" in data.columns:
                            data = data.assign(
                                tol=data_normalizer.transform(
                                    y=data["tol"], derivative=False
                                )
                                - data_normalizer.transform(y=0, derivative=False)
                            )
                    cons2 = PointConstraints(
                        derivative=deriv,
                        sense=sense,
                        bspline=self.bspline_bases,
                    )
                    cons2.point_cons(
                        data=data,
                        y_col=y_col,
                        var_dict=mos_obj_f.var_dict,
                        model=M,
                    )
        else:
            self.pt_constraints = {}
        return M
    
    def _get_sp_grid_search(
        self,
        obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    ) -> Tuple[Union[int, float]]:
        """
        Get the best smoothing parameter vector with the GCV minimizer criteria
        using grid search selection.

        Parameters
        ----------
        B_weighted : Iterable[np.ndarray]
            The weighted design matrix from the B-spline basis.
        Q_matrices : Iterable[np.ndarray]
            The array of matrices used in the GCV computation.
        y : np.ndarray
            The extended response variable sample.

        Returns
        -------
        Tuple[Union[int, float]]
            The best set of smoothing parameters
        """

        # Computes all the possible combinations for the smoothing parameters
        iter_sp = list(itertools.product(*self.sp_args["grid"]))
        # Run in parallel if the argument `parallel` is present
        if self.sp_args["parallel"]:
            gcv = Parallel(n_jobs=self.sp_args["n_jobs"])(
                delayed(GCV)(sp, obj_matrices, self.family, self.data_arrangement)
                for sp in iter_sp
            )
        else:
            gcv = [
                GCV(
                    sp=sp,
                    obj_matrices=obj_matrices,
                    family=self.family,
                    data_arrangement=self.data_arrangement,
                )
                for sp in iter_sp
            ]
        # Print the `top_n` combinations that minimizes the GCV
        if self.sp_args["top_n"] is not None:
            print_grid_search_results(
                x_val=iter_sp, obj_val=gcv, top_n=self.sp_args["top_n"]
            )

        return iter_sp[gcv.index(min(gcv))]
    
    def _get_sp_optimizer(
        self,
        obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    ) -> Tuple[Union[int, float]]:

        # All argument in `sp_args` except `verbose` are arguments from
        # scipy.optimize.minimize, so create a copy without it
        scipy_optimize_params = self.sp_args.copy()
        scipy_optimize_params.pop("verbose", None)
        # Create a simulator to print the intermediate steps of the process if
        # "verbose" is active
        if self.sp_args["verbose"]:
            gcv_sim = Simulator(GCV)
        # Get the best set of smoothing parameters
        best_sp = scipy.optimize.minimize(
            gcv_sim.simulate if self.sp_args["verbose"] else GCV,
            args=(obj_matrices, self.family, self.data_arrangement),
            callback=gcv_sim.callback if self.sp_args["verbose"] else None,
            **scipy_optimize_params,
        ).x
        return best_sp
    
    def _preprocessor(
        self, data: pd.DataFrame, y_col: str
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        
        self.data_arrangement = "scattered"
        x = [row for row in data[data.columns.drop(y_col).tolist()].values.T]
        y = data[y_col].values
        try:
            z, t = scatter_to_grid(data=data, y_col=y_col)
            if len(data) == np.prod(t.shape) and np.isnan(t).sum() == 0:
                self.data_arrangement = "gridded"
                x, y = z.copy(), t.copy()
                logging.info("Data is rearranged into a grid.")
        except RearrangingError:
            pass
        return x, y
    
    def fit(
        self,
        data: pd.DataFrame,
        y_col: str,
        y_range: Optional[Iterable[Union[int, float]]] = None,
        **kwargs,
    ):
        """
        Compute the fitted decision variables of the B-spline expansion and the
        fitted values for the response variable. The kwargs are referred to
        MOSEK parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Input data and target data.
        y_col : str
            The column name of the target variable.
        y_range : Optional[Iterable[Union[int, float]]]
            If not None, `y` is scaled in the range defined by this parameter.
            This scaling process is useful when `y` has very large norm, since
            MOSEK may not be able to find a solution in this case due to
            numerical issues. By default, None.

        Raises
        ------
        ValueError
            If the degree, difference order and number of intervals differ.
        ValueError
            If `sp_method` input is different from "grid_search" or "optimizer".
        NumericalError
            If MOSEK could not arrive to a feasible solution.
        """

        if len({len(i) for i in [self.deg, self.ord_d, self.n_int]}) != 1:
            raise ValueError("The lengths of `deg`, `ord_d`, `n_int` must agree.")

        if self.sp_method not in ["grid_search", "optimizer"]:
            raise ValueError(f"Invalid `sp_method`: {self.sp_method}.")

        self.feature_names = data.drop(columns=y_col).columns

        if data.shape[1] > 2:
            df_pred = [data.drop(columns=y_col)]
            # When out-of-sample prediction is considered, the convex hull must
            # be extended till the prediction horizon for the whole range of the
            # remaining variables
            if self.x_range:
                for key, value in self.x_range.items():
                    column_name = data.loc[:, key].name
                    for v in value:
                        df_pred.append(
                            data.drop(columns=[y_col, column_name])
                            .agg(["min", "max"])
                            .assign(**{column_name: v})
                        )
            self.data_hull = Delaunay(pd.concat(df_pred))

        x, y = self._preprocessor(data=data, y_col=y_col)
        
        # Construct the B-spline bases
        self.bspline_bases = self._get_bspline_bases(x=x)

        # Filling the arguments of the method used to determine the optimal set
        # of smoothing parameters
        _ = self._fill_sp_args()
        if y_range is not None:
            if not isinstance(self.family, Gaussian):
                raise ValueError(
                    "The argument `y_range` is only available for Gaussian data."
                )
            if len(y_range) != 2:
                raise ValueError("The range for `y` must be an interval.")
            data_normalizer = DataNormalizer(feature_range=y_range)
            _ = data_normalizer.fit(y)
            y = data_normalizer.transform(y)
        else:
            data_normalizer = None

        # Get the matrices used in the objective function
        obj_matrices = self._get_obj_func_arrays(y=y)

        # Auxiliary matrices derived from `obj_matrices`
        obj_matrices["B_mul"] = list(map(matrix_by_transpose, obj_matrices["B"]))

        # Initialize the model
        M = self._initialize_model(
            obj_matrices=obj_matrices, y_col=y_col, data_normalizer=data_normalizer
        )
        model_params = {"theta": M.getVariable("theta")}
        for i, _ in enumerate(self.deg):
            model_params[f"sp_{i}"] = M.getParameter(f"sp_{i}")

        if self.sp_method == "grid_search":
            self.best_sp = self._get_sp_grid_search(
                obj_matrices=obj_matrices,
            )
        else:
            self.best_sp = self._get_sp_optimizer(
                obj_matrices=obj_matrices,
            )
        theta_shape = model_params["theta"].getShape()
        # Set the smoothing parameters vector as the optimal obtained in the
        # unconstrained setting
        for i, sp in enumerate(self.best_sp):
            model_params[f"sp_{i}"].setValue(sp)
        for key, v in kwargs.items():
            _ = M.setSolverParam(key, v)
        try:
            # Solve the problem
            with timer(
                tag=f"Solve the problem with smoothing parameters {tuple(self.best_sp)}: "
            ):
                M.solve()
            # Extract the fitted decision variables of the B-spline expansion
            self.sol = model_params["theta"].level().reshape(theta_shape)
            if y_range is not None:
                self.sol = data_normalizer.inverse_transform(y=self.sol)
        except mosek.fusion.SolutionError as e:
            raise NumericalError(
                f"The solution for the smoothing parameter {self.best_sp} "
                f"could not be found due to numerical issues. The original error "
                f"was: {e}"
            )

        return None
    
