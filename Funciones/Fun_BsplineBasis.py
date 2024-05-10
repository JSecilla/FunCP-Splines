import math
from typing import Dict, Union

import numpy as np
from scipy.interpolate import BSpline

class Fun_BsplineBasis:

    def __init__(
        self,
        xsample: np.ndarray,
        B: np.ndarray, #Lo creamos para poder meterle una Base de input
        deg: int = 3,
        n_int: int = 40,
        prediction: Dict[str, Union[int, float]] = {},
    ):
        self.xsample = xsample
        self.deg = deg
        self.n_int = n_int
        self.prediction = prediction
        self.B = B

    def get_matrix_B(self) -> None:
        
        min_x = self.xsample.min()
        max_x = self.xsample.max()
        if self.deg < 0:
            raise ValueError("The degree of the B-spline basis must be at least 0.")
        if self.xsample.ndim != 1:
            raise ValueError("Regressor vector must be one-dimensional.")
        if self.n_int < 2:
            raise ValueError(
                "The fitting regions must be split in at least 2 intervals."
            )
        if len(set(self.prediction) - set(["backwards", "forward"])) > 0:
            raise ValueError(
                "Prediction only admits as keys `forward` and `backwards`."
            )

        if "backwards" in self.prediction:
            if self.prediction["backwards"] >= min_x:
                raise ValueError(
                    (
                        "Backwards prediction limit must stand on the "
                        "left-hand side of the regressor vector."
                    )
                )
        if "forward" in self.prediction:
            if self.prediction["forward"] <= max_x:
                raise ValueError(
                    (
                        "Forward prediction limit must stand on the "
                        "right-hand side of the regressor vector."
                    )
                )

        # Compute the distance between adjacent knots
        step_length = (max_x - min_x) / self.n_int

        # Determine how many knots at the left (right) of min(`xsample`)
        # (max(`xsample`)) are needed to extend the basis backwards (forward).
        # The step length between these new knots must be the same
        self.int_back = (
            math.ceil((min_x - self.prediction["backwards"]) / step_length)
            if "backwards" in self.prediction
            else 0
        )

        self.int_forw = (
            math.ceil((self.prediction["forward"] - max_x) / step_length)
            if "forward" in self.prediction
            else 0
        )
        # Construct the knot sequence of the B-spline basis, consisting on
        # `n_int` + 2 * `deg` + 1 equally spaced knots
        knots = np.linspace(
            min_x - (self.int_back + self.deg) * step_length,
            max_x + (self.int_forw + self.deg) * step_length,
            self.n_int + self.int_back + self.int_forw + 2 * self.deg + 1,
        )
        # To avoid floating point error, force that the (`deg` + 1)-th knot
        # coincides with the minimum value of `xsample` and the
        # (`n_int` + `deg` + 1) matches the maximum value of `xsample`
        knots[self.int_back + self.deg] = min_x
        knots[-(self.int_forw + self.deg + 1)] = max_x
        self.knots = knots

        # Construct the B-spline basis, consisting on
        # (`n_int` + `int_back` + `int_forw` + `deg`) elements
        self.bspline_basis = BSpline.construct_fast(
            t=self.knots, c=self.B, k=self.deg
        )
        # Return the design matrix of the B-spline basis
        x_eval = np.concatenate(
            [
                self.knots[self.deg : self.deg + self.int_back],
                self.xsample,
                self.knots[self.int_back + self.n_int + self.deg + 1 : -self.deg],
            ]
        )
        self.matrixB = self.B
        return None

    def get_matrices_S(self):
        
        C = np.zeros(shape=(self.deg + 1, self.deg + 1))
        for i in range(self.deg + 1):
            C[:, i] = BSpline.basis_element(t=self.knots[i : self.deg + 2 + i])(
                x=np.linspace(
                    self.knots[self.deg], self.knots[self.deg + 1], self.deg + 1
                )
            )
        S = []
        # The matrices S_k that we are looking for satisfy that S_k @ T_k = C,
        # where T_k has as columns the array (1, x, x**2, ... x**`deg`)
        # evaluated at the points
        #     linspace(knots[k + `deg`], knots[k + `deg` + 1], `deg` + 1)
        for k in range(self.n_int + self.int_back + self.int_forw):
            T_k = np.vander(
                np.linspace(
                    self.knots[k + self.deg], self.knots[k + self.deg + 1], self.deg + 1
                ),
                increasing=True,
            )
            S_k = np.linalg.solve(T_k, C)
            S.append(S_k)
        self.matrices_S = S
        return None
