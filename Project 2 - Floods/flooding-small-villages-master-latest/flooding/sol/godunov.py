"""Godunov solver wrapping for :class:`flooding.BasicFloodSim`, adds
various utilities and binding points for numerical solving in the simple
case, built for extendability.
"""

from flooding import BasicFloodSim, river_shapes

from typing import Union, Tuple, Callable, Any

import numpy as np
import matplotlib.pyplot as plt


class BasicGodunovSolver(BasicFloodSim):
    def __init__(
            self,
            domain: Tuple[float, float],
            domain_n: int,
            init_area_func: Callable[[np.ndarray, Any], np.ndarray],
            friction_coefficient: Union[float, np.ndarray],
            river_slope_angle_alpha: Union[float, np.ndarray],
            river_shape: river_shapes.RiverBed(),
            *args, **kwargs):
        """Wrapping solver based on Godunov Method of Characteristics
        as shown in Ian's Panopto lecture.

        This solver must first be initialised with the parameters of
        interest, and then can compute a solution with the
        :py:meth:`solve` method.

        :param domain: 2-tuple of min and max domain values
        :param domain_n: Length of "internal" domain, boundary points
            not included in this count.
        :param init_area: Initial area array, must be of length
            domain_n+2
        :param friction_coefficient: Fluid friction factor
        :param river_slope_angle_alpha: Slope of river
        :param river_shape: Object representing the shape of the river,
            accepts a subclass of RiverBed, *args are currently purely
            for the initialisation of this river shape object.
        :param kwargs: May be used for both initialising the river
            shape object and the area array, ensure no overlap in
            kwarg names of custom initialisation functions.
        """
        domain = np.linspace(*domain, num=domain_n+2)
        init_area = init_area_func(domain, **kwargs)
        # Most model configuration and initialisation is handled
        # by the parent class (BasicFloodSim)
        super().__init__(
            domain, init_area, friction_coefficient,
            river_slope_angle_alpha, river_shape,
            *args, **kwargs
        )
        self.dt = None
        self.curr_time = 0.

    def solve(self, time_forward):
        """Solves from current time (self.curr_time) to go
        time_forward time forwards.

        :param time_forward: amount of time for the model to
            increment forward from self.curr_time.
        """
        internal_time = 0.
        while internal_time < time_forward:
            # debugging code
            # if np.min(self.area) <= 0:
            #     print(self.area)
            self.dt = self.d_domain/self.max_velocity
            # Avoid going over the max time by checking to see if the
            # next t + dt will extend beyond user defined bounds.
            if internal_time + self.dt > time_forward:
                self.dt = time_forward - internal_time
            self.area = self.__step()
            self.apply_boundary_cons()
            internal_time += self.dt
        self.curr_time += time_forward

    def step_solve(self):
        """Steps the solver  one dt forward,
        increments curr_time internally.
        """
        self.dt = self.d_domain/self.max_velocity
        self.area = self.__step()
        self.apply_boundary_cons()
        self.curr_time += self.dt

    def __step(self):
        flux = self.flux  # Saves getting property twice
        # Update term comes from the delta of flux and requires
        # the river only flows from left to right
        dA = np.zeros_like(self.area)
        dA[1:] = flux[:-1] - flux[1:]
        return np.add(
            self.area,
            self.dt * np.divide(dA, self.d_domain)
        )

    @property
    def d_domain(self):
        return abs(self.domain[1] - self.domain[0])

    def apply_boundary_cons(self):
        self.area[0] = self.area[1]
        self.area[-1] = self.area[-2]


if __name__ == '__main__':
    test_solver = BasicGodunovSolver(
        (-4., 4.), 1000, lambda domain_vect: np.tanh(-domain_vect*5.) + 2.,
        0.6, np.pi/16., river_shapes.ParabolicBed, 1.
    )
    fig, ax = plt.subplots()
    ax.plot(test_solver.domain, test_solver.area)
    prior_v = test_solver.velocity
    test_solver.solve(0.5)
    final_v = test_solver.velocity
    v_diff = np.subtract(final_v, prior_v)
    print(np.mean(v_diff))
    ax.plot(test_solver.domain, test_solver.area)
    fig.savefig("./godunov_test_plot.pdf")
