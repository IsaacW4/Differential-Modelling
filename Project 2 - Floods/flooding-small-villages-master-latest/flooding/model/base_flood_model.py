"""
Base equations of the flooding system, solver not to be integrated
in this file, indicated changes in commit messages and please
ensure that no breaking changes are made unless absolutely
necessary. If extending the system of equations, create a
separate ".py" file to contain them, please ensure it's well
documented if you intend to make it available to others.
"""

from flooding.model.river_shapes import RiverBed

import numpy as np
from scipy import constants
from typing import Union


class BasicFloodSim:
    def __init__(
            self,
            domain: np.ndarray,
            init_area: np.ndarray,
            friction_coefficient: Union[float, np.ndarray],
            river_slope_angle_alpha: Union[float, np.ndarray],
            river_shape: RiverBed(),
            *args,
            **kwargs
    ):
        """Basic flooding modelling object, initialised with parameters
        of interest to model, all :class:`np.ndarray` arguments must
        be of same cast as domain or floats.

        :arg domain: domain over which the river is modelled.
        :arg river_shape: accepts a subclass of RiverBed, *args and
            **kwargs are currently purely for the initialisation of
            this river shape object.
        """
        self.domain = domain
        self.area = init_area
        self.friction_coefficient = friction_coefficient
        self.riverbed = river_shape(*args, **kwargs)
        self.alpha = river_slope_angle_alpha
        self.g = constants.g
        self.rho = 1000.

    @property
    def kappa(self):
        """Since there may be cases where the river slope may vary
        along the domain, kappa must be able to handle such arrays.
        """
        return np.sqrt(
            np.multiply((self.g/self.friction_coefficient), np.divide(
                np.tan(self.alpha),
                self.riverbed.wetted_length(self.area)
            ))
        )

    @property
    def flux(self):
        return np.multiply(self.kappa, np.power(self.area, (3. / 2.)))

    @property
    def velocity(self):
        return np.multiply(self.kappa, np.sqrt(self.area * self.rho))

    @property
    def max_velocity(self):
        return np.max(self.velocity)
