"""
Length-Area relating functions that describe the shape of the
river bed, ensure functions follow the signature:
(area: Union[np.ndarray, float], *args, **kwargs) ->
    length: Union[np.ndarray, float]

See preexisting functions for reference.
"""

import numpy as np

from abc import ABC, abstractmethod


class RiverBed(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def wetted_length(self, *args, **kwargs):
        pass

    def river_depth(self, *args, **kwargs):
        pass


class ParabolicBed(RiverBed):
    def __init__(self, river_slope):
        self.riverbank_slope = river_slope

    def wetted_length(self, water_area):
        river_depth = np.power(
            np.multiply(
                (9. / 16.) * water_area,
                self.riverbank_slope
            ),
            (1. / 3.)
        )

        return np.add(
            np.sqrt(
                np.add(
                    4. * np.square(river_depth),
                    np.divide(river_depth, self.riverbank_slope)
                )
            ),
            np.divide(
                np.arcsinh(
                    2. * np.sqrt(
                        np.multiply(river_depth, self.riverbank_slope))
                ),
                2. * self.riverbank_slope
            )
        )

    def river_depth(self, water_area):
        return np.power(
            np.multiply(
                (9. / 16.) * water_area,
                self.riverbank_slope
            ),
            (1. / 3.)
        )


class RectangleBed(RiverBed):
    def __init__(self, river_width):
        self.river_width = river_width

    def wetted_length(self, area):
        return np.add(
            self.river_width,
            np.divide(
                2 * area,
                self.river_width
            )
        )

    def river_depth(self, area):
        return np.divide(area, self.river_width)
