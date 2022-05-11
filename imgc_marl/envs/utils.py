from typing import List, Optional, Tuple, Union

import numpy as np
from simple_playgrounds.common.position_utils import Coordinate, CoordinateSampler


class MetaSampler(CoordinateSampler):
    """
    Constructs a sampler that samples a position from different CoordinateSamplers with a weight on each
    of them
    """

    def __init__(
        self,
        samplers: List[CoordinateSampler],
        weights: List[float] = None,
    ):
        if weights is None:
            areas = []
            for sampler in samplers:
                areas.append(sampler._length * sampler._width)
            weights = np.array(areas) / np.array(areas).sum()
        assert sum(weights) == 1
        self.weights = weights
        self.samplers = samplers
        self.sampler_indexes = range(len(samplers))

    def sample(self, coordinates: Optional[Coordinate] = None) -> Coordinate:
        sampler = self.samplers[
            np.random.choice(self.sampler_indexes, 1, p=self.weights)[0]
        ]
        return sampler.sample(coordinates)
