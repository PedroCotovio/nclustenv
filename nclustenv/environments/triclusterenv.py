
from .base import BaseEnv
from ..utils.states import State
from ..utils.helper import tensor_to_string, index_to_tensor

from gym import spaces
import numpy as np


class TriclusterEnv(BaseEnv):

    """
    This class provides an implementation of a three-dimensional gym environment with hidden triclusters.
    """

    def __init__(
            self,
            shape=None,
            n=None,
            clusters=None,
            dataset_settings=None,
            seed=None,
            metric='match_score',
            action='Action',
            max_steps=200,
            error_margin=0.05,
            penalty=0.001
    ):

        if shape is None:
            shape = [[100, 100, 2], [200, 200, 5]]

        # Enforce shape size

        if len(shape) != 3:
            raise AttributeError('Shape does not produce a tridimensional dataset')

        # Enforce ctx > 1
        if shape[0][-1] < 2:
            shape[0][-1] = 2
        elif shape[1][-1] < 2:
            shape[1][-1] = 2

        super(TriclusterEnv, self).__init__(
            shape=shape,
            n=None,
            clusters=clusters,
            dataset_settings=dataset_settings,
            seed=seed,
            metric=metric,
            action=action,
            max_steps=max_steps,
            error_margin=error_margin,
            penalty=penalty
        )

        self.state = State(generator='TriclusterGenerator', n=n, np_random=self.np_random)
        self.reset()

    def _render(self, index):
        print(tensor_to_string(index_to_tensor(self.state.as_dense, index), index))
