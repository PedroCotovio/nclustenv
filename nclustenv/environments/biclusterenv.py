
from .base import BaseEnv
from ..utils.states import State
from ..utils.helper import matrix_to_string, index_to_matrix

from gym import spaces
import numpy as np


class BiclusterEnv(BaseEnv):

    """
    This class provides an implementation of a two-dimensional gym environment with hidden biclusters.
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
            shape = [[100, 100], [200, 200]]

        if len(shape) != 2:
            raise AttributeError('Shape does not produce a bidimensional dataset')

        super(BiclusterEnv, self).__init__(
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

        self.action_space = spaces.Tuple((spaces.Discrete(2),
                                          spaces.Discrete(2),
                                          spaces.Box(low=0.0, high=1.0, shape=[1, ], dtype=np.float16)))

        self.state = State(generator='BiclusterGenerator', n=n, np_random=self.np_random)
        self.reset()

    def _render(self, index):
        print(matrix_to_string(index_to_matrix(self.state.as_dense, index), index))
