
from .base import BaseEnv
from nclustenv.utils.states import State, OfflineState
from nclustenv.utils.helper import matrix_to_string, index_to_matrix

from gym import spaces
import numpy as np

from ...utils.datasets import SyntheticDataset


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
            penalty=0.001,
            init_state=True,
            *args, **kwargs
    ):

        if shape is None:
            shape = [[100, 100], [200, 200]]

        if len(shape[0]) != 2:
            raise AttributeError('Shape does not produce a bidimensional dataset')

        super(BiclusterEnv, self).__init__(
            shape=shape,
            n=n,
            clusters=clusters,
            dataset_settings=dataset_settings,
            seed=seed,
            metric=metric,
            action=action,
            max_steps=max_steps,
            error_margin=error_margin,
            penalty=penalty,
            *args, **kwargs
        )

        if init_state:
            self.state = State(generator='BiclusterGenerator', n=n, np_random=self.np_random)
            self.reset()

    def _render(self, index):
        print(matrix_to_string(index_to_matrix(self.state.as_dense, index), index))


class OfflineBiclusterEnv(BiclusterEnv):
    """
    This class provides an implementation of an offline two-dimensional gym environment with hidden biclusters.
    """

    def __init__(
            self,
            dataset: SyntheticDataset,
            n=None,
            seed=None,
            metric='match_score',
            action='Action',
            max_steps=200,
            error_margin=0.05,
            penalty=0.001,
            train_test_split=0.8,
            *args, **kwargs
    ):

        super(OfflineBiclusterEnv, self).__init__(
            shape=dataset.shape,
            n=n,
            clusters=dataset.clusters,
            dataset_settings=dataset.settings,
            seed=seed,
            metric=metric,
            action=action,
            max_steps=max_steps,
            error_margin=error_margin,
            penalty=penalty,
            init_state=False,
            *args, **kwargs
        )

        self.state = OfflineState(dataset=dataset, train_test_split=train_test_split, n=n, np_random=self.np_random)
        self.reset()

    def reset(self, train=True):
        """
        Resets the environment to an initial state and returns an initial observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns
        -------
            observation (object)
                The initial observation.
        """

        # reset loggers
        self._current_step = 0
        self._steps_beyond_done = 0
        self._last_distances = [1.0, 1.0, 1.0]
        self._done = False

        return self.state.reset(train=train)
