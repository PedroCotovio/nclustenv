
from .base import BaseEnv
from nclustenv.utils.states import State, OfflineState
from nclustenv.utils.helper import tensor_to_string, index_to_tensor

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
            penalty=0.001,
            init_state=True
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

        if init_state:
            self.state = State(generator='TriclusterGenerator', n=n, np_random=self.np_random)
            self.reset()

    def _render(self, index):
        print(tensor_to_string(index_to_tensor(self.state.as_dense, index), index))


class OfflineTriclusterEnv(TriclusterEnv):
    """
    This class provides an implementation of an offline two-dimensional gym environment with hidden biclusters.
    """

    def __init__(
            self,
            dataset,
            n=None,
            seed=None,
            metric='match_score',
            action='Action',
            max_steps=200,
            error_margin=0.05,
            penalty=0.001,
            train_test_split=0.8
    ):
        super(OfflineTriclusterEnv, self).__init__(
            shape=None,
            n=None,
            clusters=None,
            dataset_settings=None,
            seed=seed,
            metric=metric,
            action=action,
            max_steps=max_steps,
            error_margin=error_margin,
            penalty=penalty,
            init_state=False
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
