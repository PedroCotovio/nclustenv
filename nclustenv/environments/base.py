import abc
from abc import ABC
from statistics import mean
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

from ..utils import actions, metrics
from ..utils.helper import loader

# TODO test & docs
class BaseEnv(gym.Env, ABC):

    """
    Bi/Tricluster environment
    """

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            shape,
            clusters=None,
            dataset_settings=None,
            seed=None,
            metric='match_score_1_n',
            action='Action',
            max_steps=200,
            error_margin=0.05,
            penalty=0.001
    ):

        super(BaseEnv, self).__init__()

        # Environment attributes

        if clusters is None:
            clusters = [1, 1]

        if dataset_settings is None:
            dataset_settings = {}

        # Enforce fixed settings
        dataset_settings['silence'] = True
        dataset_settings['in_memory'] = True

        self._clusters = clusters
        self.dataset_settings = dataset_settings

        # metric pointer
        self._metric = loader(metric)

        # action pointer
        self._action = loader(action)

        self.max_steps = max_steps
        self.target = error_margin
        self.penalty = penalty

        self.action_space = None
        self.observation_space = spaces.Box(low=np.array(shape[0]), high=np.array(shape[1]), dtype=np.int32)

        # Init

        self._last_distances = None
        self._current_step = None
        self._steps_beyond_done = None
        self._done = None

        self.np_random = None
        self.state = None

        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        """
        Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for calling
        `reset()` to reset this environment's state. Accepts an action and returns a tuple (observation, reward, done,
        info).

        Parameters
        ----------

        action: list
            An action provided by the agent.

        Returns
        -------

            object
                Agent's observation of the current environment.
            float
                Amount of reward returned after previous action.
            bool
                Whether the episode has ended, in which case further step() calls will return undefined results.
            dict
                Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """

        if not self._done:
            self._current_step += 1
            action_ = self._action(*action)

            # Take action
            getattr(self.state, action_.action)(action_.ntype, action_.parameter)

            # calculate volume match
            self._last_distances.pop(0)
            self._last_distances.append(1-self.volume_match)

            # check state

            if self._last_distances[-1] == 0.0:
                reward = self.get_reward(self._last_distances, True)
                self._done = True
            elif mean(self._last_distances) <= self.target:
                reward = self.get_reward(self._last_distances, True, True)
                self._done = True
            elif self._current_step > self.max_steps:
                reward = -1.0
                self._done = True
            else:
                reward = self.get_reward(self._last_distances)

        else:
            if self._steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned _done = True. You "
                    "should always call 'reset()' once you receive '_done = "
                    "True' -- any further steps are undefined behavior."
                )

            self._steps_beyond_done += 1
            reward = 0.0

        return self.state.current, reward, self._done, {}

    def get_reward(self, last_distances, goal=False, error=False):

        """
        Returns the reward for the current state.

        Returns
        -------

            float
                Current reward.

        """

        return float(last_distances[-2] - last_distances[-1] - self.penalty + (2 if goal else 0) - (1 if error else 0))

    @property
    def volume_match(self):

        """
        Returns the volume match for the current state.

        Returns
        -------

            float
                Current volume match.

        """

        return min(self._metric(self.state.cluster, self.state.hclusters))

    @property
    def best_match(self):
        matches = self._metric(self.state.cluster, self.state.hclusters)
        return matches.index(min(matches))

    def reset(self):

        # reset loggers
        self._current_step = 0
        self._steps_beyond_done = 0
        self._last_distances = [1.0, 1.0, 1.0]
        self._done = False

        # reset seed
        self.dataset_settings['seed'] = self.np_random.randint(low=1, high=10 ** 9, dtype=np.int32)

        # define shape
        try:
            shape = self.np_random.randint(low=self.observation_space.low, high=self.observation_space.high)
        except ValueError:
            shape = self.observation_space.low

        # define nclusters
        try:
            nclusters = self.np_random.randint(*self._clusters)
        except ValueError:
            nclusters = self._clusters[0]

        self.state.reset(shape=shape, nclusters=nclusters, settings=self.dataset_settings)

    @abc.abstractmethod
    def _render(self, index):
        pass

    # TODO correct render
    # If cluster not array (ax have different len) it does not render as expected

    def render(self, mode='human'):

        prefix = ''
        if not self._done:
            prefix = '(Current) '

        if 0 not in (len(ax) for ax in self.state.cluster) and self.best_match is not None:

            print('{}Found cluster'.format(prefix))
            self._render(self.state.cluster)
            print('')

            print('{}Best matched hidden cluster [from {} hidden clusters]'.format(prefix, len(self.state.hclusters)))
            self._render(self.state.hclusters[self.best_match])

        else:
            print('No cluster found yet..')

