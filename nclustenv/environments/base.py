from statistics import mean
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from nclustenv.states.base import BaseState
from nclustenv.utils.actions import Action
from nclustenv.utils import metrics


class NClusterEnv(gym.Env):

    """
    Bi/Tricluster environment
    """

    metadata = {'render.modes': ['human']}

    # TODO implement init
    def __init__(self, smax, smin, clusters, noise, missings, seed=None,
                 metric='match_score_1_n', max_steps=200, error_margin=0.05, penalty=0.001):

        super(NClusterEnv, self).__init__()

        # Init

        self.np_random = None
        self.state = None
        self.last_distances = None
        self.current_step = None

        self.seed(seed)

        self.action_space = spaces.Tuple((spaces.Discrete(2),
                                          spaces.Box(low=0.0, high=1.0, shape=[1, ], dtype=np.float16)))
        self.observation_space = spaces.Box(low=np.array(smin), high=np.array(smax), dtype=np.int32)

        # Environment attributes

        if isinstance(metric, str):
            self.metric = getattr(metrics, metric)
        else:
            self.metric = metric

        self.max_steps = max_steps
        self.target = 1.0 - error_margin
        self.penalty = penalty

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.current_step += 1
        action_ = Action(*action)

        # Take action
        getattr(self.state, action_.action)(action_.parameter)

        # calculate
        self.last_distances.pop(0)
        self.last_distances.append(self.volume_match)

        # check state

        if self.last_distances[-1] == 1.0:
            reward = self.get_reward(self.last_distances, True)
            done = True
        elif mean(self.last_distances) >= self.target:
            reward = self.get_reward(self.last_distances, True, True)
            done = True
        elif self.current_step > self.max_step:
            reward = -1
            done = True
        else:
            reward = self.get_reward(self.last_distances)
            done = False

        return self.state.state, reward, done, {}

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

        return float(self.metric(self.state.shape, self.state.cluster_index, self.state.hclusters))

    # TODO implement reset
    def reset(self):

        self.current_step = 0
        self.last_distances = [0.0, 0.0, 0.0]

    # TODO implement render
    def render(self, mode='human'):

        pass
