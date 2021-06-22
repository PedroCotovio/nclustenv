from abc import ABC
from statistics import mean
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

from ..utils.actions import Action
from ..utils import metrics

# TODO implement render(bic/tric), parse params, reset


class BaseEnv(gym.Env, ABC):

    """
    Bi/Tricluster environment
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, smax, smin, clusters, noise, missings, seed=None,
                 metric='match_score_1_n', max_steps=200, error_margin=0.05, penalty=0.001):

        super(BaseEnv, self).__init__()

        # Environment attributes

        if isinstance(metric, str):
            self.metric = getattr(metrics, metric)
        else:
            self.metric = metric

        self.max_steps = max_steps
        self.target = 1.0 - error_margin
        self.penalty = penalty

        self.action_space = spaces.Tuple((spaces.Discrete(2),
                                          spaces.Box(low=0.0, high=1.0, shape=[1, ], dtype=np.float16)))
        self.observation_space = spaces.Box(low=np.array(smin), high=np.array(smax), dtype=np.int32)

        # Init

        self.np_random = None
        self.state = None
        self.last_distances = None
        self.current_step = None
        self.steps_beyond_done = None
        self.done = None

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

        if not self.done:
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
                self.done = True
            elif mean(self.last_distances) >= self.target:
                reward = self.get_reward(self.last_distances, True, True)
                self.done = True
            elif self.current_step > self.max_steps:
                reward = -1.0
                self.done = True
            else:
                reward = self.get_reward(self.last_distances)

        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )

            self.steps_beyond_done += 1
            reward = 0.0

        return self.state.current, reward, self.done, {}

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

        return float(self.metric(self.state.shape, self.state.cluster, self.state.hclusters))

    def reset(self):

        self.current_step = 0
        self.steps_beyond_done = 0
        self.last_distances = [0.0, 0.0, 0.0]
        self.done = False

    def _render(self):

        pass
