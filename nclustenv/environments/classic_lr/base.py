import abc
from abc import ABC
from statistics import mean
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.optimize import linear_sum_assignment

from nclustenv.utils import actions, metrics
from nclustenv.utils.helper import loader


class BaseEnv(gym.Env, ABC):

    """
    Abstract class from where dimensional specific subclasses should inherit. Should not be called directly.
    This class abstracts dimensionality providing core implemented methods and abstract methods that should be
    implemented for any n-clustering environment.
    """

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            shape,
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

        """
        Parameters
        ----------

        shape: list, default [[100, 100, 2], [200, 200, 5]]
            List of length 2 where the first element is the minimum shape the observation space and the second is the
            maximum.
        clusters: [int], default [1, 1]
            List of length 2 where the first element is the minimum number of cluster to be hidden in the environment
            and the second is the maximum.
        dataset_settings: dict, default {}
            Dataset settings to be passed to generator.

            Note
            ----
                Parameters `silence` and `in_memory` should not be set, and will be overwritten.

        seed: int, default None
            Seed to initialize random object.
        metric: str or class, default 'match_score_1_n'.
            The name of a metric implemented in `utils.metrics`, or a pointer for a personalised metric.

            ================ ===========
            Implemented metrics
            ----------------------------
            name             task
            ================ ===========
            match_score      Any
            ================ ===========

        action: str or class, default 'Action'.
            The name of a action class implemented in `utils.actions`, or a pointer for a personalised action class.
        max_steps: int, default 200
            Maximum number of actions an agent can perform in a given environment.
        error_margin: float, default 0.05
            Margin of error for agent.
        penalty: float, default 0.001
            Penalty on reward per timestep (discount factor).

        Attributes
        ----------

        dataset_settings: dict
            Dataset settings to be passed to generator.
        max_steps: int
            Maximum number of actions an agent can perform in a given environment.
        target: float
            Target margin for agent.
        penalty: float
            Penalty on reward per timestep (discount factor).
        action_space: gym space
            Space from where the agent samples actions.
        observation_space: gym space
            Space from where the environment samples observations.
        np_random: numpy random object
            Random object.
        state: state object
            State object.

        """

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
        self._metric = loader(metric, metrics)

        # action pointer
        self._action = loader(action, actions)

        self.max_steps = max_steps
        self.target = error_margin
        self.penalty = penalty

        self.action_space = spaces.Tuple((spaces.Discrete(4),
                                          spaces.Box(low=0.0, high=1.0, shape=[4, 3], dtype=np.float16)))

        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(0, 1, shape=(self.N,)),
            "avail_actions": spaces.Box(0, 1, shape=(self.N,)),
            "state": spaces.Box(
                low=np.array(shape[0]),
                high=np.array(shape[1]),
                dtype=np.int32)
        })

        # Init

        self._last_distances = None
        self._current_step = None
        self._steps_beyond_done = None
        self._done = None

        self.np_random = None
        self.state = None

        self.seed(seed)

    def seed(self, seed=None):

        """
        Sets the seed for this env's random number generator(s).

        Note
        ----
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns
        -------

            [int]
                Returns the list of seeds used in this env's random
                number generators. The first value in the list should be the
                "main" seed, or the value which a reproducer should pass to
                'seed'. Often, the main seed equals the provided 'seed', but
                this won't be true if seed=None, for example.
        """

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
            getattr(self.state, action_.action)(action_.parameters)

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

        return self.state.state, reward, self._done, {}

    def get_reward(self, last_distances, goal=False, error=False):

        """
        Returns the reward for the current state.

        Returns
        -------

            float
                Current reward.

        """

        return float(
            ((last_distances[-2] - last_distances[-1])
             - self.penalty)
            + ((2 if goal else 0) - (1 if error else 0))
        )

    @property
    def volume_match(self):

        """
        Returns the volume match for the current state.

        Returns
        -------

            float
                Current volume match.

        """

        cost_matrix = self._metric(self.state.clusters, self.state.hclusters)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return (cost_matrix[row_ind, col_ind] * self.state.cluster_coverage[row_ind]).sum()

    @property
    def best_match(self):

        """
        Returns index of the hidden clusters with the best match for the current found clusters.

        Returns
        -------

            int
                Current best match.

        """

        return linear_sum_assignment(self._metric(self.state.clusters, self.state.hclusters))

    def reset(self):

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

        return self.state.reset(shape=shape, nclusters=nclusters, settings=self.dataset_settings)

    @abc.abstractmethod
    def _render(self, index):

        """
        Renders a given cluster.

        Parameters
        ----------

        index: [[int]]
            indexes of cluster on current environment.
        """
        pass

    def render(self, mode='human'):

        """
        Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note
        ----
            Make sure that your class's metadata 'render.modes' key includes
            the list of supported modes. It's recommended to call super()
            in implementations to use the functionality of this method.

        Parameters
        ----------

        mode: str, default 'human'
            The mode to render with.
        """

        prefix = ''
        if not self._done:
            prefix = '(Current) '

        i = 1

        for cluster in self.state.clusters:

            hcluster = self.state.hclusters[self.best_match]

            if 1 in (1 for ax in cluster if len(ax) > 0) and hcluster is not None:

                print('{}Found cluster {}'.format(prefix, i))
                self._render(cluster)
                print('')

                print('{}Best matched hidden cluster'.format(prefix))
                self._render(self.state.hclusters[self.best_match])

                i += 1

        if i == 1:
            print('No cluster found yet..')