#!usr/bin/env python

'''
Tests to ensure environments load and basic functionality
is satisfied.
'''


import unittest
import nclustenv
from nclustenv.version import ENV_LIST, TESTING_CONFIGS
import traceback


class TestCaseBase(unittest.TestCase):

    @staticmethod
    def _build_env(env_name, **kwargs):
        env = nclustenv.make(env_name, **kwargs)
        return env


class TestEnv(TestCaseBase):

    def setUp(self):
        self.scenarios = zip(ENV_LIST, TESTING_CONFIGS)

    def test_make(self):
        # Ensures that environments are instantiated

        for env_name, configs in self.scenarios:
            for config in configs:

                tb = None

                try:
                    _ = self._build_env(env_name, **config)
                    success = True
                except Exception as e:
                    tb = e.__traceback__
                    success = False

                self.assertTrue(success, ''.join(traceback.format_tb(tb)))

    def test_episode(self):
        # Run 50 episodes and check observation space

        for env_name, configs in self.scenarios:
            for config in configs:

                EPISODES = 5
                env = self._build_env(env_name, **config)
                for ep in range(EPISODES):
                    state = env.reset()
                    while True:
                        self.assertTrue(env.observation_space.contains(state),
                                        f"State out of range of observation space: {state}")

                        action = env.action_space.sample()
                        state, reward, done, info = env.step(action)
                        if done:
                            break

                    self.assertTrue(done)
