#!usr/bin/env python

'''
Tests to ensure environments load and basic functionality
is satisfied.
'''


import unittest
import nclustenv
from nclustenv.version import ENV_LIST
import traceback


class TestCaseBase(unittest.TestCase):

    @staticmethod
    def _build_env(env_name):
        env = nclustenv.make(env_name)
        return env


class TestEnv(TestCaseBase):

    def setUp(self):
        self.scenarios = ENV_LIST

    def test_make(self):
        # Ensures that environments are instantiated

        for env_name in self.scenarios:

            tb = None

            try:
                _ = self._build_env(env_name)
                success = True
            except Exception as e:
                tb = e.__traceback__
                success = False

            self.assertTrue(success, ''.join(traceback.format_tb(tb)))

    def test_episode(self):
        # Run 100 episodes and check observation space

        for env_name in self.scenarios:

            EPISODES = 100
            env = self._build_env(env_name)
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
