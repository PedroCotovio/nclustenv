
from .base import BaseEnv
from ..utils.states import State


class TriclusterEnv(BaseEnv):

    def __init__(self):
        # parse args

        super(TriclusterEnv, self).__init__()

        self.state = State(generator='TriclusterGenerator')
        self.reset()

    def render(self, mode='human'):
        pass