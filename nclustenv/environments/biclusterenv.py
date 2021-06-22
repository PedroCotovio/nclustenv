
from .base import BaseEnv
from ..utils.states import State


class BiclusterEnv(BaseEnv):

    def __init__(self):

        # parse args

        super(BiclusterEnv, self).__init__()

        self.state = State()
        self.reset()

    def render(self, mode='human'):
        pass
