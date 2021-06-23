
from .base import BaseEnv
from ..utils.states import State
from ..utils.helper import tensor_to_string, index_to_tensor


class TriclusterEnv(BaseEnv):

    def __init__(self):
        # parse args

        super(TriclusterEnv, self).__init__()

        self.state = State(generator='TriclusterGenerator')
        self.reset()

    def _render(self, index):
        return print(tensor_to_string(index_to_tensor(self.state.as_dense, index), index))
