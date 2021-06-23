
from .base import BaseEnv
from ..utils.states import State
from ..utils.helper import matrix_to_string, index_to_matrix


class BiclusterEnv(BaseEnv):

    def __init__(self):

        # parse args

        super(BiclusterEnv, self).__init__()

        self.state = State()
        self.reset()

    def _render(self, index):
        return print(matrix_to_string(index_to_matrix(self.state.as_dense, index), index))