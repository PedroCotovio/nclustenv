
from .base import BaseEnv
from ..utils.states import State
from ..utils.helper import matrix_to_string, index_to_matrix


class BiclusterEnv(BaseEnv):

    def __init__(
            self,
            shape=None,
            clusters=None,
            dataset_settings=None,
            seed=None,
            metric='match_score_1_n',
            max_steps=200,
            error_margin=0.05,
            penalty=0.001
    ):

        if shape is None:
            shape = [[100, 100], [200, 200]]

        super(BiclusterEnv, self).__init__(
            shape=shape,
            clusters=clusters,
            dataset_settings=dataset_settings,
            seed=seed,
            metric=metric,
            max_steps=max_steps,
            error_margin=error_margin,
            penalty=penalty
        )

        self.state = State(generator='BiclusterGenerator')
        self.reset()

    def _render(self, index):
        return print(matrix_to_string(index_to_matrix(self.state.as_dense, index), index))