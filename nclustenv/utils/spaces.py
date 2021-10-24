
import gym
from dgl import DGLHeteroGraph
from .helper import retrive_skey
import numpy as np


class DGLHeteroGraphSpace(gym.spaces.Box):

    def __init__(
            self,
            shape,
            n=None,
            clusters=None,
            settings=None,
            np_random=None,
            *args, **kwargs

    ):
        if np_random is None:
            np_random = np.random.RandomState()

        if clusters is None:
            clusters = [1, 1]

        if settings is None:
            settings = {}

        self.n = n
        self.clusters = clusters
        self.settings = settings
        self._np_random = np_random

        super(DGLHeteroGraphSpace, self).__init__(
            low=np.array(shape[0]),
            high=np.array(shape[1]),
            *args, **kwargs
        )

    def _sample(self, low, high, discrete=True) -> int:

        """
        Returns a random sample from a defined space.

        Returns
        -------

            int or float
                Space random sample.

        """

        try:
            if discrete:
                return self.np_random.randint(low=low, high=high)
            else:
                return self.np_random.random(low=low, high=high)
        except ValueError:
            return low

    def _node_labels(self, labels):

        res = [ntypes for ntypes in labels]
        res.insert(0, res.pop())

        return res

    def sample(self):

        """
        Returns a randomized sample of the dataset space.

        Returns
        -------

            list
                Dataset shape.

            int
                Number of clusters.

            dict
                Dataset advanced settings.

        """

        # Sample basic settings
        shape = super(DGLHeteroGraphSpace, self).sample()
        nclusters = self._sample(*self.clusters)

        # Get Settings
        settings = {'seed': self.np_random.randint(low=1, high=10 ** 9, dtype=np.int32)}

        ## Fixed
        for key, value in self.settings['fixed'].items():
            settings[key] = value

        ## Discrete
        for key, value in self.settings['discrete'].items():
            settings[key] = value[self._sample(0, len(value))]

        ## Continuous
        for key, value in self.settings['continuous'].items():
            settings[key] = self._sample(low=value[0], high=value[1], discrete=False)

        return shape, nclusters, settings

    def contains(self, x: DGLHeteroGraph) -> bool:

        # Retrive shape
        shape = np.array([x.nodes(ntype).shape[0] for ntype in self._node_labels(x.ntypes)])

        # Check inicialization
        if self.n:
            init = self.n
        else:
            init = 1

        # Verify settings

        if retrive_skey('dstype', self.settings, 'NUMERIC') == 'NUMERIC':
            values = retrive_skey('minval', self.settings, -10.0) <= min(x.edata['w']).item() \
                     and retrive_skey('maxval', self.settings, 10.0) <= max(x.edata['w']).item()

            realval = retrive_skey('realval', self.settings, True)
            dtype = x.edata['w'].isreal().all().item()

            settings = realval == dtype and values

        else:
            symbols = retrive_skey('symbols', self.settings)

            if symbols is None:
                symbols = [i for i in range(retrive_skey('nsymbols', self.settings, 10))]

            settings = x.edata['w'].apply_(lambda y: y in symbols).bool().all().item()

        return (
            isinstance(x, DGLHeteroGraph)
            and super(DGLHeteroGraphSpace, self).contains(shape)
            and len(x.nodes[self._node_labels(x.ntypes)[0]]) == init
            and settings
        )



