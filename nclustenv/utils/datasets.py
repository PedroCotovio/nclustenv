
from dgl.data import DGLDataset
import numpy as np

from nclustenv.utils.spaces import DGLHeteroGraphSpace
from nclustenv.utils.helper import parse_ds_settings
from nclustenv.utils.states import State

import os
import torch as th
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info


class SyntheticDataset(DGLDataset):

    def __init__(
            self,
            length=10,
            shape=None,
            n=None,
            clusters=None,
            dataset_settings=None,
            seed=None,
            generator='BiclusterGenerator',
            name='synthetic',
            save_dir=None,
            verbose=False
    ):

        if dataset_settings is None:
            dataset_settings = {}

        if clusters is None:
            clusters = [1, 1]

        self.graphs = None
        self.labels = None

        self._n = length

        np_random = np.random.RandomState(seed)

        self._observation_space = {
                'shape': shape,
                'n': n,
                'clusters': clusters,
                'settings': parse_ds_settings(dataset_settings),
                'np_random': np_random,
                'dtype': np.int32
        }

        self._state = {
            'generator': generator,
            'n': n,
            'np_random': np_random
        }

        super().__init__(
            name=name,
            raw_dir=save_dir,
            verbose=verbose
        )

    def process(self):

        _observation_space = DGLHeteroGraphSpace(**self._observation_space)
        _state = State(**self._state)

        self.graphs = []
        self.labels = []

        for _ in range(self._n):
            _state.reset(*_observation_space.sample(), not_init=True)
            self.graphs.append(_state.current)
            self.labels.append(_state.hclusters)


    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs)
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        save_info(info_path, {
            'labels': self.labels,
            'observation_space': self._observation_space,
            'state': self._state
        })

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        self.labels = load_info(info_path)['labels']
        self._observation_space = load_info(info_path)['observation_space']
        self._state = load_info(info_path)['state']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

    @property
    def defined(self):

        if self._observation_space['n']:
            return True
        return False

    @property
    def shape(self):
        return self._observation_space['shape']

    @property
    def clusters(self):
        return self._observation_space['clusters']

    @property
    def settings(self):
        return self._observation_space['settings']

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
