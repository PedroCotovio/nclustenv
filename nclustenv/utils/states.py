
import nclustgen
import numpy as np
from .helper import loader


class State:

    """
    State class to store current environment state.
    """

    def __init__(self, generator='BiclusterGenerator', n=None, np_random=None):

        """
        Parameters
        ----------

        generator: str or class, default BiclusterGenerator.
            The name of a generator from the nclustgen tool, or the class for a personalised generator (not advised).

        n: int, default None
            The number of clusters to find.

        np_random: pointer, default None
            Random object. If undefined np.random will be used

        Attributes
        ----------

        n: int
            The number of clusters to find.
        defined: object
            If the number of clusters to find is known
        cluster_coverage: list[float]
            An ordered list of with the percentage of coverage for every hidden cluster.

        """

        if np_random is None:
            np_random = np.random

        self._cls = loader(generator, nclustgen)
        self.n = n
        self.defined = n is not None

        self._generator = None
        self._ntypes = None
        self._np_random = np_random

        self.cluster_coverage = None

    @property
    def shape(self):

        """
        Returns the state's shape.

        Returns
        -------

            list
                Shape of current state.

        """
        return self._generator.X.shape

    @property
    def clusters(self):

        """
        Returns the current found clusters indexes (Current solution).

        Returns
        -------

            list
                Found clusters.

        """

        return [[[i
                  for i, val in enumerate(self.current.nodes[ntype].data[j]) if val == 1]
                 for ntype in self._ntypes]
                for j in range(len(self.current.nodes[self._ntypes[0]].data))]

    @property
    def hclusters(self):
        """
        Returns hidden clusters index (Goal).

        Returns
        -------

            list
                Hidden clusters.

        """

        return self._generator.Y

    @property
    def hclusters_size(self):
        """
        Returns the size of the hidden clusters.

        Returns
        -------

            list[int]
                Ordered list hidden cluster sizes.

        """

        return [sum(map(len, cluster)) for cluster in self._generator.Y]

    @property
    def max_hclusters_size(self):

        """
        Returns the current state clusters max possible size.

        Returns
        -------

            float
                Percentage of cluster coverage.

        """

        if not self.defined:

            return sum(self.hclusters_size)

        else:
            sizes = self.hclusters_size

            # Sliding window algorithm
            curr_sum = sum(sizes[:self.n])
            res = curr_sum
            for i in range(self.n, len(sizes)):
                curr_sum += sizes[i] - sizes[i - self.n]
                res = max(res, curr_sum)

            return res

    @property
    def coverage(self):

        """
        Returns the current state hidden cluster total coverage.

        Returns
        -------

            float
                Percentage of cluster coverage.

        """

        return self._generator.coverage

    @property
    def current(self):

        """
        Returns the current state.

        Returns
        -------

            dgl graph
                Current state.

        """

        return self._generator.graph

    @property
    def as_dense(self):

        """
        Returns the current state as a dense array.

        Returns
        -------

            numpy array
                Current state as a dense array.

        """

        return self._generator.X

    def _set_cluster_coverage(self):
        """
        Returns a list of hidden clusters coverage.

        Returns
        -------

            list[float]
                Ordered list hidden cluster coverage.

        """

        max_size = self.max_hclusters_size
        sizes = self.hclusters_size

        return [cluster/max_size for cluster in sizes]

    def _set_node(self, x, ntype, param):

        """
        Sets the cluster value for given node.

        Parameters
        ----------

        x: int
            value to set [0, 1]
        ntype: int
            Index of node type.
        param: float
            Node to set [0, 1]

        """

        if x in [0, 1]:
            # parse ntype index to string
            ntype = self._ntypes[ntype]
            # parse param into node index
            index = int(param * len(self.current.nodes(ntype)))
            # set value on node data
            self.current.nodes[ntype].data[0][index] = x

    def add(self, ntype, param):

        """
        Adds a given node to the cluster.

        Parameters
        ----------

        ntype: int
            Index of node type.
        param: float
            Node to set [0, 1]

        """

        self._set_node(1, ntype, param)

    def remove(self, ntype, param):

        """
        Removes a given node from the cluster.

        Parameters
        ----------

        ntype: int
            Index of node type.
        param: float
            Node to set [0, 1]

        """

        self._set_node(0, ntype, param)

    def reset(self, shape, nclusters, settings=None):

        """
        Resets the state (generates new state)

        Parameters
        ----------

        shape: list[int]
            Shape of new state.
        nclusters: int
            Number of hidden clusters.
        settings: dict
            Dataset settings (nclustgen).

        Returns
        -------

            dgl graph
                Current state.

        """
        if settings is None:
            settings = {}

        # generate
        self._generator = self._cls(**settings)
        self._generator.generate(*shape, nclusters=nclusters)
        self._generator.to_graph(framework='dgl', device='gpu')

        # update ntype
        self._ntypes = [ntypes for ntypes in self.current.ntypes]
        self._ntypes.insert(0, self._ntypes.pop())

        # update cluster coverage
        self.cluster_coverage = self._set_cluster_coverage()

        return self.current




