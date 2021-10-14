
import nclustgen
import numpy as np
from .helper import loader, real_to_ind
import torch as th


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
                  for i, val in enumerate(self.current.nodes[ntype].data[j]) if val]
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

    def _set_node(self, x, params):

        """
        Sets the cluster value for given node.

        Parameters
        ----------

        x: bool
            value to set
        params: list[float]
            List of parameters, [ntype, node, cluster], range: [0, 1]

        """

        if x is bool:
            # parse param(ntype) into string
            ntype = self._ntypes[real_to_ind(self._ntypes, params[0])]
            # parse param(node) into index
            index = real_to_ind(self.current.nodes(ntype), params[1])
            # parse param(cluster) into index
            cluster = real_to_ind(self.current.nodes[ntype].data, params[2])
            # set value on node data
            self.current.nodes[ntype].data[cluster][index] = x

    def _reset_clusters_index(self):
        """
        Resets the index of the cluster in the graph.
        """

        keys = self.current.nodes[self._ntypes[0]].data.keys()
        for i, key in enumerate(keys):
            self.current.ndata[i] = self.current.ndata.pop(key)

    def add(self, params):

        """
        Adds a given node to the cluster.

        Parameters
        ----------

        params: list[float]
            List of parameters, [ntype, node, cluster], range: [0, 1]

        """

        self._set_node(True, params[:3])

    def remove(self, params):

        """
        Removes a given node from the cluster.

        Parameters
        ----------

        params: list[float]
            List of parameters, [ntype, node, cluster], range: [0, 1]

        """

        self._set_node(False, params[:3])

    def merge(self, params):

        """
        Merges two clusters

        Parameters
        ----------

        params: list[float]
            List of parameters, [cluster1, cluster2], range: [0, 1]

        """

        params = params[:2]

        if not self.defined:

            # get index to set
            index = len(self.clusters)

            # parse params(clusters) into index
            cluster1, cluster2 = [real_to_ind(self.clusters, param) for param in params]

            # Set new cluster
            if cluster1 != cluster2:
                for ntype in self._ntypes:
                    self.current.nodes[ntype].data[index] = th.bitwise_or(
                        self.current.nodes[ntype].data[cluster1], self.current.nodes[ntype].data[cluster2]
                    )

            # Delete previous clusters
            self.current.ndata.pop(cluster1)
            self.current.ndata.pop(cluster2)

            # reset index
            self._reset_clusters_index()

    def split(self, params):

        """
        Splits two clusters

        Parameters
        ----------

        params: list[float]
            List of parameters, [cluster], range: [0, 1]

        """

        params = params[:1]

        if not self.defined:

            # get indexes to set
            index1 = len(self.clusters)
            index2 = index1 + 1

            # parse param(cluster) into index
            cluster = real_to_ind(self.clusters, params[0])

            for ntype in self._ntypes:

                # select partition point
                index = self._np_random.randint(
                    low=0, high=len(self.current.nodes[ntype].data[cluster]), dtype=np.int32
                )

                # create new clusters
                self.current.nodes[ntype].data[index1] = self.current.nodes[ntype].data[cluster][:index]
                self.current.nodes[ntype].data[index2] = self.current.nodes[ntype].data[cluster][index:]

                # delete previous cluster
                self.current.ndata.pop(cluster)

                # reset index
                self._reset_clusters_index()

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

        if self.defined:
            self._generator.to_graph(framework='dgl', device='gpu', nclusters=self.n)
        else:
            self._generator.to_graph(framework='dgl', device='gpu')

        # update ntype
        self._ntypes = [ntypes for ntypes in self.current.ntypes]
        self._ntypes.insert(0, self._ntypes.pop())

        # update cluster coverage
        self.cluster_coverage = self._set_cluster_coverage()

        return self.current




