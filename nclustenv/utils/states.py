
import nclustgen


class State:

    def __init__(self, generator='BiclusterGenerator'):

        self._cls = (getattr(nclustgen, generator) if isinstance(generator, str) else generator)
        self._generator = None
        self._ntypes = None

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
    def cluster(self):

        """
        Returns the current found cluster's index (Desired goal).

        Returns
        -------

            list
                Found cluster.

        """

        return [[i for i, val in enumerate(self.current.nodes[ntype].data['c']) if val == 1]
                for ntype in self._ntypes]

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

    def _set_node(self, x, ntype, param):

        # parese ntype index to string
        ntype = self._ntypes[ntype]
        # parse param into node index
        index = int(param * len(self.current.nodes(ntype)))
        # set value on node data
        self.current.nodes[ntype].data['c'][index] = x

    def add(self, ntype, param):
        self._set_node(1, ntype, param)

    def remove(self, ntype, param):
        self._set_node(0, ntype, param)

    def reset(self, shape, nclusters, settings=None):

        if settings is None:
            settings = {}

        # generate
        self._generator = self._cls(**settings)
        self._generator.generate(*shape, nclusters=nclusters)
        self._generator.to_graph(framework='dgl', device='gpu')

        # update ntype
        self._ntypes = [ntypes for ntypes in self.current.ntypes]
        self._ntypes.insert(0, self._ntypes.pop())




