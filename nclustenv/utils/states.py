
import nclustgen
from .helper import loader


class State:

    """
    State class to store current environment state.
    """

    def __init__(self, generator='BiclusterGenerator'):

        """
        Parameters
        ----------

        generator: str or class.
            The name of a generator from the nclustgen tool, or the class for a personalised generator (not advised).

        Attributes
        ----------

        _cls: class
            The generator class for the state.

        _generator: object
            The generator object for the state.

        _ntypes: list[str]
            An ordered list of the axis labels.

        """

        self._cls = loader(nclustgen, generator)
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
    def coverage(self):

        """
        Returns the current state hidden cluster coverage.

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
            self.current.nodes[ntype].data['c'][index] = x


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




