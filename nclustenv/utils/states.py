
import nclustgen

# TODO implement add, remove, cluster


class State:

    def __init__(self, generator='BiclusterGenerator'):

        self._cls = (getattr(nclustgen, generator) if isinstance(generator, str) else generator)
        self._generator = None

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
        Returns the current found clusters index (Desired goal).

        Returns
        -------

            list
                Found clusters.

        """
        pass

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

    def add(self, param):
        pass

    def remove(self, param):
        pass

    def reset(self, shape, nclusters, settings=None):

        if settings is None:
            settings = {}

        self._generator = self._cls(**settings)

        self._generator.generate(*shape, nclusters=nclusters)



