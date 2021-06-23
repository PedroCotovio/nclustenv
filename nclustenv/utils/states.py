
import nclustgen

# TODO implement add, remove, cluster, generate


class State:

    def __init__(self, generator='BiclusterGenerator'):

        self.generator = (getattr(nclustgen, generator) if isinstance(generator, str) else generator)

    @property
    def shape(self):

        """
        Returns the state's shape.

        Returns
        -------

            list
                Shape of current state.

        """
        return self.generator.X.shape

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

        return self.generator.Y

    @property
    def current(self):

        """
        Returns the current state.

        Returns
        -------

            dgl graph
                Current state.

        """

        return self.generator.graph

    @property
    def as_dense(self):

        """
        Returns the current state as a dense array.

        Returns
        -------

            numpy array
                Current state as a dense array.

        """

        return self.generator.X

    def add(self, param):
        pass

    def remove(self, param):
        pass

    def generate(self):
        pass



