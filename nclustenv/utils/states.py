
import nclustgen


class State:

    def __init__(self, generator='BiclusterGenerator'):

        self.generator = (getattr(nclustgen, generator) if isinstance(generator, str) else generator)

    @property
    def shape(self):

        """
        Returns the shape of current state.

        Returns
        -------

            list
                Shape of current state.
        """
        return self.generator.X.shape

    @property
    def cluster_index(self):
        pass

    @property
    def cluster(self):
        pass

    @property
    def current(self):
        return self.generator.graph

    @property
    def hclusters(self):
        return self.generator.Y

    def add(self, param):
        pass

    def remove(self, param):
        pass

    def generate(self):
        """

        """
        pass



