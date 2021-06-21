

class BaseState:

    def __init__(self, ):

        # Init
        self.generator = None



    @property
    def shape(self):

        """
        Returns the shape of current state.

        Returns
        -------

            list
                Shape of current state.
        """
        pass

    @property
    def cluster_index(self):
        pass

    @property
    def cluster(self):
        pass

    @property
    def state(self):
        return self.generator.graph

    @property
    def hclusters(self):
        return self.generator.Y

    def add(self, param):

        self.steps += 1

    def remove(self, param):

        self.steps += 1



