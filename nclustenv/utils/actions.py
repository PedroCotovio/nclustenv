
class Action:
    """"
    Action class to store the action for the environment.
    """
    def __init__(self, index, ntype, param, labels=None):
        """"

        Parameters
        ----------

            index: int
                The index of the selected action.

                ======== ====================
                    Default Actions
                -----------------------------
                index    action
                ======== ====================
                0        add
                1        remove

            param: list[float]
                The parameter of an action.

                **Range**: [0, 1]

        """
        if labels is None:
            labels = ['add', 'remove']

        self.index = index
        self.ntype = ntype

        self._parameter = param
        self._labels = labels

    @property
    def action(self):

        """
        Returns the action selected.

        Returns
        -------

            str
                The selected action.

        """

        return self._labels[self.index]

    # TODO Confirm type of continuous param output
    @property
    def parameter(self):
        """"
        Returns the parameter related to the action selected.

        Returns
        -------

            float
                The parameter related to this action.

        """
        try:
            if len(self._parameter) > 1:
                res = self._parameter[self.index][self.ntype]
            else:
                res = self._parameter[0]
        except TypeError:
            res = self._parameter

        return float(max(min(res, 1.0), 0.0))
