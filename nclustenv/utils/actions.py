
class Action:
    """"
    Action class to store the action for the environment.
    """
    def __init__(self, id_, param, labels=['add', 'remove']):
        """"

        Parameters
        ----------

            id_: int
                The id of the selected action.

                ======== ====================
                    Default Actions
                -----------------------------
                index    action
                ======== ====================
                0        add
                1        remove

            param: float
                The parameter of an action.

                **Range**: [0, 1]

        """
        self.id = id_
        self.parameter = param
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

        return '_{}'.format(self._labels[self.id])

    @property
    def parameter(self):
        """"
        Returns the parameter related to the action selected.

        Returns
        -------

            float
                The parameter related to this action.

        """
        if len(self.parameter) == len(self._labels):
            res = self.parameter[self.id]
        else:
            res = self.parameter[0]

        return float(max(min(res, 1), 0))
