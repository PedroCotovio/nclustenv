
def index_to_matrix(x, index):

    """Returns a sub-matrix of `x` given by the `index`."""

    return [[x[row][col] for col in index[1]] for row in index[0]]


def index_to_tensor(x, index):

    """Returns a sub-tensor of `x` given by the `index`."""

    return [index_to_matrix(x[ctx], index[:2]) for ctx in index[2]]


def matrix_to_string(matrix, index=None, title=''):

    """Returns a matrix as a printable string"""

    if index:
        temp = [[title] + ['y{}'.format(i) for i in index[1]]]
        for i, idx in enumerate(index[0]):

            idx = 'x{}'.format(idx)

            try:
                temp.append([idx] + matrix[i])
            except IndexError:
                temp.append(idx)

        matrix = temp

    return '\n'.join([''.join(['{:10}'.format(str(item)) for item in row]) for row in matrix])


def tensor_to_string(tensor, index=None):

    """Returns a tensor as a printable string"""

    title = ['' for _ in tensor]

    if index:
        title = ['z{}'.format(i) for i in index[2]]
        index = index[:2]

    return '\n\n'.join([matrix_to_string(ctx, index, title[i]) for i, ctx in enumerate(tensor)])


def loader(cls, module=None):

    """Loads a method from a pointer or a string"""

    return getattr(module, cls) if isinstance(cls, str) else cls


def real_to_ind(x, param):
    """Parses real values into list indexes"""

    return int(param * len(x))


def clusters_from_bool(graph, ntypes, hclusters=False):

    """Returns the clusters of a graph as a list of lists"""

    if hclusters:
        keys = [key for key in graph.nodes[ntypes[0]].data.keys() if isinstance(key, str)]
    else:
        keys = [key for key in graph.nodes[ntypes[0]].data.keys() if isinstance(key, int)]

    return [[[i
              for i, val in enumerate(graph.nodes[ntype].data[j]) if val]
             for ntype in ntypes]
            for j in keys]
