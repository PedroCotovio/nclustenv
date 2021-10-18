
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


def parse_ds_settings(settings, enforced=None):

    """Parse dataset settings into actionable dict"""

    if enforced is None:
        enforced = {
            'silence': True,
            'in_memory': True,
            'seed': None
        }

    new_settings = {
        'fixed': {},
        'discrete': {},
        'continuous': {},
    }

    keys = list(settings.keys())

    # Enforce fixed settings

    for key in list(enforced.keys()):
        if key in keys:
            keys.remove(key)

        new_settings['fixed'][key] = enforced[key]

    for key in keys:
        if settings[key]['randomize']:
            if settings[key]['type'] == 'categorical':
                new_settings['discrete'][key] = settings[key]['value']

            elif settings['key']['type'] == 'continuous':
                new_settings['continuous'][key] = settings[key]['value']

        else:
            new_settings['fixed'][key] = settings[key]['value']

    return new_settings



