

def index_to_matrix(x, cluster):

    return [[x[row][col] for col in cluster[1]] for row in cluster[0]]


def index_to_tensor(x, cluster):

    return [index_to_matrix(x[ctx], cluster[:2]) for ctx in cluster[2]]


def matrix_to_string(matrix, index=None, title=''):

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

    title = ['' for _ in tensor]

    if index:
        title = ['z{}'.format(i) for i in index[2]]
        index = index[:2]

    return '\n\n'.join([matrix_to_string(ctx, index, title[i]) for i, ctx in enumerate(tensor)])


def loader(module, cls):
    return getattr(module, cls) if isinstance(cls, str) else cls
