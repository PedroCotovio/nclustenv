

def index_to_matrix(x, cluster):

    return [[x[row][col] for col in cluster[1]] for row in cluster[0]]


def index_to_tensor(x, cluster):

    return [index_to_matrix(x[ctx], cluster[:2]) for ctx in cluster[2]]


def matrix_to_string(matrix, index=None, title=''):

    if index:
        matrix = [[title] + index[0]] + [[index[1][i]] + vec for i, vec in enumerate(matrix.tolist())]

    return '\n'.join([''.join(['{:10}'.format(item) for item in row]) for row in matrix])


def tensor_to_string(tensor, index=None):

    title = ['' for _ in tensor]

    if index:
        title = ['z{}'.format(i) for i in index[2]]
        index = index[:2]

    return '\n\n'.join([matrix_to_string(ctx, index, title[i]) for i, ctx in enumerate(tensor)])
