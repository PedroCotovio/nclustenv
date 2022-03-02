
basic_v1 = {
            'shape': [[10, 10], [10, 10]],
            'n': 1,
            'clusters': [1, 1],
            'dataset_settings': {
                'dstype': dict(value='Symbolic'),
                'patterns': dict(value=[['CONSTANT', 'CONSTANT']]),
                'symbols': dict(value=[-1, 1]),
                'bktype': dict(value='UNIFORM'),
                'clusterdistribution': dict(value=[['UNIFORM', 4, 6], ['UNIFORM', 2, 4]]),
                'contiguity': dict(value=None),
                'plaidcoherency': dict(value='NO_OVERLAPPING')
            },
            'max_steps': 1000,
}

basic_v2 = basic_v1.copy()
basic_v2['shape'] = [[8, 6], [8, 6]]

base = {
            'shape': [[100, 10], [100, 10]],
            'n': 5,
            'clusters': [5, 5],
            'dataset_settings': {
                'dstype': dict(value='Symbolic'),
                'patterns': dict(value=[['CONSTANT', 'CONSTANT']]),
                'symbols': dict(value=[-1, 1]),
                'bktype': dict(value='UNIFORM'),
                'clusterdistribution': dict(value=[['UNIFORM', 8, 12], ['UNIFORM', 4, 6]]),
                'contiguity': dict(value=None),
                'plaidcoherency': dict(value='NO_OVERLAPPING')
            },
            'max_steps': 1000,
        }
