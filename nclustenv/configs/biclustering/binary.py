
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
            'max_steps': 500,
        }
