VERSION='0.1.0'

ENV_LIST = [
    'BiclusterEnv-v0',
    'TriclusterEnv-v0',
    # 'OfflineBiclusterEnv-v0',
    # 'OfflineTriclusterEnv-v0'
]

TESTING_CONFIGS = [
    [
        {
            'shape': [[50, 10], [200, 50]],
            'n': 1,
            'clusters': [1, 3],
            'dataset_settings': {
                'patterns': {
                    'value': [
                        [['CONSTANT', 'CONSTANT'], ['Additive', 'Additive']],
                        [['Additive', 'Constant'], ['CONSTANT', 'CONSTANT']]
                    ],
                    'type': 'categorical',
                    'randomize': True
                },
                'realval': {
                    'value': [True, False],
                    'type': 'categorical',
                    'randomize': True
                },
                'maxval': {
                    'value': 11.0,
                },
                'minval': {
                    'value': [-10.0, 1.0],
                    'type': 'continuous',
                    'randomize': True
                }
            },
            'max_steps': 150
        },
        {
            'shape': [[100, 10], [200, 50]],
            'n': None,
            'clusters': [1, 3],
        },
        {
            'shape': [[100, 10], [200, 50]],
            'n': 5,
            'clusters': [1, 5],
        }
    ],
    [
        {
            'shape': [[50, 10, 2], [200, 50, 4]],
            'n': 1,
            'clusters': [1, 3],
            'dataset_settings': {
                'patterns': {
                    'value': [
                        [['Constant', 'Constant', 'Constant'], ['Additive', 'Additive', 'Additive']],
                        [['Constant', 'Additive', 'Additive'], ['Constant', 'Constant', 'Constant']]
                    ],
                    'type': 'categorical',
                    'randomize': True
                },
                'realval': {
                    'value': [True, False],
                    'type': 'categorical',
                    'randomize': True
                },
                'maxval': {
                    'value': 11.0,
                },
                'minval': {
                    'value': [-10.0, 1.0],
                    'type': 'continuous',
                    'randomize': True
                }
            },
            'max_steps': 150
        },
        {
            'shape': [[100, 10, 3], [200, 50, 5]],
            'n': None,
            'clusters': [1, 3],
        },
        {
            'shape': [[100, 10, 2], [200, 50, 3]],
            'n': 5,
            'clusters': [1, 5],
        }
    ],
]
