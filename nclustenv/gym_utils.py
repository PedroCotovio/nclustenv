import numpy as np


def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        # print(self.env_config)
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self, key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                            type(getattr(self, key))(value))
            else:
                setattr(self, key, value)


# Get Ray to work with gym registry
def create_env(config, *args, **kwargs):
    if type(config) == dict:
        env_name = config['env']
    else:
        env_name = config
    if env_name == 'BiclusterEnv-v0':
        from nclustenv.environments.classic_lr.biclusterenv import BiclusterEnv as env
    elif env_name == 'TriclusterEnv-v0':
        from nclustenv.environments.classic_lr.triclusterenv import TriclusterEnv as env
    elif env_name == 'OfflineBiclusterEnv-v0':
        from nclustenv.environments.classic_lr.biclusterenv import OfflineBiclusterEnv as env
    elif env_name == 'OfflineTriclusterEnv-v0':
        from nclustenv.environments.classic_lr.triclusterenv import OfflineTriclusterEnv as env
    return env