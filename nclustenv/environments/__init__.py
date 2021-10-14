
from gym.envs.registration import register

# Classic Environments
register(id='BiclusterEnv-v0',
    entry_point='nclustenv.environments.classic_lr.biclusterenv:BiclusterEnv'
)

register(id='TriclusterEnv-v0',
    entry_point='nclustenv.environments.classic_lr.triclusterenv:TriclusterEnv'
)

# Curriculum Environment
register(id='CurriculumEnv-v0',
    entry_point='nclustenv.environments.curriculum_lr.curriculumenv:CurriculumEnv'
)