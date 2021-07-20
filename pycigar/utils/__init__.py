"""Contains all attack generators in PyCIGAR."""
from pycigar.utils.attack_generator import *
from pycigar.utils.pycigar_registration import pycigar_register
__all__ = [
    'AttackGenerator',
    'AttackGeneratorEvaluation',
    'HeterogeneousAttackGenerator',
    'HeterogeneousAttackGeneratorEvaluation',

    'DuplicateAttackGenerator'
]

pycigar_register(
    id='DuplicateAttackGenerator',
    entry_point='pycigar.utils:DuplicateAttackGenerator'
)

pycigar_register(
    id='AttackGenerator',
    entry_point='pycigar.utils:AttackGenerator'
)
pycigar_register(
    id='AttackGeneratorEvaluation',
    entry_point='pycigar.utils:AttackGeneratorEvaluation'

)
pycigar_register(
    id='HeterogeneousAttackGenerator',
    entry_point='pycigar.utils:HeterogeneousAttackGenerator'
)
pycigar_register(
    id='HeterogeneousAttackGeneratorEvaluation',
    entry_point='pycigar.utils:HeterogeneousAttackGeneratorEvaluation'
)