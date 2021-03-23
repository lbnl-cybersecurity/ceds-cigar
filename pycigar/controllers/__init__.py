from pycigar.controllers.rl_controller import RLController
from pycigar.controllers.base_controller import BaseController
from pycigar.controllers.fixed_controller import FixedController
from pycigar.controllers.mimic_controller import MimicController
from pycigar.controllers.adaptive_inverter_controller import AdaptiveInverterController
from pycigar.controllers.adaptive_fixed_controller import AdaptiveFixedController
from pycigar.controllers.unbalanced_fixed_controller import UnbalancedFixedController
from pycigar.controllers.adaptive_unbalanced_fixed_controller import AdaptiveUnbalancedFixedController

from pycigar.controllers.battery_peak_shaving_controller_cent import BatteryPeakShavingControllerCent
from pycigar.controllers.battery_peak_shaving_controller_dist import BatteryPeakShavingControllerDist

from pycigar.utils.pycigar_registration import pycigar_register, pycigar_make, pycigar_spec

__all__ = [
    "RLController",
    "BaseController",
    "FixedController",
    "MimicController",
    "AdaptiveFixedController",
    "AdaptiveInverterController",
    "UnbalancedFixedController",
    "BatteryPeakShavingControllerCent",
    "BatteryPeakShavingControllerDist"
]

pycigar_register(
    id='rl_controller',
    entry_point='pycigar.controllers:RLController'
)

pycigar_register(
    id='adaptive_fixed_controller',
    entry_point='pycigar.controllers:AdaptiveFixedController'
)

pycigar_register(
    id='adaptive_inverter_controller',
    entry_point='pycigar.controllers:AdaptiveInverterController'
)

pycigar_register(
    id='unbalanced_fixed_controller',
    entry_point='pycigar.controllers:UnbalancedFixedController'
)

pycigar_register(
    id='adaptive_unbalanced_fixed_controller',
    entry_point='pycigar.controllers:AdaptiveUnbalancedFixedController'
)

pycigar_register(
    id='fixed_controller',
    entry_point='pycigar.controllers:FixedController'
)

pycigar_register(
    id='battery_peak_shaving_controller_cent',
    entry_point='pycigar.controllers:BatteryPeakShavingControllerCent'
)

pycigar_register(
    id='battery_peak_shaving_controller_dist',
    entry_point='pycigar.controllers:BatteryPeakShavingControllerDist'
)