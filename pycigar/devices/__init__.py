from pycigar.devices.pv_inverter_device import PVDevice
from pycigar.devices.regulator_device import RegulatorDevice
from pycigar.devices.base_device import BaseDevice
from pycigar.utils.pycigar_registration import pycigar_register

__all__ = ["BaseDevice", "PVDevice", "RegulatorDevice"]

pycigar_register(
    id='pv_device',
    entry_point='pycigar.devices:PVDevice'
)

pycigar_register(
    id='regulator_device',
    entry_point='pycigar.devices:RegulatorDevice'
)
