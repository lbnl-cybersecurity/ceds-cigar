from pycigar.utils.pycigar_registration import pycigar_register

from pycigar.devices.base_device import BaseDevice
from pycigar.devices.pv_inverter_device import PVDevice
from pycigar.devices.regulator_device import RegulatorDevice
from pycigar.devices.battery_storage_device import BatteryStorageDevice
from pycigar.devices.battery_storage_device_VVC import BatteryStorageDeviceVVC
__all__ = ["BaseDevice", "PVDevice", "RegulatorDevice", "BatteryStorageDevice", "BatteryStorageDeviceVVC"]

pycigar_register(
    id='pv_device',
    entry_point='pycigar.devices:PVDevice'
)

pycigar_register(
    id='regulator_device',
    entry_point='pycigar.devices:RegulatorDevice'
)

pycigar_register(
    id='battery_storage_device',
    entry_point='pycigar.devices:BatteryStorageDevice'
)

pycigar_register(
    id='battery_storage_device_vvc',
    entry_point='pycigar.devices:BatteryStorageDeviceVVC'
)
