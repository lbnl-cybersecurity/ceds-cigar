import re
import copy
import importlib
import warnings

from pycigar.utils.exeptions import FatalPyCIGARError

devcon_id_re = re.compile(r'[a-z_]+$')


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class PyCIGARSpec(object):
    """A specification for a particular instance of the device/controller. Used
    to register the parameters for official evaluations.
    Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the device/controller class (e.g. module.name:Class)
        kwargs (dict): The kwargs to pass to the environment class
    """

    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

        match = devcon_id_re.search(id)
        if not match:
            raise FatalPyCIGARError('Attempted to register malformed device/controller ID: {}. (Currently all IDs must be of the form {}.)'.format(id, devcon_id_re.pattern))
        self._devcon_name = match.group()

    def make(self, **kwargs):
        """Instantiates an instance of the device/controller with appropriate kwargs"""
        if self.entry_point is None:
            raise FatalPyCIGARError('Attempting to make deprecated device/controller {}. (HINT: is there a newer registered version of this device/controller?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            devcon = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            devcon = cls(**_kwargs)

        # Make the environment aware of which spec it came from.
        spec = copy.deepcopy(self)
        spec._kwargs = _kwargs

        return devcon

    def __repr__(self):
        return "PyCIGARSpec({})".format(self.id)


class PyCIGARRegistry(object):
    """Register an device/controller by ID. IDs remain stable over time and are
    guaranteed to resolve to the same device/controller dynamics (or be
    desupported). The goal is that results on a particular environment
    should always be comparable, and not depend on the version of the
    code that was running.
    """

    def __init__(self):
        self.devcon_specs = {}

    def make(self, path, **kwargs):
        spec = self.spec(path)
        devcon = spec.make(**kwargs)
        return devcon

    def all(self):
        return self.devcon_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            # catch ImportError for python2.7 compatibility
            except ImportError:
                raise FatalPyCIGARError('A module ({}) was specified for the device/controller but was not found'.format(mod_name))
        else:
            id = path

        match = devcon_id_re.search(id)
        if not match:
            raise FatalPyCIGARError('Attempted to look up malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id.encode('utf-8'), devcon_id_re.pattern))

        try:
            return self.devcon_specs[id]
        except KeyError:
            # Parse the env name and check to see if it matches the non-version
            # part of a valid env (could also check the exact number here)
            devcon_name = match.group(1)
            matching_devcons = [valid_devcon_name for valid_devcon_name, valid_devcon_spec in self.devcon_specs.items()
                             if devcon_name == valid_devcon_spec._devcon_name]
            if matching_devcons:
                raise FatalPyCIGARError('Device/controller {} not found (valid versions include {})'.format(id, matching_devcons))
            else:
                raise FatalPyCIGARError('No registered device/controller with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.devcon_specs:
            raise FatalPyCIGARError('Cannot re-register id: {}'.format(id))
        self.devcon_specs[id] = PyCIGARSpec(id, **kwargs)

# Have a global registry
pycigar_registry = PyCIGARRegistry()

def pycigar_register(id, **kwargs):
    return pycigar_registry.register(id, **kwargs)

def pycigar_make(id, **kwargs):
    return pycigar_registry.make(id, **kwargs)

def pycigar_spec(id):
    return pycigar_registry.spec(id)