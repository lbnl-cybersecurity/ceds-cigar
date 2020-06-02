from threading import Lock


class Singleton(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instance = None

    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class logger(metaclass=Singleton):
    """Logger for logging information during experiment.

    There is only one logger instance per environment. If the logger is not created, it will
    be created in the first call, otherwise it return the created instance.

    If you want to log any information during experiment, simply import the logger and call logger().
    For example:
    >>> Logger = logger()
    >>> Logger.log('s701a', 'voltage', 1.1)

    It will log the voltage value at bus s701a as 1.1. Notice that this function will append the value
    to the voltage of bus s701a.

    There are 2 types of information in logger:
        * log_dict: the default logging information (voltage, active power, reactive power,...)
        * custom_metrics: the metrics that user want to save in custom experiment.

    Parameters
    ----------
    metaclass : Singleton, optional
        inherit from Singleton to make sure there is only 1 instance, by default Singleton

    Returns
    -------
    logger
        the logging instance.
    """
    log_dict = {}
    custom_metrics = {}
    active = False

    def log(self, object, params, value):
        """Add value to log_dict.
        The method will append the new value with historical value at the same object and same params.

        Parameters
        ----------
        object : str
            Object ID in the simulation. For example, buses ID, inverters ID,...
        params : str
            The params that you want to track. For example, voltage, active power,...
        value : float
            The value of the params.
        """
        if self.active:
            if object not in self.log_dict:
                self.log_dict[object] = {}
            if params not in self.log_dict[object]:
                self.log_dict[object][params] = [value]
            else:
                self.log_dict[object][params].append(value)

    def log_single(self, object, params, value):
        """Add value to log_dict only once.
        The method will overwrite the historical value with new value at the same object and same params.

        Parameters
        ----------
         object : str
            Object ID in the simulation. For example, buses ID, inverters ID,...
        params : str
            The params that you want to track. For example, voltage, active power,...
        value : float
            The value of the params.
        """
        if self.active:
            if object not in self.log_dict:
                self.log_dict[object] = {}

            self.log_dict[object][params] = value

    def reset(self):
        """Reset the instance for new logging action.
        """
        self.log_dict = {}
        self.custom_metrics = {}

    def set_active(self, active=True):
        """Activate the logger.
        Any logging call when the logger is deactivated, is ignored.

        Parameters
        ----------
        active : bool, optional
            activate the logger, by default True.

        Returns
        -------
        bool
            return the state of logger.
        """
        self.active = active
        return self.active


def test_logger():
    Logger = logger()
    Logger.log('voltage', 'random', 1)
    print(Logger.log_dict)
    Logger.reset()
    Logger.log('voltage', 'random', 1)


if __name__ == "__main__":
    # process1 = Thread(target=test_logger, args=())
    # process2 = Thread(target=test_logger, args=())
    # process1.start()
    # process2.start()
    test_logger()
