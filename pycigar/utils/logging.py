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
    log_dict = {}
    custom_metrics = {}
    active = False

    def __init__(self):
        pass

    def log(self, object, params, value):
        if self.active:
            if object not in self.log_dict:
                self.log_dict[object] = {}
            if params not in self.log_dict[object]:
                self.log_dict[object][params] = [value]
            else:
                self.log_dict[object][params].append(value)

    def log_single(self, object, params, value):
        if self.active:
            if object not in self.log_dict:
                self.log_dict[object] = {}

            self.log_dict[object][params] = value
    
    def reset(self):
        self.log_dict = {}
        self.custom_metrics = {}

    def set_active(self, active=True):
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
