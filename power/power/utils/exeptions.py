"""Power-specific exceptions."""


class FatalPowerError(Exception):
    """Exception class for Flow errors which do not allow for continuation."""

    def __init__(self, msg):
        Exception.__init__(self, msg)
