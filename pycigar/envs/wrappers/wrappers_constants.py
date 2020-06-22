import numpy as np

# action space discretization
DISCRETIZE = 30
# the initial action for inverter
# INIT_ACTION = np.array([0.98, 1.01, 1.01, 1.04, 1.08])

# single head action
ACTION_CURVE = np.array([-0.03, 0., 0., 0.03, 0.07])
ACTION_LOWER_BOUND = 0.8
ACTION_UPPER_BOUND = 1.1

ACTION_MIN_SLOPE = 0.02  # actually the slope is stepper when value is small
ACTION_MAX_SLOPE = 0.07

# number of frames to keep
NUM_FRAMES = 6


ACTION_RANGE = 0.1
ACTION_STEP = 0.05
DISCRETIZE_RELATIVE = int((ACTION_RANGE / ACTION_STEP)) * 2 + 1

