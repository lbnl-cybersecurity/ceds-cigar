import numpy as np

room = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])

room_reward = np.array([
    [ 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, 10],
])

action = {
    "move_up":    np.array([0, -1]),
    "move_down":  np.array([0,  1]),
    "move_left":  np.array([-1, 0]),
    "move_right": np.array([0,  1]),
}

start_position = np.array([0, 0])
goal_position =  np.array([4, 4])

def is_legal_move(position, move):
    new_position = position + move
    if sum(new_position < 0) == 0:
        return True
    else:
        return False
