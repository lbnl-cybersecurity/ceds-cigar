
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    """Summary
    This is the buffer object, which stores experience that our agents have experienced in the past
    Attributes:
        buffer (s,a,r,t,s2): get an experience 
        buffer_size (int): total size of the buffer
        count (int): the actual number of exp in the buffer
    """

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        """Summary
        this function is to add an experience into the buffer, 1 experience contains:
            state (s), action taking at this state (a), reward we receive (r), 
            it's the end of episode or not - terminal (t), the next state after taking the action (s2) 
        
        Args:
            s (state): state
            a (np.array): action
            r (float): reward 
            t (bool): terminal
            s2 (state): the next state
        """
        experience = (s, a, r, t, s2)
        # if the buffer is still not full, keep adding
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            #if the buffer is full, pop out the oldest experience and adding new experience
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """Summary
        
        Returns:
            int: actual number of exp in the buffer at the moment
        """
        return self.count

    def sample_batch(self, batch_size):
        """Summary
        
        Args:
            batch_size (int): number of exp we want to get
        
        Returns:
            a batch of experience: state, action, reward, terminal, next state
        """
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        """Summary
        clear all the exp in buffer
        """
        self.buffer.clear()
        self.count = 0