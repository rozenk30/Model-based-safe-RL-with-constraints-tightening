from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    # def add(self, state, action, reward, new_state, done):
    #     #     experience = (state, action, reward, new_state, done)
    #     #     if self.num_experiences < self.buffer_size:
    #     #         self.buffer.append(experience)
    #     #         self.num_experiences += 1
    #     #     else:
    #     #         self.buffer.popleft()
    #     #         self.buffer.append(experience)
    def add(self, s1, s2, s3, s4, u1, u2, action1, action2, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss):
        experience = (s1, s2, s3, s4, u1, u2, action1, action2, z1_ss, z2_ss, z3_ss, z4_ss, u1_ss, u2_ss)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0