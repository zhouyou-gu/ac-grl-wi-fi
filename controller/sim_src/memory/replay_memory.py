import random
from collections import deque
from threading import Lock, Semaphore

import numpy as np


class memory():
    def __init__(self, id = 0, size=100, n_sync_step = 10,asynchronization=False):
        self.id = id
        self.size = size
        self.n_sync_step = n_sync_step
        self.asynchronization = asynchronization
        self.data = deque(maxlen=self.size)

        self.counter = 0
        self.step_lock = Lock()
        self.sample_lock = Lock()
        self.sample_lock.acquire()

    def async_step(self, d, info=None):
        self.counter += 1
        return self.data.append(d)

    def async_sample(self, batch_size):
        if len(self.data) < batch_size:
            return None
        return random.sample(self.data, batch_size)

    def step(self, d, info = None):
        if not self.asynchronization:
            self.step_lock.acquire()

        self.async_step(d, info)

        if not self.asynchronization:
            self.sample_lock.release()

    def sample(self, batch_size = 20):
        if not self.asynchronization:
            self.sample_lock.acquire()

        self.async_sample(batch_size)

        if not self.asynchronization:
            self.step_lock.release()