import time
import datetime

import torch
import numpy as np


class Timer(object):
    """
    Simple object to keep track of time
    """
    def __init__(self):
        super(Timer, self).__init__()
        self.times = []
        self.start()

    def start(self):
        # reset time array and get current time
        self.times = []
        self.times.append(time.time())

    def tick(self):
        # append current time and get
        # elapsed time since last tick
        self.times.append(time.time())

    def aver(self, last_only=False):
        # calculate the average elapsed time
        # between consecutive ticks

        # if we only have one tick then return 0
        if len(self.times) <= 1:
            return 0.0

        # last_only means elapsed time for last 2 ticks
        if last_only:
            last_diff = self.times[-1] - self.times[-2]
            return str(datetime.timedelta(seconds=last_diff)).split('.')[0]

        diff = np.diff(self.times)
        average_time = np.mean(diff)

        return str(datetime.timedelta(seconds=average_time)).split('.')[0]


class MetricTracker(object):
    """
    This object keeps track of a certain metric
    through iterations, and provides easy access
    to metric summaries
    """
    def __init__(self):
        super(MetricTracker, self).__init__()
        # we have a 2 arrays, one for value
        # and a second for index, which we use
        # when we want to average
        self.values = []
        self.indices = []

    def reset(self):
        # reset arrays
        self.values = []
        self.indices = []

    def add(self, val, ind):
        # add new values to arrays
        self.values.append(val)
        self.indices.append(ind)

    def aver(self):
        # average is the sum of all values
        # divided by the sum of all indices
        return np.sum(self.values) / (np.sum(self.indices) + 1e-20)


def get_accuracy(predicted, gt):
    """
    Measure prediction by looking at the
    distance between the predicted point and
    GT, and check how many values are below
    a certain threshold

    :param predicted: Nx3 tensor of predicted points
    :param gt: Nx3 tensor of GT points

    :return: accuracy as a float between 0 and 1
    """
    threshold = 0.02
    distances = torch.sqrt(torch.sum((predicted - gt) ** 2, dim=1))
    corrects = torch.sum(distances <= threshold).double()
    return corrects / gt.size(0)
