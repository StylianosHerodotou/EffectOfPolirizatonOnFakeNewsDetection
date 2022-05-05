from abc import ABC, abstractmethod


class AbstractPerformanceTracker(ABC):
    def __init__(self):
        self.criterion = None

    @abstractmethod
    def loss_function(self, output, labels, *args):
        pass

    @abstractmethod
    def metric_function(self, output, labels, *args):
        pass

    @abstractmethod
    def desired_metric_function(self, output, labels, *args):
        pass

    def track_performance(self, output, labels, *args):
        loss = self.loss_function(output, labels, *args)
        metric = self.metric_function(output, labels, *args)
        return loss, metric
