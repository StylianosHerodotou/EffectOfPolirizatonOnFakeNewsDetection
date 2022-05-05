from abc import ABC, abstractmethod


class AbstractPerformanceTracker(ABC):
    def __init__(self):
        self.criterion = None

    @abstractmethod
    def loss_function(self, output, labels, **kwargs):
        pass

    @abstractmethod
    def metric_function(self, output, labels, **kwargs):
        pass

    @abstractmethod
    def desired_metric_function(self, output, labels, **kwargs):
        pass

    def track_performance(self, output, labels, **kwargs):
        loss = self.loss_function(output, labels, **kwargs)
        metric = self.metric_function(output, labels, **kwargs)
        return loss, metric
