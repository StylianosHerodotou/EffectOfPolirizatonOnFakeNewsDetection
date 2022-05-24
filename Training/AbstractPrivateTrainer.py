from abc import ABC, abstractmethod

from Training.AbstractTrainer import AbstractTrainer


class AbstractPrivateTrainer(AbstractTrainer,ABC):
    def __init__(self):
        super().__init__()