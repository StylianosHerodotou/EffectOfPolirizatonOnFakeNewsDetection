from abc import ABC, abstractmethod

from Training.AbstractTrainer import AbstractTrainer


class AbstractPublicTrainer(AbstractTrainer,ABC):
    def __init__(self):
        super().__init__()