from dataclasses import dataclass


@dataclass
class Control:
    def __init__(self, index, level):
        self.index = index
        self.level = level
