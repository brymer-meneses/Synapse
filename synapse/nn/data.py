from synapse import Tensor
import numpy as np

from typing import Iterator, NamedTuple

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class BatchIterator:

    def __init__(self, batch_size: int, shuffle: bool = False) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> Iterator[Batch]:
        starting_points = np.arange(0, len(inputs), self.batch_size)
 
        if self.shuffle:
            np.random.shuffle(starting_points)

        for start in starting_points:

            end = start + self.batch_size
            batch_inputs = inputs[ start : end ]
            batch_targets = targets[ start : end]

            batch_inputs, batch_targets = Tensor(batch_inputs.T), Tensor(batch_targets.T)
            yield Batch(batch_inputs, batch_targets)


