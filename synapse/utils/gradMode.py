
from typing import Any

class __GradMode:
    createGraph = True

    @classmethod
    def setState(cls, state: bool) -> None:
        cls.createGraph = state
        return

    @classmethod
    def evalGrad(cls) -> bool:
        return cls.createGraph

GradState = __GradMode()

class NoGrad:
    def __init__(self):
        self.prev =  GradState.evalGrad()
        return

    def __enter__(self):
        GradState.setState(state=False)
        return

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        GradState.setState(self.prev)
        return




