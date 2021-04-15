
from typing import Any

class _GradMode:
    _is_recording = True

    @classmethod
    def set_state(cls, record_grad: bool) -> None:
        cls._is_recording = record_grad
        return
    
    @property
    def is_recording(cls) -> bool:
        return cls._is_recording

    def __str__(self) -> str:
        if self.is_recording:
            return f"Synapse: Gradients are being recorded."
        else:
            return f"Synapse: Gradients are not being recorded."

        return 


grad_state = _GradMode()

class no_grad:
    def __init__(self):
        self.prev =  grad_state.is_recording
        return

    def __enter__(self):
        grad_state.set_state(record_grad=False)
        return

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        grad_state.set_state(self.prev)
        return




