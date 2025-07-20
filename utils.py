from enum import Enum


class State(Enum):
    IDLE = 0
    MOUSE_MOVEMENT = 1
    DRAGGING = 2
    SCROLLING = 3
