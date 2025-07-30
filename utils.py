# /move-mouse-with-finger/utils.py

from enum import auto, Enum


class State(Enum):
    MOUSE_MOVEMENT = auto()
    IDLE = auto()  # Default state
    DRAGGING = auto()  # Left hand is pinched
    SCROLLING = auto()  # Left hand is a fist
