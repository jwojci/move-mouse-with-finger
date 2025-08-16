# MouseCam - Control your mouse through your camera 

## Description
This application enables users to control their mouse through a gesture-based system using a standard webcam. It utilizes a two-handed approach for precision: the left hand is detected for action gestures (like clicking and scrolling), while the right hand controls the cursor's position.


---

## Demo 

<video src="./demo/demo.mp4" controls="false" autoplay="true" loop="true" muted="true" playsinline="true" width="100%"></video>

---

## Features 

* **Multi-threaded Architecture:** The application decouples camera I/O and model inference from the main thread to maintain a responsive user experience.
* **Kalman Filter Smoothing:** A Kalman filter is implemented to smooth the raw hand-tracking coordinates, eliminating jitter and providing stable cursor control.
* **State-Driven Actions:** A state machine manages user actions (e.g., `IDLE`, `DRAGGING`, `SCROLLING`) based on detected gestures.
* **Dynamic Configuration:** A real-time settings GUI allows for on-the-fly adjustments of sensitivity, acceleration, and gesture detection thresholds.

---

## Technology Stack

* **Core:** Python 3.11.5
* **Package Management:** uv
* **Computer Vision:** OpenCV, MediaPipe
* **Mouse Control:** PyAutoGUI
* **Signal Processing:** NumPy, filterpy
* **GUI:** Tkinter

--- 

## Installation and Usage 

### Prerequisites

* Python 3.10 or newer.
* [uv installed](https://docs.astral.sh/uv/getting-started/installation/) 

### Setup 

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jwojci/MouseCam.git](https://github.com/jwojci/MouseCam.git)
    cd MouseCam
    ```

2.  **Create, sync and activate a virtual environment:**
    ```bash
    uv venv
    uv sync 
    .venv\Scripts\activate
    ```

3.  **Run the application:**
    ```bash
    uv run main.py
    ```

## Controls 

The system uses a two-hand control scheme for better precision.

* **Control Hand (Right Hand):** Used for pointing to move the cursor and for vertical movement during scrolling.
* **Action Hand (Left Hand):** Used to change states and trigger actions.

| Action          | Gesture                                                              |
| --------------- | -------------------------------------------------------------------- |
| **Activate/Pause** | Show a "Victory" (✌️) gesture with the action hand.              
| **Move Cursor** | Point with the control hand's index finger.                          |
| **Scroll** | Make a "Thumb Up" (👍) gesture with the action hand.
| **Drag & Drop** | Pinch the action hand's thumb and index finger.                      |
| **Click** | Quickly pinch & release the action hand's thumb and index finger.                      |
