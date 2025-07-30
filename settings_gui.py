import tkinter as tk
from tkinter import ttk
import threading
import config


class SettingsGUI(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.root = None

    def run(self):
        self.root = tk.Tk()
        self.root.title("Settings")
        self.root.geometry("300x350")
        self.root.attributes("-topmost", True)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        self.create_slider(main_frame, "Sensitivity", config, "SENSITIVITY", 0.1, 3.0)
        self.create_slider(
            main_frame, "Drag Sensitivity", config, "DRAG_SENSITIVITY", 0.1, 3.0
        )
        self.create_slider(
            main_frame, "Scroll Sensitivity", config, "SCROLL_SENSITIVITY", 10, 100
        )
        self.create_slider(
            main_frame, "Acceleration", config, "ACCELERATION_FACTOR", 0.0, 0.1
        )
        self.create_slider(
            main_frame, "Pinch Threshold", config, "PINCH_THRESHOLD", 0.01, 0.1
        )
        self.create_slider(
            main_frame, "Pointing Threshold", config, "POINTING_THRESHOLD", 0.01, 0.1
        )
        self.create_slider(
            main_frame, "Fist Threshold", config, "FIST_THRESHOLD", 0.05, 0.25
        )

        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.mainloop()

    def create_slider(self, parent, label, module, attr_name, from_, to_):
        frame = ttk.Frame(parent, padding=(0, 5))
        frame.pack(fill="x")
        ttk.Label(frame, text=label).pack(side="top", anchor="w")

        def update_config(value):
            setattr(module, attr_name, float(value))

        slider = ttk.Scale(
            frame, from_=from_, to=to_, orient="horizontal", command=update_config
        )
        slider.set(getattr(module, attr_name))
        slider.pack(fill="x")

    def stop(self):
        if self.root:
            self.root.destroy()
            self.root = None
