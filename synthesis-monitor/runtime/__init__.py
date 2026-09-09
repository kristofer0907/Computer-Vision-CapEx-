"""Transport between the capture loop, the thermal logger and the browser.

Nothing in here knows anything about chemistry. It moves frames from a camera
to a pipeline to a browser.

The multiprocessing supervisor that used to live here is gone: the pipeline
runs in one process now (main.py). What is left is the thread-safe slot and
ring buffer the dashboard reads, and the message dataclasses.
"""

from runtime.messages import PreviewMessage, ThermalMessage, WorkerStatus

__all__ = ["PreviewMessage", "ThermalMessage", "WorkerStatus"]
