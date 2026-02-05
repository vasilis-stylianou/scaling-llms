import time
import torch

class DeviceTimer:
    """
    Unified timing API for CPU or CUDA.

    Modes:
      - sync=True: synchronize around timing for accurate wall-clock
      - sync=False: minimal sync; better for throughput benchmarking
    """

    def __init__(self, device: str = "cpu", sync: bool = False):
        assert device in ("cpu", "cuda")
        self.device = device
        self.sync = sync

        self._t0 = None
        self._t1 = None
        self._cpu_t0 = None
        self._cpu_t1 = None

        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("TimeDevice(device='cuda') but CUDA is not available.")
            self._t0 = torch.cuda.Event(enable_timing=True)
            self._t1 = torch.cuda.Event(enable_timing=True)

    # --- API ---
    def start(self):
        if self.device == "cuda":
            if self.sync:
                torch.cuda.synchronize()
            self._t0.record(torch.cuda.current_stream())
        else:
            self._cpu_t0 = time.perf_counter()

    def stop(self):
        if self.device == "cuda":
            self._t1.record(torch.cuda.current_stream())
            if self.sync:
                torch.cuda.synchronize()
        else:
            self._cpu_t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self.device == "cuda":
            # Need sync at least when reading the event elapsed time.
            if (self._t0 is None) or (self._t1 is None):
                raise RuntimeError("CUDA timer missing start/stop.")
            if not self.sync:
                torch.cuda.synchronize()
            return float(self._t0.elapsed_time(self._t1))
        else:
            if (self._cpu_t0 is None) or (self._cpu_t1 is None):
                raise RuntimeError("CPU timer missing start/stop.")
            return float((self._cpu_t1 - self._cpu_t0) * 1e3)

