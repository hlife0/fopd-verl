"""Record NVML GPU-busy samples (driver ~200 ms windows, wall-clock us timestamps) to CSV.

Usage: python gpu_busy_sampler.py OUT.csv 4,5,6,7   (stops on SIGTERM/SIGINT)
"""

import signal
import sys
import time

import pynvml

out_path, devs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
pynvml.nvmlInit()
handles = {d: pynvml.nvmlDeviceGetHandleByIndex(d) for d in devs}
last = {d: int(time.time() * 1e6) for d in devs}
running = True


def _stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)
with open(out_path, "w") as fh:
    fh.write("gpu,ts,util\n")
    while running:
        for d, h in handles.items():
            try:
                _, samples = pynvml.nvmlDeviceGetSamples(h, pynvml.NVML_GPU_UTILIZATION_SAMPLES, last[d])
            except pynvml.NVMLError:
                continue
            for s in samples:
                if s.timeStamp > last[d]:
                    fh.write(f"{d},{s.timeStamp / 1e6:.4f},{s.sampleValue.uiVal}\n")
                    last[d] = s.timeStamp
        fh.flush()
        time.sleep(1.0)
