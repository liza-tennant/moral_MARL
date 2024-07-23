"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

simple time profiler for code optimisation.
"""

import time
from collections import deque

import numpy as np


class TimeProfiler:
    def __init__(self, profile_time=True, maxlen=100):
        self.times = dict()
        for name in ["start", "elapsed"]:
            self.times[name] = dict()
        self.maxlen = maxlen
        self.profile_time = profile_time

    def start(self, name):
        if self.profile_time:
            self.times["start"][name] = time.time()

    def end(self, name):
        if self.profile_time:
            if name not in self.times["elapsed"]:
                self.times["elapsed"][name] = deque(maxlen=self.maxlen)
            self.times["elapsed"][name].append(time.time() - self.times["start"][name])

    def summarise(self, path_save, idx_run=0):
        if self.profile_time:
            #if path_save: 
            #    method = "a"
            #else: 
            #    method = "w"
            with open(path_save, "a") as fp:
                title = f"[TimeProfiler-{idx_run}] ### Elapsed Time ###"
                print(title)
                fp.write(title + "\n")
                len_longest = 0
                for k in self.times["elapsed"]:
                    if len(k) > len_longest:
                        len_longest = len(k)
                for k in self.times["elapsed"]:
                    mean_times = np.mean(self.times["elapsed"][k])
                    std_times = np.std(self.times["elapsed"][k])
                    num_items = len(self.times["elapsed"][k])
                    buffer_line = " " * (len_longest - len(k))
                    line = (
                        buffer_line
                        + f"    {k}({num_items:>4} items)"
                        + f" | mean: {mean_times:.4f} sec | std: {std_times:.4f} sec"
                    )
                    print(line)
                    fp.write(line + "\n")
                fp.write("\n")