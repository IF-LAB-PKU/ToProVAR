import torch.cuda.nvtx as nvtx
import time
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
import uuid
import torch

class GlobalNVTXProfiler:
    def __init__(self):
        self.stats = defaultdict(list)
        self._current_ranges = {}

    @contextmanager
    def range(self, tag, color=None):
        range_uuid = uuid.uuid4()
        start_time = time.perf_counter()
        
        torch.cuda.synchronize()
        with nvtx.range(tag, color=color):
            self._current_ranges[range_uuid] = (tag, start_time)
            try:
                yield
            finally:
                if range_uuid in self._current_ranges:
                    tag, start = self._current_ranges.pop(range_uuid)
                    duration = time.perf_counter() - start
                    self.stats[tag].append(duration)
        torch.cuda.synchronize()

    def print_stats(self):
        print("\n===== 跨文件 NVTX 标记统计结果 =====")
        for tag, durations in self.stats.items():
            mean = np.mean(durations) * 1000
            std = np.std(durations) * 1000
            count = len(durations)
            total = sum(durations) * 1000
            print(f"标记: {tag}")
            print(f"  执行次数: {count}")
            print(f"  平均耗时: {mean:.4f} ms")
            print(f"  总耗时: {total:.4f} ms")
            print(f"  标准差: {std:.4f} ms\n")
        print("====================================\n")

nvtx_profiler = GlobalNVTXProfiler()
