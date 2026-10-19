import time


def get_peak_mem() -> int:
    with open('/proc/self/cgroup') as f:
        cgroup = next(f).rstrip().split('/', 1)[1]
    with open(f'/sys/fs/cgroup/{cgroup}/memory.peak') as f:
        return int(next(f).rstrip())


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter_ns()
        self._end = 0

    def start(self) -> None:
        self._start = time.perf_counter_ns()

    def stop(self) -> None:
        self._end = time.perf_counter_ns()

    def __enter__(self) -> None:
        self.start()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.stop()

    @property
    def ns(self) -> int:
        if self._end < self._start:
            self.stop()
        return self._end - self._start

    @property
    def us(self) -> float:
        return self.ns / 1e3

    @property
    def ms(self) -> float:
        return self.ns / 1e6

    @property
    def s(self) -> float:
        return self.ns / 1e9

    @property
    def hms(self) -> str:
        s = self.s
        return f'{int(s//3600):02}:{int(s//60%60):02}:{int(s%60):02}.{int(s%1*1000):03}'

    def __str__(self) -> str:
        if self.ns < 1000:
            return f'{self.ns} ns'
        elif self.us < 1000:
            return f'{self.us} μs'
        elif self.ms < 1000:
            return f'{self.ms} ms'
        elif self.s < 60:
            return f'{self.s:.3f} s'
        else:
            return self.hms
