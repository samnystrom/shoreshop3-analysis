import time
from abc import ABC, abstractmethod
from typing import Sequence, Self

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotnine as pn
import pycwt


# polyfill
def override(x):
    return x


def time_regularize(df: pd.DataFrame, cols: str | Sequence[str], dt: float) -> pd.DataFrame:
    '''
    df should have a "time" column which is a pandas datetime. cols are the columns to interpolate.

    Returns: dataframe with columns date, ...cols
    '''

    if isinstance(cols, str):
        cols = (cols, )

    start = df['time'].min()
    range = (df['time'].max() - start).days
    t = np.arange(0, range, dt)

    out_cols = {'time': t}
    in_t = df['time'].apply(lambda time: (time - start).days)
    for col in cols:
        out_cols[col] = np.interp(t, in_t, df[col])

    df_interp = pd.DataFrame(out_cols)
    df_interp['time'] = df_interp['time'].apply(lambda time: start + pd.Timedelta(days=time))
    df_interp['date'] = df_interp['time'].apply(lambda time: time.date().isoformat())

    # time is no longer meaningful since we sampled on days

    return df_interp.drop(columns='time')


def format_period(period: int | float) -> str:
    period = int(period)
    if period > 365:
        years = round(period / 365)
        return f'{years} year' + ('s' if years > 1 else '')
    elif period > 30:
        months = round(period / 30)
        return f'{months} month' + ('s' if months > 1 else '')
    elif period > 7:
        weeks = round(period / 7)
        return f'{weeks} week' + ('s' if weeks > 1 else '')
    else:
        return f'{period} day' + ('s' if period > 1 else '')


class WaveletResult(ABC):
    @abstractmethod
    def dt(self) -> float:
        pass

    @abstractmethod
    def timeseries(self) -> pd.DataFrame | None:
        '''
        Returns: dataframe with columns date (iso8601 str), x (float), or None if n/a
        '''
        pass

    @abstractmethod
    def power(self) -> np.ndarray:
        '''
        Returns: ndarray shape (J, N)
        '''
        pass

    @abstractmethod
    def period(self) -> np.ndarray:
        '''
        Returns: ndarray shape J
        '''
        pass

    @abstractmethod
    def coi(self) -> np.ndarray:
        '''
        Returns: ndarray shape N
        '''
        pass

    @abstractmethod
    def xlabel(self, x) -> str:
        pass

    @abstractmethod
    def colorbar_label(self) -> str:
        pass

    def global_power(self, dropnan: bool = False) -> np.ndarray:
        # Mask out the COI by setting all values within it to zero and adjusting the mean calculation accordingly
        period = self.period()
        coi = self.coi()
        mask = np.tile(period, len(coi)).reshape((len(coi), len(period))).T >= coi
        power = self.power().copy()
        power[mask] = 0
        counts = np.count_nonzero(~mask, axis=1)
        # Ignore divide by zero, results in nan for scales with no valid data, which is desired behavior
        with np.errstate(divide='ignore'):
            # An arithmetic mean in log space is equivalent to a geometric mean,
            # which is the right kind of mean for wavelet powers.
            mean = np.sum(power, axis=1) / counts
        if dropnan:
            mean = mean[~np.isnan(mean)]
        return mean

    def plot(self, title: str | None = None) -> mpl.figure.Figure:
        power = self.power()
        j, n = power.shape
        period = np.log2(self.period())
        # global_power = out.global_power()
        t = np.arange(0, n, 1)

        data = pd.DataFrame({
            'period': np.repeat(period, n),
            't': np.tile(t, j),
            'power': power.flatten(),
        })

        coi = pd.DataFrame({
            'x': np.concatenate([t, t[-1:] + self.dt(), t[-1:] + self.dt(), t[:1] - self.dt(), t[:1] - self.dt()]),
            'y': np.concatenate([np.log2(self.coi()), [1e-9], period[-1:] + 1, period[-1:] + 1, [1e-9]]),
        })

        return (
            pn.ggplot(data)
            + pn.geom_raster(pn.aes('t', 'period', fill='power'))
            + pn.geom_polygon(pn.aes('x', 'y'), data=coi, fill='white', color='black', alpha=0.5)
            + pn.scale_fill_continuous(cmap_name='viridis')
            + pn.scale_x_continuous(
                breaks=np.arange(0, n, (365*5 + 2) / self.dt()),
                labels=lambda breaks: list(map(self.xlabel, breaks)),
            )
            + pn.scale_y_continuous(labels=lambda breaks: [format_period(int(2**y)) for y in breaks])
            + pn.coord_cartesian(xlim=(0, n), ylim=(np.min(period), np.max(period)), expand=False)
            + pn.labs(title=title or '', x='', y='Period', fill=self.colorbar_label())
            + pn.theme(figure_size=(8, 4))
        )


class CWT(WaveletResult):
    _dt: float
    _dates: pd.Series
    _signal: np.ndarray # shape: N
    _wave: np.ndarray # shape: J, N
    _scales: np.ndarray # shape: J
    _freqs: np.ndarray # shape: J
    _coi: np.ndarray # shape: N
    _fft: np.ndarray
    _fftfreqs: np.ndarray

    def __init__(self, dates: pd.Series, signal: np.ndarray, dt: float, *args, **kwargs) -> None:
        self._dt = dt
        self._dates = dates
        self._signal = signal
        start = time.perf_counter_ns()
        self._wave, self._scales, self._freqs, self._coi, self._fft, self._fftfreqs = pycwt.cwt(signal, dt, *args, **kwargs)
        elapsed = time.perf_counter_ns() - start

    @override
    def dt(self) -> float:
        return self._dt

    @override
    def timeseries(self) -> pd.DataFrame:
        return pd.DataFrame({'date': self._dates, 'x': self._signal})

    @override
    def power(self) -> np.ndarray:
        return np.log2(np.abs(self._wave) ** 2)

    @override
    def period(self) -> np.ndarray:
        return 1 / self._freqs

    @override
    def coi(self) -> np.ndarray:
        return self._coi

    @override
    def xlabel(self, x) -> str:
        idx = int(round(x))
        if 0 <= idx < len(self._dates):
            return self._dates[idx][:4]
        return ''

    @override
    def colorbar_label(self) -> str:
        return 'Wavelet power ($\log_2$)'


class WCT(WaveletResult):
    _dt: float
    _dates: pd.Series
    _wct: np.ndarray # shape: J, N
    _awct: np.ndarray # shape: J, N
    _coi: np.ndarray # shape: N
    _freq: np.ndarray # shape: J
    _sig: np.ndarray # shape: J

    @classmethod
    def fromdata(cls, dates: pd.Series, y1: np.ndarray, y2: np.ndarray, dt: float, *args, **kwargs) -> Self:
        return cls(dt, dates, *pycwt.wct(y1, y2, dt, *args, **kwargs))

    def __init__(self, dt: float, dates: pd.Series, wct: np.ndarray, awct: np.ndarray, coi: np.ndarray, freq: np.ndarray, sig: np.ndarray) -> None:
        self._dt = dt
        self._dates = dates
        self._wct = wct
        self._awct = awct
        self._coi = coi
        self._freq = freq
        self._sig = sig

    @override
    def timeseries(self) -> pd.DataFrame | None:
        return None

    @override
    def dt(self) -> float:
        return self._dt

    @override
    def power(self) -> np.ndarray:
        return self._wct

    @override
    def period(self) -> np.ndarray:
        return 1 / self._freq

    @override
    def coi(self) -> np.ndarray:
        return self._coi

    @override
    def xlabel(self, x) -> str:
        idx = int(round(x))
        if 0 <= idx < len(self._dates):
            return self._dates[idx][:4]
        return ''

    @override
    def colorbar_label(self) -> str:
        return 'Coherence ($R^2$)'

