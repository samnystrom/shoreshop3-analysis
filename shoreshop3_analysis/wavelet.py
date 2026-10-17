import math
from abc import ABC, abstractmethod
from typing import Sequence, Self, override

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotnine as pn
import pycwt


def time_regularize(
    df: pd.DataFrame,
    dt: float | pd.Timedelta,
    cols: Sequence[str] | str | None = None,
    time_col: str = 'time',
    start: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    '''
    Linearly interpolate `cols` onto a regular time grid.

    Parameters
    ----------
    df       : must contain a datetime64 column `time_col`. Need not be sorted.
               Rows with NaT time are dropped; duplicate times are averaged.
    cols     : column name or names to interpolate (numeric). All columns if None.
    time_col : name of the datetime column.
    dt       : grid spacing, in days if a number, or a pd.Timedelta.
    start    : optional grid anchor. The grid is start + k*dt for integer k,
               beginning at the first such point >= the first observation.
               If None, the grid starts at the first observation.

    Returns
    -------
    DataFrame with columns ["time", *cols]. Grid points cover
    [first grid point, last observation] inclusive, so no point is
    extrapolated. NaNs in a column are skipped for that column, which
    means gaps are bridged by straight lines.
    '''

    if cols is None:
        cols = df.columns.values.tolist()
        cols.remove(time_col)
    elif isinstance(cols, str):
        cols = [cols]
    else:
        cols = list(cols)

    # round because pd.Timedelta(days=0.3) truncates to 1 ns short of 0.3 days
    step = dt if isinstance(dt, pd.Timedelta) else pd.Timedelta(round(dt * 86_400e9), unit="ns")
    if step <= pd.Timedelta(0):
        raise ValueError(f"dt must be positive, got {dt!r}")

    d = df[[time_col, *cols]].dropna(subset=[time_col])
    if d.empty:
        raise ValueError('df has no rows with a valid time')
    d = d.groupby(time_col, sort=True)[cols].mean().reset_index()

    t_min, t_max = d[time_col].iloc[0], d[time_col].iloc[-1]
    if start is None:
        first = t_min
    else:
        start = pd.Timestamp(start)
        # ceil((t_min - start) / step) in exact integer nanoseconds
        k = max(0, -((start - t_min) // step))
        first = start + k * step

    if first > t_max:
        n = 0
    else:
        n = (t_max - first) // step + 1 # inclusive of the endpoint
    grid = first + pd.to_timedelta(np.arange(n, dtype=np.int64) * step.value, unit='ns')

    x_out = (grid - first) / step # float, in units of dt
    x_in = ((d[time_col] - first) / step).to_numpy()

    out = {time_col: grid}
    for col in cols:
        y = d[col].to_numpy(dtype=float)
        ok = ~np.isnan(y)
        if ok.sum() == 0:
            out[col] = np.full(n, np.nan)
        else:
            # mask grid points outside this column's own valid range
            v = np.interp(x_out, x_in[ok], y[ok])
            v[(x_out < x_in[ok][0]) | (x_out > x_in[ok][-1])] = np.nan
            out[col] = v
    return pd.DataFrame(out)


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

    def plot(self, title: str | None = None) -> pn.ggplot:
        power = self.power()
        j, n = power.shape
        period = np.log2(self.period())
        #global_power = self.global_power(dropnan=True)
        t = np.arange(0, n, 1)

        data = pd.DataFrame({
            'period': np.repeat(period, n),
            't': np.tile(t, j),
            'power': power.flatten(),
        })

        #global_data = pd.DataFrame({
        #    'period': period[:len(global_power)],
        #    'power': global_power,
        #})

        coi = pd.DataFrame({
            'x': np.concatenate([t, t[-1:] + self.dt(), t[-1:] + self.dt(), t[:1] - self.dt(), t[:1] - self.dt()]),
            'y': np.concatenate([np.log2(self.coi()), [1e-9], period[-1:] + 1, period[-1:] + 1, [1e-9]]),
        })

        time_scale = pn.scale_x_continuous(
            expand=(0, 0),
            breaks=np.arange(0, n, (365*5 + 2) / self.dt()),
            labels=lambda breaks: list(map(self.xlabel, breaks)),
        )
        period_scale = pn.scale_y_continuous(expand=(0, 0), labels=lambda breaks: [format_period(int(2**y)) for y in breaks])
        period_lim = (np.min(period), np.max(period))

        timeseries = self.timeseries()
        timeseries['t'] = t
        if timeseries is not None:
            timeseries_plot = (
                pn.ggplot(timeseries)
                + pn.geom_line(pn.aes('t', 'x', group=1))
                #+ pn.scale_x_date(date_breaks='5 years', date_labels='%Y')
                + time_scale
                + pn.labs(x='', y='Shoreline position (m)')
            )

        scalogram = (
            pn.ggplot(data)
            + pn.geom_raster(pn.aes('t', 'period', fill='power'))
            + pn.geom_polygon(pn.aes('x', 'y'), data=coi, fill='white', color='black', alpha=0.5)
            + pn.scale_fill_continuous(cmap_name='viridis')
            + time_scale
            + period_scale
            + pn.coord_cartesian(xlim=(0, n), ylim=period_lim)
            + pn.labs(title=title or '', x='', y='Period', fill=self.colorbar_label())
        )
        #global_plot = (
        #    pn.ggplot(global_data)
        #    + pn.geom_path(pn.aes('power', 'period'))
        #    + period_scale
        #    + pn.coord_cartesian(ylim=period_lim)
        #    + pn.labs(x=self.colorbar_label(), y='Period')
        #)

        if timeseries is not None:
            #composite = (timeseries_plot / scalogram) | ((pn.ggplot() + pn.theme_void()) / global_plot)
            return timeseries_plot / scalogram + pn.theme(figure_size=(10, 7))
        else:
            return scalogram + pn.theme(figure_size=(10, 4)) #| global_plot


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
        self._wave, self._scales, self._freqs, self._coi, self._fft, self._fftfreqs = pycwt.cwt(signal, dt, *args, **kwargs)

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
        return 'Power ($\\log_2$)'


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

