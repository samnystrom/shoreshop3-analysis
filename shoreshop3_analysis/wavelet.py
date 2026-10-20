import math
from abc import ABC, abstractmethod
from typing import Sequence, Self, override

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotnine as pn
import pyleoclim
import skimage
import scipy


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
    units = [(365, 'year'), (30, 'month'), (1, 'day')]
    for length, unit in units:
        if length == 1 or period > length:
            x = round(period / length)
            return str(x) + ' ' + unit + ('s' if x != 1 else '')
    assert False, 'unreachable'


class WaveletResult(ABC):
    @abstractmethod
    def dt(self) -> float:
        pass

    @abstractmethod
    def timeseries(self) -> pd.DataFrame | None:
        '''
        Returns: dataframe with columns group, date (iso8601 str), x (float); or None if n/a
        '''
        pass

    @abstractmethod
    def trend(self) -> pd.DataFrame:
        '''
        Returns: dataframe with columns group, a, b for y=ax+b
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
    def sig(self) -> np.ndarray | None:
        '''
        Returns: ndarray shape J
        '''
        pass

    @abstractmethod
    def xlabel(self, x) -> str:
        pass

    @abstractmethod
    def colorbar_label(self) -> str:
        pass

    @abstractmethod
    def colorbar_limits(self) -> tuple[float, float] | None:
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
        with np.errstate(divide='ignore', invalid='ignore'):
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
        t = np.arange(n)

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
        period_scale = pn.scale_y_continuous(
            expand=(0, 0),
            breaks=np.log2(np.array([2, 17, 31*3, 367, 366*4, 366*14])),
            labels=lambda breaks: [format_period(int(2**y)) for y in breaks],
        )
        period_lim = (np.min(period), np.max(period))

        sig = self.sig()
        contours_geom = pn.geom_blank()
        if sig is not None:
            mask = power > sig[:, np.newaxis]
            sigma = 1
            # different blur for the two differently-scaled axes
            sigma = sigma, sigma / (j / n * 2)
            pad = int(3 * sigma[1])
            padded = np.pad(mask.astype(float), pad, constant_values=0.0)
            soft = scipy.ndimage.gaussian_filter(padded, sigma=sigma)

            dfs = []
            for i, c in enumerate(skimage.measure.find_contours(soft, 0.5)):
                dfs.append(pd.DataFrame({
                    'group': i,
                    'x': c[:, 1] - 1 - pad,
                    'y': (c[:, 0] - 1 - pad) / (j - 1) * (period_lim[1] - period_lim[0]) + period_lim[0],
                }))
            contours = pd.concat(dfs, ignore_index=True)

            contours_geom = pn.geom_polygon(
                pn.aes('x', 'y', group='group'),
                data=contours,
                color='k',
                #fill='white',
                alpha=0.0,
            )

        timeseries = self.timeseries()
        if timeseries is not None:
            timeseries['t'] = t
            #trend = self.trend()
            #trend['b'] = -trend['a'] * n / 2
            timeseries_plot = (
                pn.ggplot(timeseries)
                + pn.geom_line(pn.aes('t', 'x', group=1))
                #+ pn.geom_abline(pn.aes(slope='a', intercept='b', color='"Trend (removed)"'), data=trend)
                #+ pn.scale_color_manual({'Trend (removed)': 'tab:red'}, name='')
                #+ pn.scale_x_date(date_breaks='5 years', date_labels='%Y')
                + time_scale
                + pn.labs(x='', y='Shoreline position (m)')
                + pn.theme(legend_title=pn.element_blank())
            )

        scalogram = (
            pn.ggplot(data)
            + pn.geom_raster(pn.aes('t', 'period', fill='power'))
            + contours_geom
            + pn.geom_polygon(pn.aes('x', 'y'), data=coi, fill='white', color='black', alpha=0.5)
            + pn.scale_fill_continuous(cmap_name='viridis', limits=self.colorbar_limits())
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
    _times: pd.Series
    _slope: float
    _intercept: float
    _signal: np.ndarray # shape: N
    _wave: np.ndarray # shape: J, N
    _freqs: np.ndarray # shape: J
    _coi: np.ndarray # shape: N

    def __init__(self, times: pd.Series, x: np.ndarray, dt: float) -> None:
        self._dt = dt
        self._times = times
        self._signal = x

        n, dj = x.size, 1/12
        w0 = 6 # Morlet parameter

        # pycwt's grid: s0=-1 -> 2*dt/flambda, J=-1 -> round(log2(n*dt/s0)/dj)
        flambda = 4 * np.pi / (w0 + np.sqrt(2 + w0**2))   # Fourier factor of the Morlet wavelet
        s0 = 2 * dt / flambda
        J = int(np.round(np.log2(n * dt / s0) / dj))
        self._freqs = 1 / (flambda * s0 * 2 ** (np.arange(J + 1) * dj))

        t = np.arange(n) * dt
        self._slope, self._intercept = np.polyfit(t, x, 1)
        xd = x - (self._slope * t + self._intercept)

        m = int(2 ** np.ceil(np.log2(n)))
        ts = pyleoclim.Series(np.arange(m) * dt, np.concatenate([xd, np.zeros(m - n)]), verbose=False)
        wave = ts.wavelet(method='cwt', freq=self._freqs, settings={'standardize': False, 'pad': False, 'mother': 'MORLET', 'param': w0})

        self._wave = np.asarray(wave.amplitude)[:n].T
        self._coi = flambda / np.sqrt(2) * dt * (n / 2 - np.abs(np.arange(n) - (n - 1) / 2))

    @override
    def dt(self) -> float:
        return self._dt

    @override
    def timeseries(self) -> pd.DataFrame:
        return pd.DataFrame({'time': self._times, 'x': self._signal})

    @override
    def trend(self) -> pd.DataFrame:
        return pd.DataFrame({
            'a': [self._slope],
            'b': [self._intercept],
        })

    @override
    def power(self) -> np.ndarray:
        return np.log2(self._wave)
        #return np.log2(np.abs(self._wave) ** 2)

    @override
    def period(self) -> np.ndarray:
        return 1 / self._freqs

    @override
    def coi(self) -> np.ndarray:
        return self._coi

    @override
    def sig(self) -> np.ndarray | None:
        return None

    @override
    def xlabel(self, x) -> str:
        idx = int(round(x))
        if 0 <= idx < len(self._times):
            return self._times[idx].year
        return ''

    @override
    def colorbar_label(self) -> str:
        return 'Power ($\\log_2$)'

    @override
    def colorbar_limits(self) -> tuple[float, float] | None:
        return None


class WCT(WaveletResult):
    _dt: float
    _times: pd.Series
    #_x: np.ndarray
    #_y: np.ndarray
    #_x_slope: float
    #_x_intercept: float
    #_y_slope: float
    #_y_intercept: float
    _wct: np.ndarray # shape: J, N
    _coi: np.ndarray # shape: N
    _freq: np.ndarray # shape: J
    _sig: np.ndarray # shape: J

    @classmethod
    def fromdata(cls, times: pd.Series, x: np.ndarray, y: np.ndarray, dt: float, dj: float = 1/12, w0: int = 6) -> Self:
        n = x.size
        flambda = 4 * np.pi / (w0 + np.sqrt(2 + w0**2))
        s0 = 2 * dt / flambda
        J = int(np.round(np.log2(n * dt / s0) / dj))
        freqs = 1 / (flambda * s0 * 2 ** (np.arange(J + 1) * dj))
        coi = flambda / np.sqrt(2) * dt * (n / 2 - np.abs(np.arange(n) - (n - 1) / 2))

        t = np.arange(n) * dt
        
        ax, bx = np.polyfit(t, x, 1)
        ay, by = np.polyfit(t, y, 1)
        xd = x - (ax * t + bx)
        yd = y - (ay * t + by)

        sx = pyleoclim.Series(time=t, value=xd, verbose=False)
        sy = pyleoclim.Series(time=t, value=yd, verbose=False)
        coh = sx.wavelet_coherence(sy, method='cwt', settings={
            'freq': np.sort(freqs), 'pad': False, 'standardize': True,
            'smooth_factor': 0.5, 'mother': 'MORLET', 'param': w0})
        order = np.argsort(np.asarray(coh.frequency))[::-1] # pycwt's scale order
        wct = np.asarray(coh.wtc).T[order]
        freq = np.asarray(coh.frequency)[order]

        settings = {'freq': np.sort(freqs), 'pad': False, 'smooth_factor': 0.5,
                    'mother': 'MORLET', 'param': w0}

        np.random.seed(0)
        st = coh.signif_test(number=100, method='ar1sim', qs=[0.95], mute_pbar=True, nproc=1)
        thr = np.asarray(st.signif_qs[0].scalogram_list[0].amplitude).T[order]
        valid = (1 / freq)[:, np.newaxis] < coi[np.newaxis, :]
        has_valid = valid.any(axis=1)
        sig = np.full(thr.shape[0], np.nan)
        sig[has_valid] = np.nanmedian(np.where(valid[has_valid], thr[has_valid], np.nan), axis=1)
        sig[~has_valid] = np.nanmedian(thr[~has_valid], axis=1)

        return cls(dt, times, wct, coi, freq, sig)

    def __init__(self, dt: float, times: pd.Series, wct: np.ndarray, coi: np.ndarray, freq: np.ndarray, sig: np.ndarray) -> None:
        self._dt = dt
        self._times = times
        self._wct = wct
        self._coi = coi
        self._freq = freq
        self._sig = sig

    @override
    def timeseries(self) -> pd.DataFrame | None:
        return None

    @override
    def trend(self) -> pd.DataFrame:
        return pd.DataFrame({
            'a': [self._x_slope, self._y_slope],
            'b': [self._x_intercept, self._y_intercept],
        })

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
    def sig(self) -> np.ndarray | None:
        return self._sig

    @override
    def xlabel(self, x) -> str:
        idx = int(round(x))
        if 0 <= idx < len(self._times):
            return self._times[idx].year
        return ''

    @override
    def colorbar_label(self) -> str:
        return 'Coherence ($R^2$)'

    @override
    def colorbar_limits(self) -> tuple[float, float] | None:
        return (0.0, 1.0)
