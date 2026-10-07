from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pycwt


# polyfill
def override(x):
    return x


class WaveletResult(ABC):
    @abstractmethod
    def dt(self) -> float:
        pass

    @abstractmethod
    def power(self) -> np.ndarray:
        pass

    @abstractmethod
    def period(self) -> np.ndarray:
        pass

    @abstractmethod
    def coi(self) -> np.ndarray:
        pass

    @abstractmethod
    def xlabel(self, x, pos) -> str:
        pass

    @abstractmethod
    def colorbar_label(self) -> str:
        pass

    def global_power(self) -> np.ndarray:
        return np.mean(self.power(), axis=1)

    def plot(self, title: str | None = None) -> mpl.figure.Figure:
        power = self.power()
        # fft_power = np.abs(fft) ** 2
        period = self.period()
        # global_power = out.global_power()
        t = np.arange(0, len(self.coi()), 1)
    
        fig, ax = plt.subplots()
    
        fig.tight_layout()
    
        contourset = ax.contourf(t, np.log2(period), power, levels=64, extend='both', cmap='Blues')
        ax.fill(
            np.concatenate([t, t[-1:] + self.dt(), t[-1:] + self.dt(), t[:1] - self.dt(), t[:1] - self.dt()]),
            np.concatenate([np.log2(self.coi()), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
            'white',
            alpha=0.5,
            hatch='x',
        )

        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: self.xlabel(x, pos)))
        #plt.setp(plt.gca().get_xticklabels(), rotation=-45, ha='left')
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator((365*5 + 2) / self.dt()))

        def y_to_period(y, pos):
            y = int(2**y)
            if y > 365:
                years = round(y/365)
                return f'{years} year' + ('s' if years > 1 else '')
            elif y > 30:
                months = round(y/30)
                return f'{months} month' + ('s' if months > 1 else '')
            else:
                return f'{y} day' + ('s' if y > 1 else '')
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(y_to_period))
    
        fig.colorbar(contourset, label=self.colorbar_label())
    
        ax.set_xlim(np.min(t), np.max(t))
        ax.set_ylim(np.min(np.log2(period)), np.max(np.log2(period)))
    
        ax.set_ylabel('Period')
        if title:
            ax.set_title(title)
    
        fig.set_size_inches(8, 4)
    
        return fig


class CWT(WaveletResult):
    _dt: float
    _dates: pd.Series
    _wave: np.ndarray # shape: nf, nt
    _scales: np.ndarray # shape: nf
    _freqs: np.ndarray # shape: nf
    _coi: np.ndarray # shape: nt
    _fft: np.ndarray
    _fftfreqs: np.ndarray

    def __init__(self, dates: pd.Series, signal: np.ndarray, dt: float, *args, **kwargs) -> None:
        self._dt = dt
        self._dates = dates
        self._wave, self._scales, self._freqs, self._coi, self._fft, self._fftfreqs = pycwt.cwt(signal, dt, *args, **kwargs)

    @override
    def dt(self) -> float:
        return self._dt

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
    def xlabel(self, x, pos) -> str:
        idx = int(round(x))
        if 0 <= idx < len(self._dates):
            return self._dates[idx][:4]
        return ''

    @override
    def colorbar_label(self) -> str:
        return 'Wavelet power (log2)'


class WCT(WaveletResult):
    _dt: float
    _dates: pd.Series
    _wct: np.ndarray
    _awct: np.ndarray
    _coi: np.ndarray
    _freq: np.ndarray
    _sig: np.ndarray

    def __init__(self, dates: pd.Series, y1: np.ndarray, y2: np.ndarray, dt: float, *args, **kwargs) -> None:
        self._dt = dt
        self._dates = dates
        self._wct, self._awct, self._coi, self._freq, self._sig = pycwt.wct(y1, y2, dt, *args, **kwargs)

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
    def xlabel(self, x, pos) -> str:
        idx = int(round(x))
        if 0 <= idx < len(self._dates):
            return self._dates[idx][:4]
        return ''

    @override
    def colorbar_label(self) -> str:
        return 'Wavelet coherence $R^2$'

