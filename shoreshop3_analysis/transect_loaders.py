from abc import ABC, abstractmethod
from typing import override

import numpy as np
import pandas as pd
import geopandas as gp
import plotnine as pn

from shoreshop3_analysis import data, wavelet, wct


def _year(dt: np.datetime64) -> int:
    return dt.astype('datetime64[Y]').astype(int) + 1970


class TransectLoader(ABC):
    @abstractmethod
    def get_title(self) -> str:
        pass

    @abstractmethod
    def get_transects(self) -> list[str]:
        pass

    @abstractmethod
    def get_timeseries(self, transect: str) -> pd.DataFrame | None:
        '''
        Returns: dataframe with columns time (datetime64) and x (float), or none if
            a timeseries wouldn't be meaningful for this loader
        '''
        pass

    @abstractmethod
    def get_wavelet(self, transect: str) -> wavelet.WaveletResult | None:
        pass


class ObservationsLoader(TransectLoader):
    def __init__(self, obs: data.Observations) -> None:
        self._obs = obs

    @override
    def get_title(self) -> str:
        lower, upper = map(_year, self._obs.get_range())
        return f'Global Wavelet Power ({lower}–{upper}): {self._obs.name()}'

    @override
    def get_transects(self) -> list[str]:
        return self._obs.get_transects()

    @override
    def get_timeseries(self, transect: str) -> pd.DataFrame | None:
        return self._obs.get_single(transect)

    @override
    def get_wavelet(self, transect: str) -> wavelet.WaveletResult | None:
        df = self.get_timeseries(transect)
        return wavelet.CWT(df['time'], df['x'].values, dt=self._obs.dt())


class ModelsLoader(TransectLoader):
    def __init__(self, models: data.Models, model: str) -> None:
        self._models = models
        self._model = model

    @override
    def get_title(self) -> str:
        lower, upper = map(_year, self._models.get_range())
        return f'Global Wavelet Power ({lower}–{upper}): model={self._model}'

    @override
    def get_transects(self) -> list[str]:
        return self._models.get_transects()

    @override
    def get_timeseries(self, transect: str) -> pd.DataFrame | None:
        data = self._models.get_model_transect_data(self._model, transect)
        return None if len(data) == 0 else data

    @override
    def get_wavelet(self, transect: str) -> wavelet.WaveletResult | None:
        df = self.get_timeseries(transect)
        if df is None:
            return None
        return wavelet.CWT(df['time'], df['x'].values, dt=1)


class ModelWCTLoader(TransectLoader):
    def __init__(self, loader: wct.ModelsWCT, model: str) -> None:
        self._wct = loader
        self._model = model

    @override
    def get_title(self) -> str:
        lower, upper = map(_year, self._wct.get_range())
        return f'Global Wavelet Coherence ({lower}–{upper}): model={self._model}'

    @override
    def get_transects(self) -> list[str]:
        return self._wct.get_transects()

    @override
    def get_timeseries(self, transect: str) -> pd.DataFrame | None:
        return None

    @override
    def get_wavelet(self, transect: str) -> wavelet.WaveletResult | None:
        return self._wct.get_wct(self._model, transect)
