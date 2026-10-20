from abc import ABC, abstractmethod
from typing import override

import numpy as np
import pandas as pd
import geopandas as gp
import plotnine as pn

from shoreshop3_analysis import data, wavelet, wct


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


class CoastsatLoader(TransectLoader):
    def __init__(self, cs_data: data.CoastsatData) -> None:
        self._cs_data = cs_data

    @override
    def get_title(self) -> str:
        return 'Global Wavelet Power (2020–2023): CoastSat'

    @override
    def get_transects(self) -> list[str]:
        return self._cs_data.get_transects()

    @override
    def get_timeseries(self, transect: str) -> pd.DataFrame | None:
        return self._cs_data.get_single(transect)

    @override
    def get_wavelet(self, transect: str) -> wavelet.WaveletResult | None:
        df = self.get_timeseries(transect)
        return wavelet.CWT(df['time'], df['x'].values, dt=self._cs_data.dt())


class ModelDataLoader(TransectLoader):
    def __init__(self, models: data.MultiProfileModels, model: str) -> None:
        self._models = models
        self._model = model

    @override
    def get_title(self) -> str:
        return f'Global Wavelet Power (2020–2023): model={self._model}'

    @override
    def get_transects(self) -> list[str]:
        return self._models.get_transects(self._model)

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
    def __init__(self, loader: wct.MultiProfileWCT, model: str) -> None:
        self._wct = loader
        self._model = model

    @override
    def get_title(self) -> str:
        return f'Global Wavelet Coherence (2020–2023): model={self._model}'

    @override
    def get_transects(self) -> list[str]:
        return self._wct.get_transects(self._model)

    @override
    def get_timeseries(self) -> pd.DataFrame | None:
        return None

    @override
    def get_wavelet(self, transect: str) -> wavelet.WaveletResult | None:
        return self._wct.get_wct(self._model, transect)
