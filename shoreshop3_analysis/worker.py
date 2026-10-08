import time
import numpy as np
import pandas as pd
import pycwt

from shoreshop3_analysis import data, wavelet


_coastsat_data: data.CoastsatData | None = None
_models: data.MultiProfileModels | None = None


def init() -> None:
    global _coastsat_data
    global _models
    _coastsat_data = data.CoastsatData()
    _models = data.MultiProfileModels(scratchdir='/scratch')


def run(model: str, transect: str) -> tuple[float, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    assert _coastsat_data is not None and _models is not None

    #start = time.perf_counter()

    dt = 8.0

    obs = _coastsat_data.get_single(transect)
    obs_interp = wavelet.time_regularize(obs, 'x', dt=dt)
    model_data = _models.get_model_transect_data(model, transect).dropna()
    if len(model_data) == 0: # this transect isn't present for the model
        return None
    joined = obs_interp.merge(model_data, on='date', how='left', suffixes=('_obs', '_model'))

    try:
        wct, awct, coi, freq, sig = pycwt.wct(
            joined['x_obs'].values,
            joined['x_model'].values,
            dt=dt,
            dj=1/12,
            s0=-1,
            J=-1,
            sig=False,
            significance_level=0.95,
            wavelet='morlet',
            normalize=True,
            #mc_count=20,
            cache=False,
        )
    except e:
        # AR(1) warning
        with open('badcount', 'ab') as f:
            f.write(b'\0')
        print(e)
        return None

    sig = np.zeros(freq.shape[0])

    #elapsed = time.perf_counter() - start
    #print(f'{model} {transect} took {elapsed:.3f} s: {len(joined)}')

    return (model, transect, dt, list(joined['date'].values), wct, awct, coi, freq, sig)
