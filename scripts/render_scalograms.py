import os
import sys
from pathlib import Path
from concurrent import futures

import numpy as np
import xarray as xr
import plotnine as pn

from shoreshop3_analysis import data, wct, wavelet, transect_loaders


_coastsat: data.CoastsatData | None = None
_models: data.MultiProfileModels | None = None
_wcts: data.MultiProfileWCT | None = None


def run(model: str, transect: str) -> None:
    global _coastsat
    global _models
    global _wcts
    _coastsat = _coastsat or data.CoastsatData()
    _models = _models or data.MultiProfileModels()
    _wcts = _wcts or wct.MultiProfileWCT()

    power_path = data.workdir / 'scalograms' / 'power' / model / f'{transect}.png'
    coherence_path = data.workdir / 'scalograms' / 'coherence' / model / f'{transect}.png'
    os.makedirs(power_path.parent, exist_ok=True)
    os.makedirs(coherence_path.parent, exist_ok=True)

    if not power_path.exists():
        if model == 'CoastSat':
            loader = transect_loaders.CoastsatLoader(_coastsat)
        else:
            loader = transect_loaders.ModelDataLoader(_models, model)
        w = loader.get_wavelet(transect)
        if w is not None:
            p = w.plot()
            p.items[0] += pn.labs(title=f'Wavelet Power Spectrum (1980-2023): {model} usa_NC_{transect}')
            p.save(power_path)
    
    if not coherence_path.exists() and model != 'CoastSat':
        loader = transect_loaders.ModelWCTLoader(_wcts, model)
        w = loader.get_wavelet(transect)
        if w is not None:
            p = w.plot()
            p += pn.labs(title=f'Wavelet Coherence (1980-2023): {model} usa_NC_{transect}')
            p.save(coherence_path)

    print(f'finished: {model} {transect}')


def main(args: list[str]) -> int:
    print('Initializing...')

    task = 0
    ntasks = 1
    if len(args) > 2:
        task = int(args[1])
        ntasks = int(args[2])

    wcts = wct.MultiProfileWCT()
    all_models = wcts.get_models() + ['CoastSat']
    all_transects = wcts.get_all_transects()

    with futures.ProcessPoolExecutor() as exec:
        fs = []
        print('Queueing...')
        i = 0
        for model in all_models:
            for transect in all_transects[::10]:
                if i % ntasks == task:
                    fs.append(exec.submit(run, model, transect))
                i += 1

        print('Executing...')
        for i, future in enumerate(futures.as_completed(fs)):
            print(f'{i+1}/{len(fs)}')
            future.result()

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))