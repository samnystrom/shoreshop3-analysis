import os
import sys
from pathlib import Path
from concurrent import futures

import numpy as np
import plotnine as pn

from shoreshop3_analysis import data, wct, wavelet, transect_loaders


_obs: data.Observations | None = None
_models: data.Models | None = None
_wcts: data.ModelsWCT | None = None


def _year(dt: np.datetime64) -> int:
    return dt.astype('datetime64[Y]').astype(int) + 1970


def run(model: str, transect: str) -> None:
    global _coastsat
    global _models
    global _wcts
    _obs = _obs or data.Observations()
    _models = _models or data.Models()
    _wcts = _wcts or wct.ModelsWCT()

    power_path = data.scalograms_path / 'power' / model / f'{transect}.png'
    coherence_path = data.scalograms_path / 'coherence' / model / f'{transect}.png'
    os.makedirs(power_path.parent, exist_ok=True)
    os.makedirs(coherence_path.parent, exist_ok=True)

    if not power_path.exists():
        if model == _obs.name():
            loader = transect_loaders.ObservationsLoader(_coastsat)
        else:
            loader = transect_loaders.ModelsLoader(_models, model)
        w = loader.get_wavelet(transect)
        if w is not None:
            p = w.plot()
            lower, upper = map(_year, _models.get_range())
            p.items[0] += pn.labs(title=f'Wavelet Power Spectrum ({lower}-{upper}): {model} {transect}')
            p.save(power_path)
    
    if not coherence_path.exists() and model != 'FRF':
        loader = transect_loaders.ModelWCTLoader(_wcts, model)
        w = loader.get_wavelet(transect)
        if w is not None:
            p = w.plot()
            lower, upper = map(_year, _wcts.get_range())
            p += pn.labs(title=f'Wavelet Coherence ({lower}-{upper}): {model} {transect}')
            p.save(coherence_path)

    print(f'finished: {model} {transect}')


def main(args: list[str]) -> int:
    print('Initializing...')

    task = 0
    ntasks = 1
    if len(args) > 2:
        task = int(args[1])
        ntasks = int(args[2])

    obs = data.Observations()
    wcts = wct.ModelsWCT()
    all_models = wcts.get_models() + [obs.name()]
    all_transects = wcts.get_all_transects()

    with futures.ProcessPoolExecutor() as exec:
        fs = []
        print('Queueing...')
        i = 0
        for model in all_models:
            for transect in all_transects[::1]:
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