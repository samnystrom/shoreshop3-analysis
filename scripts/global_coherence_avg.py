import sys
from concurrent import futures

import numpy as np
import xarray as xr
import zarr

from shoreshop3_analysis import data, wct


_wcts: wct.MultiProfileWCT | None = None
_arr: zarr.Array | None = None


def run(model: str, transect: str, model_idx: int, transect_idx: int) -> None:
    global _wcts
    global _arr
    _wcts = _wcts or wct.MultiProfileWCT()
    _arr = _arr or zarr.open_group(data.global_coherence_path, mode='r+', zarr_format=3)

    idx = model_idx, transect_idx

    if not np.isnan(_arr['period'][*idx, 0]):
        return

    w = _wcts.get_wct(model, transect)
    if w is None:
        return
    gp = w.global_power(dropnan=True)
    j = gp.size

    _arr['period'][*idx, :j] = w.period()[:j] # shorter because of COI masking
    _arr['coherence'][*idx, :j] = gp

    print(f'finished: {model} {transect} ({model_idx}, {transect_idx})')


def main(args: list[str]) -> int:
    print('Initializing...')

    task = 0
    ntasks = 1
    if len(args) > 2:
        task = int(args[1])
        ntasks = int(args[2])

    wcts = wct.MultiProfileWCT()
    all_models = wcts.get_models()
    all_transects = wcts.get_all_transects()
    M, T, j, n = wcts._ds.coherence.shape

    if not data.global_coherence_path.exists():
        ds = xr.Dataset(
            data_vars={
                'period':    (('model', 'transect', 'scale'), np.full((M, T, j), np.nan, dtype=np.float64)),
                'coherence': (('model', 'transect', 'scale'), np.full((M, T, j), np.nan, dtype=np.float64)),
            },
            coords={
                'model': all_models,
                'transect': all_transects,
                'scale': np.arange(j),
            },
            attrs={
                'title': '',
                'source': '',
            },
        )
        encoding = {
            'period': {'chunks': (1, 1, j)},
            'coherence': {'chunks': (1, 1, j)},
        }
        ds.to_zarr(data.global_coherence_path, consolidated=False, zarr_format=3, encoding=encoding)

    with futures.ProcessPoolExecutor() as exec:
        fs = []
        print('Queueing...')
        for a, model in enumerate(all_models):
            for b, transect in enumerate(all_transects[task::ntasks]):
                fs.append(exec.submit(run, model, transect, a, b))

        print('Executing...')
        for i, future in enumerate(futures.as_completed(fs)):
            print(f'{i+1}/{len(fs)}')
            future.result()

    print()

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
