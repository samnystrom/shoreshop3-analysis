import sys
from concurrent import futures

import numpy as np
import zarr

from shoreshop3_analysis import data, wct


_wcts: wct.ModelsWCT | None = None
_grp: zarr.Group | None = None


def run(model: str, transect: str, model_idx: int, transect_idx: int) -> None:
    global _wcts
    global _grp
    _wcts = _wcts or wct.ModelsWCT()
    _grp = _grp or zarr.open_group(data.global_coherence_path, mode='r+', zarr_format=3)

    idx = model_idx, transect_idx

    if not np.isnan(_grp['period'][*idx, 0]):
        return

    w = _wcts.get_wct(model, transect)
    if w is None:
        return
    gp = w.global_power(dropnan=True)
    j = gp.size

    _grp['period'][*idx, :j] = w.period()[:j] # shorter because of COI masking
    _grp['coherence'][*idx, :j] = gp

    print(f'finished: {model} {transect} ({model_idx}, {transect_idx})')


def main(args: list[str]) -> int:
    print('Initializing...')

    task = 0
    ntasks = 1
    if len(args) > 2:
        task = int(args[1])
        ntasks = int(args[2])

    wcts = wct.ModelsWCT()
    all_models = wcts.get_models()
    all_transects = wcts.get_transects()
    M, T, j, n = wcts._ds.coherence.shape

    if not data.global_coherence_path.exists():
        grp = zarr.open_group(data.global_coherence_path, mode='w', zarr_format=3)

        grp.create_array(name='model', chunks=(M,), dimension_names=('model',), data=np.array(all_models))
        grp.create_array(name='transect', chunks=(T,), dimension_names=('transect',), data=np.array(all_transects))
        grp.create_array(name='scale', chunks=(j,), dimension_names=('scale',), data=np.arange(j))

        kwargs = dict(shape=(M, T, j), chunks=(1, 1, j), dimension_names=('model', 'transect', 'scale'),
                      dtype=np.float64, fill_value=np.nan)
        grp.create_array(name='period', **kwargs)
        grp.create_array(name='coherence', **kwargs)

    with futures.ProcessPoolExecutor() as exec:
        fs = []
        print('Queueing...')
        i = 0
        for a, model in enumerate(all_models):
            for b, transect in enumerate(all_transects):
                if i % ntasks == task:
                    fs.append(exec.submit(run, model, transect, a, b))
                i += 1

        print('Executing...')
        for i, future in enumerate(futures.as_completed(fs)):
            print(f'{i+1}/{len(fs)}')
            future.result()

    print()

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
