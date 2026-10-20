import os
import sys
from pathlib import Path
from concurrent import futures
import matplotlib.pyplot as plt
from shoreshop3_analysis import data, wct, transect_loaders, plot


_models: data.Models | None = None
_wcts: wct.ModelsWCT | None = None


def run(model: str) -> None:
    global _models
    global _wcts
    _models = _models or data.Models()
    try:
        _wcts = _wcts or wct.ModelsWCT()
    except FileNotFoundError:
        pass

    print(model)

    loaders = [(transect_loaders.ModelsLoader(_models, model), 'power')]
    if _wcts is not None:
        loaders.append((transect_loaders.ModelWCTLoader(_wcts, model), 'coherence'))

    all_transects = _models.get_transects()

    for loader, dir in loaders:
        base = data.multiplots_path / dir

        path = base / f'multi/{model}.png'
        if not path.exists():
            os.makedirs(path.parent, exist_ok=True)
            p = plot.plot_multi_global_sampled(loader, 80, 80)
            if p is not None:
                p.save(path, dpi=180)

        path = base / f'cape-fear/{model}.png'
        if not path.exists():
            os.makedirs(path.parent, exist_ok=True)
            p = plot.plot_multi_global_dense(loader, offset=60, n_transects=80, stride=1, max_leaders=20, title='Cape Fear')
            if p is not None:
                p.save(path, dpi=180)

        path = base / f'cape-lookout/{model}.png'
        if not path.exists():
            os.makedirs(path.parent, exist_ok=True)
            p = plot.plot_multi_global_dense(loader, offset=400, n_transects=80, stride=1, max_leaders=20, title='Cape Lookout')
            if p is not None:
                p.save(path, dpi=180)

        path = base / f'cape-hatteras/{model}.png'
        if not path.exists():
            os.makedirs(path.parent, exist_ok=True)
            p = plot.plot_multi_global_dense(loader, offset=615, n_transects=80, stride=1, max_leaders=20, title='Cape Hatteras')
            if p is not None:
                p.save(path, dpi=180)

        #densedir = base / f'dense/{model}'
        #os.makedirs(densedir, exist_ok=True)
        #for offset in range(0, len(all_transects), 80):
        #    first = all_transects[offset]
        #    last = all_transects[offset + 79]
        #    p = plot.plot_multi_global_dense(loader, offset=offset, n_transects=80, stride=1, max_leaders=20, title=f'{first}-{last}')
        #    if p is not None:
        #        p.save(densedir / f'{first}-{last}.png', dpi=180)

    plt.close()


def main(args: list[str]) -> int:
    task = 0
    ntasks = 1
    if len(args) > 2:
        task = int(args[1])
        ntasks = int(args[2])

    models = data.Models()
    with futures.ProcessPoolExecutor() as exec:
        for f in exec.map(run, models.get_models()[task::ntasks]):
            pass

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
