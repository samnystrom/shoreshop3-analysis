import sys
from pathlib import Path
from concurrent import futures
from shoreshop3_analysis import data, wct, transect_loaders, plot


_models: data.MultiProfileModels | None = None
_wcts: wct.MultiProfileWCT | None = None


def plot_multi(model):
    global _models
    global _wcts
    _models = _models or data.MultiProfileModels()
    _wcts = _wcts or wct.MultiProfileWCT()

    print(model)

    loaders = [
        # (transect_loaders.ModelDataLoader(_models, model), 'power'),
        (transect_loaders.ModelWCTLoader(_wcts, model), 'coherence'),
    ]

    for loader, dir in loaders:
        base = Path(__file__).parent.parent / 'figures' / dir

        path = base / f'multi/{model}.png'
        if not path.exists():
            p = plot.plot_multi_global_sampled(loader, 80, 80)
            p.save(path, dpi=180)

        path = base / f'cape-fear/{model}.png'
        if not path.exists():
            p = plot.plot_multi_global_dense(loader, offset=60, n_transects=80, stride=1, max_leaders=20, title='Cape Fear')
            p.save(path, dpi=180)

        path = base / f'cape-lookout/{model}.png'
        if not path.exists():
            p = plot.plot_multi_global_dense(loader, offset=400, n_transects=80, stride=1, max_leaders=20, title='Cape Lookout')
            p.save(path, dpi=180)

        path = base / f'cape-hatteras/{model}.png'
        if not path.exists():
            p = plot.plot_multi_global_dense(loader, offset=615, n_transects=80, stride=1, max_leaders=20, title='Cape Hatteras')
            p.save(path, dpi=180)


def main() -> int:
    models = data.MultiProfileModels()
    with futures.ProcessPoolExecutor() as exec:
        for f in exec.map(plot_multi, models.get_models()):
            f.result()
    return 0


if __name__ == '__main__':
    sys.exit(main())
