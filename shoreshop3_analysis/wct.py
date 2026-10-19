import json
import os
import sys
import sqlite3
from concurrent import futures

import numpy as np
import pandas as pd
import xarray as xr
import dask.array
import zarr

from shoreshop3_analysis import util, data, worker, wavelet


class MultiProfileWCT:
    def __init__(self) -> None:
        if data.model_coherence_path.exists():
            self._ds = xr.open_zarr(data.model_coherence_path, consolidated=False)
            return

        self._coastsat_data = data.CoastsatData()
        self._models = data.MultiProfileModels()
        all_models = self._models.get_models()
        all_transects = self._models.get_all_transects()
        M, T = len(all_models), len(all_transects)

        dt = 8

        n = 0
        min_time, max_time = self._coastsat_data.get_range()
        diff = (max_time - min_time) / np.timedelta64(1, 'D')
        n = int(np.ceil(diff / dt))

        flambda = 4*np.pi / (6 + np.sqrt(2 + 6**2))
        j = int(np.round(12 * np.log2(n * flambda / 2))) + 1

        self._ds = xr.Dataset(
            data_vars={
                'start':     (('model', 'transect'),          np.full((M, T), 0, dtype=int)),
                'period':    (('model', 'transect', 'scale'), np.full((M, T, j), np.nan, dtype=np.float32)),
                'sig':       (('model', 'transect', 'scale'), np.full((M, T, j), np.nan, dtype=np.float32)),
                'coi':       (('model', 'transect', 'time'),  np.full((M, T, n), np.nan, dtype=np.float32)),
                'coherence': (('model', 'transect', 'scale', 'time'), dask.array.full((M, T, j, n), np.nan, dtype=np.float32, chunks=(1, 1, j, n))),
            },
            coords={
                'model': all_models,
                'transect': all_transects,
                'time': np.arange(n),
                'scale': np.arange(j),
            },
            attrs={
                'title': '',
                'source': '',
                'dt': dt,
            },
        )
        encoding = {
            'start': {'chunks': (1, 10)},
            'period': {'chunks': (1, 10, j)},
            'sig': {'chunks': (1, 10, j)},
            'coi': {'chunks': (1, 10, n)},
        }
        self._ds.to_zarr(data.model_coherence_path, compute=False, consolidated=False, zarr_format=3, encoding=encoding)

    def compute(self) -> None:
        timer = util.Timer()
        print('Initializing...')

        coastsat_data = data.CoastsatData()
        models = data.MultiProfileModels()

        arr = zarr.open_group(data.model_coherence_path, mode='r+', zarr_format=3)

        mem_gb = util.get_peak_mem() / 1024**3

        with futures.ProcessPoolExecutor(max_workers=os.process_cpu_count()-1) as executor:
            fs = []
            all_models = models.get_models()
            all_transects = models.get_all_transects()
            for i, j in np.array(np.isnan(self._ds.period[:, :, 0].values).nonzero()).T:
                fs.append(executor.submit(worker.run, all_models[i], all_transects[j], (i, j)))

            for i, future in enumerate(futures.as_completed(fs)):
                timer.stop()
                elapsed = timer.ns
                predicted_time = elapsed / (i+1) * len(fs)
                predicted = util.Timer()
                predicted._start = timer._start
                predicted._end = predicted._start + predicted_time
                if i % 10 == 0:
                    mem_gb = util.get_peak_mem() / 1024**3
                print(f'\rProcessing result {i+1}/{len(fs)}: peak mem={mem_gb:.1f} GiB time={timer}/{predicted}', end='', flush=True)
                sys.stdout.flush()

                r = future.result()
                if r is None:
                    continue # transect is missing from this model or WCT failed

                arr['start'][r.model_idx, r.transect_idx] = int((r.start - np.datetime64(0, 'D')) / np.timedelta64(1, 'D'))
                arr['period'][r.model_idx, r.transect_idx, :r.j] = r.period.astype(np.float32)
                arr['sig'][r.model_idx, r.transect_idx, :r.j] = r.sig.astype(np.float32)
                arr['coi'][r.model_idx, r.transect_idx, :r.n] = r.coi.astype(np.float32)

        print(f'\nFinished in {timer}')

    def get_models(self) -> list[str]:
        return self._ds.model.values.tolist()

    def get_transects(self, model: str) -> list[str]:
        return self._ds.transect.values.tolist()

    def get_all_transects(self) -> list[str]:
        return self._ds.transect.values.tolist()

    def get_wct(self, model: str, transect: str) -> wavelet.WCT | None:
        model_idx = self.get_models().index(model)
        transect_idx = self.get_all_transects().index(transect)

        coh = self._ds.coherence[model_idx, transect_idx, :, :].values
        n = len(coh[0][~np.isnan(coh[0])])
        coh = coh[~np.isnan(coh)].reshape((-1, n))
        j, n = coh.shape

        start = int(self._ds.start[model_idx, transect_idx].values)
        start = np.datetime64(0, 'D') + np.timedelta64(start, 'D')
        period = self._ds.period[model_idx, transect_idx, :j].values.astype(np.float64)
        sig = self._ds.sig[model_idx, transect_idx, :j].values
        coi = self._ds.coi[model_idx, transect_idx, :n].values.astype(np.float64)

        times = start + (np.arange(n) * np.timedelta64(1, 'D'))
        times = pd.Series(times)

        dt = 8

        return wavelet.WCT(dt, times, coh, coi, 1 / period, sig)
