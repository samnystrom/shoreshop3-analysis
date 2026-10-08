import json
import os
import sys
import sqlite3
import time
from concurrent import futures

import numpy as np
import pandas as pd

from shoreshop3_analysis import data, worker, wavelet


class MultiProfileWCT:
    def __init__(self) -> None:
        self._path = data.scratchdir / 'models-nc-1980-2023-cwt.sqlite'

        self._conn = sqlite3.connect(self._path)

        self._conn.execute('pragma foreign_keys = on')
        self._conn.execute('pragma cache_size = -4000000')
        self._conn.execute('pragma mmap_size = 4000000000')
        # benchmarks show no performance difference with larger pages and a
        # very slight increase in storage space
        # self._conn.execute('pragma page_size = 65536')

        self._conn.executescript('''
create table if not exists model(
    id integer primary key,
    name text unique not null
);

create table if not exists transect(
    id integer primary key,
    name text unique not null
);

create table if not exists metadata(
    id integer primary key,
    model_id integer references model not null,
    transect_id integer references transect not null,
    n integer not null, -- number of timesteps
    j integer not null, -- number of scales
    dt real not null, -- days
    dates text not null, -- JSON array length N of iso8601 dates
    freq blob, -- size J, LE, float64
    sig blob, -- size J, LE, float64
    coi blob, -- size N, LE, float64
    unique(model_id, transect_id)
);

-- sqlite can't jump into the middle of a large row without reading everything in front,
-- so storing the large blobs in their own tables ensures access remains fast.

create table if not exists wct(
    id integer primary key references metadata,
    wct blob -- shape (J,N), LE, row-major, float64
);

create table if not exists awct(
    id integer primary key references metadata,
    awct blob -- shape (J,N), LE, row-major, float64
);
''')

    def compute(self, progress: bool = True) -> None:
        def format_duration(s: float) -> str:
            return f'{int(s/3600)}:{int(s/60%60):02}:{int(s%60):02}'

        if progress:
            start = time.perf_counter()
            print('Initializing...')

        coastsat_data = data.CoastsatData()
        coastsat_data.populate()
        models = data.MultiProfileModels(scratchdir='/scratch')
        models.populate(progress=False)

        if progress:
            init_time = time.perf_counter() - start
            print(f'Initialization finished in {format_duration(init_time)}')

        for transect in list(models.get_all_transects()):
            with self._conn:
                self._conn.execute('''
                    insert or ignore into transect (name) values (?)
                ''', (transect, ))

        with futures.ProcessPoolExecutor(
            # process_cpu_count is python 3.13
            # max_workers=os.process_cpu_count()-1,
            max_workers=31,
            initializer=worker.init,
        ) as executor:

            if progress:
                start = time.perf_counter()

            fs = []
            for model in list(models.get_models()):
                #if model != 'CCOST':
                #    continue
    
                with self._conn:
                    self._conn.execute('''
                        insert or ignore into model (name) values (?)
                    ''', (model, ))
    
                for transect in list(models.get_transects(model)):
                    #if not transect.startswith('0001'):
                    #    continue

                    if self._conn.execute('''
                        select 1
                        from metadata
                            join model on metadata.model_id = model.id
                            join transect on metadata.transect_id = transect.id
                        where model.name = ? and transect.name = ?
                    ''', (model, transect)).fetchone() is None:
                        fs.append(executor.submit(worker.run, model, transect))

            for i, future in enumerate(futures.as_completed(fs)):
                if progress:
                    elapsed = time.perf_counter() - start
                    predicted_time = elapsed / (i+1) * len(fs)
                    elapsed_s = format_duration(elapsed)
                    predicted_s = format_duration(predicted_time)
                    print(f'\rProcessing result {i+1}/{len(fs)}: {elapsed_s}/{predicted_s}', end='', flush=True)
                    sys.stdout.flush()

                result = future.result()
                if result is None:
                    continue # transect is missing from this model or WCT failed
                model, transect, dt, dates, wct, awct, coi, freq, sig = result
                dates_json = json.dumps(dates)
                j, n = wct.shape
                wct = np.ascontiguousarray(wct, dtype=np.float64).data
                awct = np.ascontiguousarray(awct, dtype=np.float64).data
                coi = np.ascontiguousarray(coi, dtype=np.float64).data
                freq = np.ascontiguousarray(freq, dtype=np.float64).data
                sig = np.ascontiguousarray(sig, dtype=np.float64).data
                with self._conn:
                    meta_rowid = self._conn.execute('''
                        insert into metadata
                            (model_id, transect_id, n, j, dt, dates, freq, sig, coi)
                        values (
                            (select id from model where name = :model),
                            (select id from transect where name = :transect),
                            :n,
                            :j,
                            :dt,
                            :dates,
                            zeroblob(:j*8),
                            zeroblob(:j*8),
                            zeroblob(:n*8)
                        )
                    ''', {
                        'model': model,
                        'transect': transect,
                        'n': n,
                        'j': j,
                        'dt': dt,
                        'dates': dates_json,
                    }).lastrowid
    
                    with self._conn.blobopen('metadata', 'freq', meta_rowid) as blob:
                        blob.write(freq)
                    with self._conn.blobopen('metadata', 'sig', meta_rowid) as blob:
                        blob.write(sig)
                    with self._conn.blobopen('metadata', 'coi', meta_rowid) as blob:
                        blob.write(coi)
    
                    wct_rowid = self._conn.execute('''
                        insert into wct (id, wct) values (?, zeroblob(?))
                    ''', (meta_rowid, n*j*8)).lastrowid
                    with self._conn.blobopen('wct', 'wct', wct_rowid) as blob:
                        blob.write(wct)
    
                    awct_rowid = self._conn.execute('''
                        insert into awct (id, awct) values (?, zeroblob(?))
                    ''', (meta_rowid, n*j*8)).lastrowid
                    with self._conn.blobopen('awct', 'awct', awct_rowid) as blob:
                        blob.write(awct)

        if progress:
            elapsed = time.perf_counter() - start + init_time
            print(f'\nFinished in {format_duration(elapsed)}')

    def get_wct(self, model: str, transect: str) -> wavelet.WCT:
        pass

