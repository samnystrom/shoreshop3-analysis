import os
import os.path
import sqlite3
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pycwt

from shoreshop3_analysis import wavelet


shoreshop_path = Path.home() / 'abmurraylab/shoreshop3'
input_coastsat_path = shoreshop_path / 'inputdata/NC_CoastSat_smoothed.zip'
input_frf_path = shoreshop_path / 'inputdata/hindcast_1980_2023/shorelines_and_profiles/FRF_Profiles.zip'
submissions_path = shoreshop_path / 'submissions/UserSubmissionsCleaned'

workdir = Path('/work/sln33')
scratchdir = Path('/scratch')

duck_1980_2023_paths = {
    'CEERD-CHL-Devoe': os.path.join(submissions_path, 'CEERD-CHL-Devoe/mipDuck_1980-2023_waves_simplified_ShoreFor_BruunSLR_ERDCCHL_SRD.csv'),
    'CEERD-CHL-Cohn-CSHORE': os.path.join(submissions_path, 'CEERD-CHL-Cohn/CSHORE/mipDuck_1980-2023_CSHORE.csv'),
    'CEERD-CHL-Holzenthal-GENCADE': os.path.join(submissions_path, 'CEERD-CHL-Holzenthal/GENCADE/mipDuck_1980-2023_HolzenthalDingGENCADE.csv'),
    'COCollective': os.path.join(submissions_path, 'COCollective/mipDuck_1980-2023_waves_simplified-COCollective.csv'),
    'CoSMoS-COAST-conv_IA': os.path.join(submissions_path, 'CoSMoS-COAST_SV/Duck/CoSMoS_COAST_conv_model_with_interannual/mipDuck_1980-2023_CoSMoS_COAST_conv_model_with_interannual_SV.csv'),
    'CoSMoS-COAST-conv': os.path.join(submissions_path, 'CoSMoS-COAST_SV/Duck/CoSMoS_COAST_conv_model_without_interannual/mipDuck_1980-2023_CoSMoS_COAST_conv_model_without_interannual_SV.csv'),
    'CoSMoS-COAST-dmd_IA': os.path.join(submissions_path, 'CoSMoS-COAST_SV/Duck/CoSMoS_COAST_dmd_model_with_interannual/mipDuck_1980-2023_CoSMoS_COAST_dmd_model_with_interannual_SV.csv'),
    'CoSMoS-COAST-dmd': os.path.join(submissions_path, 'CoSMoS-COAST_SV/Duck/CoSMoS_COAST_dmd_model_without_interannual/mipDuck_1980-2023_CoSMoS_COAST_dmd_model_without_interannual_SV.csv'),
    'CSIRO_nearshore': os.path.join(submissions_path, 'CSIRO_nearshore/mipDuck_1980-2023_CWaCS.csv'),
    'CSIRO_nearshore-smoothed': os.path.join(submissions_path, 'CSIRO_nearshore/mipDuck_1980-2023_CWaCS_smoothed.csv'),
    'GEOOCEAN': os.path.join(submissions_path, 'GEOOCEAN/mipDuck_1980-2023_GeoOcean_bulk_SLP_Qs_partitions_N.csv'),
    'GEOOCEAN-cross_eps': os.path.join(submissions_path, 'GEOOCEAN/mipDuck_1980-2023_GeoOcean_cross_eps_bulk_SLP_Qs_partitions_N.csv'),
    'ICM-TCN': os.path.join(submissions_path, 'ICM-TCN/mipDuck_1980-2023_ICM-TCN_transect.csv'),
    'IdSOR': os.path.join(submissions_path, 'IdSOR/mipDuck_1980-2023_WaveBulkParameters_IdSOR.csv'),
    'IHCantabriaCMEG-Crossformer': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/Crossformer/mipDuck_1980-2023-Crossformer.csv'),
    'IHCantabriaCMEG-DLinear': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/DLinear/mipDuck_1980-2023-DLinear.csv'),
    'IHCantabriaCMEG-ETSformer': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/ETSformer/mipDuck_1980-2023-ETSformer.csv'),
    'IHCantabriaCMEG-FEDformer': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/FEDformer/mipDuck_1980-2023-FEDformer.csv'),
    'IHCantabriaCMEG-iTransformer': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/iTransformer/mipDuck_1980-2023-iTransformer.csv'),
    'IHCantabriaCMEG-MillerDean': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/MillerDean/mipDuck_1980-2023-MillerDean.csv'),
    'IHCantabriaCMEG-moirai': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/moirai/mipDuck_1980-2023-moirai.csv'),
    'IHCantabriaCMEG-MultiPatchFormer': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/MultiPatchFormer/mipDuck_1980-2023-MultiPatchFormer.csv'),
    'IHCantabriaCMEG-Nonstationary_Transformer': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/Nonstationary_Transformer/mipDuck_1980-2023-Nonstationary_Transformer.csv'),
    'IHCantabriaCMEG-PatchTST': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/PatchTST/mipDuck_1980-2023-PatchTST.csv'),
    'IHCantabriaCMEG-SPADS': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/SPADS/mipDuck_1980-2023-SPADS.csv'),
    'IHCantabriaCMEG-SPADS-Huber': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/SPADS-Huber/mipDuck_1980-2023-SPADS-Huber.csv'),
    'IHCantabriaCMEG-SPADS-Scale': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/SPADS-Scale/mipDuck_1980-2023-SPADS-Scale.csv'),
    'IHCantabriaCMEG-TiDE': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/TiDE/mipDuck_1980-2023-TiDE.csv'),
    'IHCantabriaCMEG-TimeFilter': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/TimeFilter/mipDuck_1980-2023-TimeFilter.csv'),
    'IHCantabriaCMEG-TimeMixer': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/TimeMixer/mipDuck_1980-2023-TimeMixer.csv'),
    'IHCantabriaCMEG-timer_xl': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/timer_xl/mipDuck_1980-2023-timer_xl.csv'),
    'IHCantabriaCMEG-TimesNet': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/TimesNet/mipDuck_1980-2023-TimesNet.csv'),
    'IHCantabriaCMEG-TimeXer': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/TimeXer/mipDuck_1980-2023-TimeXer.csv'),
    'IHCantabriaCMEG-ttm': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/ttm/mipDuck_1980-2023-ttm.csv'),
    'IHCantabriaCMEG-Yates09': os.path.join(submissions_path, 'IHCantabriaCMEG/submission_Duck_objective_1/Yates09/mipDuck_1980-2023-Yates09.csv'),
    'LEGI-ShoreFor': os.path.join(submissions_path, 'LEGI/mipDuck_1980-2023_waves_bulk-LEGIShoreFor.csv'),
    'LEGI-ShoreForeKFcp45': os.path.join(submissions_path, 'LEGI/mipDuck_1980-2023_waves_bulk-LEGIShoreForeKFcp45.csv'),
    'LEGI-ShoreForeKFcp55': os.path.join(submissions_path, 'LEGI/mipDuck_1980-2023_waves_bulk-LEGIShoreForeKFcp55.csv'),
    'mtraboulsi689': os.path.join(submissions_path, 'mtraboulsi689/mipDuck_1980-2023_waves_simplified-MohammadTraboulsi.csv'),
    # Missing: ANTOLINEZ (directory is empty)
    # Missing: ShEPreMo (24 files, don't know which to use)
    'UN-IHE-Elghandour-DA_N05': os.path.join(submissions_path, 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_N05.csv'),
    'UN-IHE-Elghandour-DA_N125': os.path.join(submissions_path, 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_N125.csv'),
    'UN-IHE-Elghandour-DA_C02': os.path.join(submissions_path, 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_C02.csv'),
    'UN-IHE-Elghandour-DA_STORM_B05': os.path.join(submissions_path, 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_STORM_B05.csv'),
    'UN-IHE-Elghandour-DA_STORM_B125': os.path.join(submissions_path, 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_STORM_B125.csv'),
    'UNSW-WRL-Calcraft': os.path.join(submissions_path, 'UNSW-WRL-Calcraft/mipDuck_1980-2023_MTFB.csv'),
    'UNSW-WRL-Mao': os.path.join(submissions_path, 'UNSW-WRL-Mao/shoreshop3Duck/mipDuck_1980-2023_GraphShore.csv'),
}

nc_1980_2023_paths = {
    'BRIE': os.path.join(submissions_path, 'BRIE/mipNC_1980-2023-BRIE.csv'),
    'CCOST': os.path.join(submissions_path, 'CCOST/mipNC_1980-2023_DESM-reconstruction-CCOST.csv'),
    'CEERD-CHL-Devoe': os.path.join(submissions_path, 'CEERD-CHL-Devoe/mipNC_1980-2023_waves_simplified_ShoreFor_BruunSLR_CUDEMslope_ERDCCHL_SRD.csv'),
    'COCollective': os.path.join(submissions_path, 'COCollective/mipNC_1980-2023_CenturyHindcast-COCollective.csv'),
    'CoSMoS-COAST-conv_IA': os.path.join(submissions_path, 'CoSMoS-COAST_SV/NC/CoSMoS_COAST_conv_model_with_interannual/mipNC_1980-2023_CoSMoS_COAST_conv_model_with_interannual_SV.csv'),
    'CoSMoS-COAST-conv': os.path.join(submissions_path, 'CoSMoS-COAST_SV/NC/CoSMoS_COAST_conv_model_without_interannual/mipNC_1980-2023_CoSMoS_COAST_conv_model_without_interannual_SV.csv'),
    'CoSMoS-COAST-Vitousek_SciReports_2024': os.path.join(submissions_path, 'CoSMoS-COAST_SV/NC/CoSMoS_COAST_Vitousek_SciReports_2024_model/mipNC_1980-2023_CoSMoS_COAST_Vitousek_SciReports_2024_model_SV.csv'),
    'CoSMoS-COAST-CoastSat_trends': os.path.join(submissions_path, 'CoSMoS-COAST_SV/NC/CoastSat_trends/mipNC_1980-2023_CoastSat_trends_SV.csv'),
    'CoSMoS-COAST-CoastSat_trends_bruun_ssp585': os.path.join(submissions_path, 'CoSMoS-COAST_SV/NC/CoastSat_trends_plus_bruun_ssp585/mipNC_1980-2023_CoastSat_trends_plus_bruun_ssp585_SV.csv'),
    'CoSMoS-COAST-DSAS_trends': os.path.join(submissions_path, 'CoSMoS-COAST_SV/NC/DSAS_trends/mipNC_1980-2023_DSAS_SV.csv'),
    'CoSMoS-COAST-persistence': os.path.join(submissions_path, 'CoSMoS-COAST_SV/NC/persistence/mipNC_1980-2023_persistence_SV.csv'),
    'CRCHI': os.path.join(submissions_path, 'CRCHI/mipNC_1980-2023_waves_simplified_smoothed_shorelines-CRCHI.csv'),
    'GEOOCEAN-cross_eps': os.path.join(submissions_path, 'GEOOCEAN/mipNC_1980-2023_GeoOcean_cross_eps_bulk_SLP_Qs_partitions_N.csv'),
    'ICM-TCN': os.path.join(submissions_path, 'ICM-TCN/mipNC_1980-2023_ICM-TCN_eof.csv'),
    'IdSOR': os.path.join(submissions_path, 'IdSOR/mipNC_1980-2023_WaveBulkParameters_IdSOR.csv'),
    'mtraboulsi689': os.path.join(submissions_path, 'mtraboulsi689/mipNC_1980-2023_waves_simplified-MohammadTraboulsi.csv'),
    'UNSW-WRL-Mao': os.path.join(submissions_path, 'UNSW-WRL-Mao/shoreshop3NC/mipNC_1980-2023_GraphShore.csv'),
}


class CoastsatData:
    def __init__(self) -> None:
        self._path = scratchdir / 'coastsat.sqlite'
        self._conn = sqlite3.connect(self._path)

        # the worker processes use this class, so avoid writing if the db is already initialized
        if self._conn.execute('select 1 from sqlite_schema limit 1').fetchone() is None:
            self._conn.executescript('''
create table transect(
    id integer primary key,
    name text unique not null
);

create table data(
    transect_id integer references transect not null,
    datetime real not null, -- julianday
    x real,
    primary key(transect_id, datetime)
) without rowid;
''')

    def populate(self) -> None:
        with zipfile.ZipFile(input_coastsat_path, 'r') as zf:
            for info in zf.infolist():
                if not info.filename.startswith('NC_CoastSat_smoothed/usa_NC_'):
                    continue

                transect = info.filename.removeprefix('NC_CoastSat_smoothed/usa_NC_').removesuffix('.csv')

                if self._conn.execute('''
                    select 1 from transect where name = ?
                ''', (transect, )).fetchone() is not None:
                    continue

                with self._conn:
                    transect_id = self._conn.execute('''
                        insert into transect (name) values (?)
                    ''', (transect, )).lastrowid
    
                    with zf.open(info.filename, 'r') as file:
                        rows = []
                        for line in file:
                            date, x = line.decode('utf-8').rstrip().split(',')
                            rows.append((transect_id, date, float(x)))
                    self._conn.executemany('''
                        insert into data (transect_id, datetime, x) values (?, julianday(?), ?)
                    ''', rows)

    def get_all(self) -> pd.DataFrame:
        df = pd.read_sql_query('''
            select transect.name as transect, datetime(datetime) as time, x
            from data
                join transect on data.transect_id = transect.id
            order by transect, time
        ''', self._conn)
        df['time'] = pd.to_datetime(df['time'])
        return df

    def get_single(self, transect_id: str) -> pd.DataFrame:
        df = pd.read_sql_query('''
            select datetime(datetime) as time, x
            from data
                join transect on data.transect_id = transect.id
            where transect.name = :transect
            order by time
        ''', self._conn, params={'transect': transect_id})
        df['time'] = pd.to_datetime(df['time'])
        return df


def load_frf_single(transect_id: str) -> pd.DataFrame:
    '''
    returns df with columns time, date, xFRF, yFRF, latitude, longitude
    '''
    with zipfile.ZipFile(input_frf_path, 'r') as zf:
        with zf.open(f'FRF_Profiles/{transect_id}/shorelinePosAtyFRF1.csv') as file:
            df = pd.read_csv(file)

    df['date'] = df['time'].apply(lambda s: s[:10])
    df['time'] = pd.to_datetime(df['time'])

    return df


def load_single_profile_models() -> pd.DataFrame:
    '''
    returns df with columns date, frf1, frf1006, model
    '''

    dfs = []
    for model, path in duck_1980_2023_paths.items():
        df = pd.read_csv(path, header=1, names=('date', 'frf1', 'frf1006')).dropna()
        df['model'] = model
        dfs.append(df)

    df = pd.concat(dfs)

    def fix_date(date: str) -> str:
        if '/' not in date:
            return date
        m, d, y = list(map(int, date.split('/')))
        return f'{y:04}-{m:02}-{d:02}'

    df['date'] = df['date'].apply(fix_date)

    return df


class MultiProfileModels:
    # Takes about 10 minutes to generate the database,
    # 10-20 seconds to copy it to/from /scratch,
    # 15 ms to read a model+transect pair

    def __init__(self, scratchdir: str | None = None) -> None:
        self._scratchdir = scratchdir
        fname = 'models-nc-1980-2023.sqlite'
        self._perm_path = Path(__file__).parent.parent / fname
        if scratchdir:
            self._active_path = Path(scratchdir) / fname
        else:
            self._active_path = self._perm_path

        if self._perm_path.exists() and self._scratchdir and not self._active_path.exists():
            subprocess.run(['cp', self._perm_path, self._active_path])

        self._conn = sqlite3.connect(self._active_path)

        self._conn.execute('pragma foreign_keys = on')
        self._conn.execute('pragma cache_size = -4000000')
        self._conn.execute('pragma mmap_size = 4000000000')

        # since this class is used by the process pool, it must avoid writes if the db is already initialized
        if self._conn.execute('select 1 from sqlite_schema limit 1').fetchone() is None:
            self._conn.executescript('''
create table if not exists model(
    id integer primary key,
    name text unique not null
);

create table if not exists transect(
    id integer primary key,
    name text unique not null
);

create table if not exists data(
    model_id integer references model not null,
    transect_id integer references transect not null,
    date real not null, -- julianday
    x real,
    primary key(model_id, transect_id, date)
) without rowid;
''')

    def populate(self, progress: bool = True) -> None:
        def fix_date(date: str) -> str:
            if '/' not in date:
                return date
            m, d, y = list(map(int, date.split('/')))
            return f'{y:04}-{m:02}-{d:02}'

        start = time.perf_counter_ns()

        updated = False

        for i, (model, path) in enumerate(nc_1980_2023_paths.items()):
            if progress:
                print(f'\rParsing model {i}/{len(nc_1980_2023_paths)}: {model}...                                                         ', end='')
            with open(path) as file:
                header = next(file)
                transects = [col.removeprefix('usa_NC_') for col in header.rstrip().split(',')[1:]]
                transect_ids = []
                with self._conn:
                    row = self._conn.execute('select id from model where name = ?', (model, )).fetchone()
                    if row is None:
                        model_id = self._conn.execute('insert or ignore into model (name) values (?)', (model, )).lastrowid
                        updated = True
                    else:
                        continue
                    for transect in transects:
                        self._conn.execute('insert or ignore into transect (name) values (?)', (transect, ))
                        id, = self._conn.execute('select id from transect where name = ?', (transect, )).fetchone()
                        transect_ids.append(id)

                    for line in file:
                        values = line.rstrip().split(',')
                        date = fix_date(values[0])
                        rows = [
                            (model_id, transect_id, date, None if x == '' else float(x))
                            for transect_id, x in zip(transect_ids, values[1:])
                        ]
                        self._conn.executemany('''
                            insert or ignore into data (model_id, transect_id, date, x)
                            values (?, ?, julianday(?), ?)
                        ''', rows)

        elapsed = (time.perf_counter_ns() - start) / 1e6
        if progress:
            print(f'\nParsed all models in {elapsed:.3f} ms')

        if updated and self._scratchdir:
            subprocess.run(['cp', self._active_path, self._perm_path])

    def get_models(self) -> set[str]:
        return {model for model, in self._conn.execute('select name from model').fetchall()}

    def get_all_transects(self) -> set[str]:
        return {transect for transect, in self._conn.execute('select name from transect').fetchall()}

    def get_transects(self, model: str) -> set[str]:
        rows = self._conn.execute('''
            select distinct transect.name
            from data
                join model on data.model_id = model.id
                join transect on data.transect_id = transect.id
            where model.name = ?
        ''', (model, )).fetchall()
        return {transect for transect, in rows}

    def get_model_transect_data(self, model: str, transect: str) -> pd.DataFrame:
        '''
        Returns: dataframe with columns model, transect, date, x
        '''
        return pd.read_sql_query('''
            select model.name as model, transect.name as transect, date(date) as date, x
            from data
                join model on data.model_id = model.id
                join transect on data.transect_id = transect.id
            where model.name = :model and transect.name = :transect
            order by date
        ''', self._conn, params={'model': model, 'transect': transect})
