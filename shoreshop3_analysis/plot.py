import numpy as np
import pandas as pd
import geopandas as gp
import plotnine as pn
import shapely

from shoreshop3_analysis import data, wct, wavelet, transect_loaders


def _get_profiles_shoreline() -> tuple[gp.GeoDataFrame, gp.GeoDataFrame]:
    profiles = data.load_profiles()
    shoreline = data.load_shoreline()

    profiles['id'] = profiles['id'].apply(lambda id: id.removeprefix('usa_NC_'))
    profiles = profiles.drop(columns=['site_id', 'cil', 'ciu', 'trend', 'beach_slope', 'n_points', 'n_points_nonan'])
    profiles = profiles.set_index('id')

    easting = profiles.geometry.apply(lambda geom: geom.coords[-1][0])
    northing = profiles.geometry.apply(lambda geom: geom.coords[-1][1])

    xlim = (easting.min(), easting.max())
    ylim = (northing.min(), northing.max())

    center = ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2)
    angle = 40
    profiles.geometry = profiles.geometry.rotate(angle, origin=center)
    shoreline.geometry = shoreline.geometry.rotate(angle, origin=center)

    profiles['easting'] = profiles.geometry.apply(lambda geom: geom.coords[-1][0])
    profiles['northing'] = profiles.geometry.apply(lambda geom: geom.coords[-1][1])

    return profiles, shoreline


def _sample_profiles(profiles: gp.GeoDataFrame, ylim: tuple[float, float], n: int) -> gp.GeoDataFrame:
    profiles['group'] = (
        (profiles['northing'] - ylim[0]) / (ylim[1] - ylim[0]) * (n - 1)
    ).astype(int)
    def pick_midpoint(group):
        i = len(group) // 2
        return group.iloc[i] if len(group) > 0 else None
    return profiles.groupby('group').apply(pick_midpoint).reset_index().drop(columns='group')


def _select_profiles(
    profiles: gp.GeoDataFrame,
    all_transects: list[str],
    n_transects: int,
) -> list[str]:
    xlim = (profiles['easting'].min(), profiles['easting'].max())
    ylim = (profiles['northing'].min(), profiles['northing'].max())

    selected_profiles = profiles.copy().reset_index(names='id').sort_values('northing')
    selected_profiles = selected_profiles[selected_profiles['id'].isin(all_transects)]

    selected_profiles = _sample_profiles(selected_profiles, ylim, n_transects)

    return selected_profiles['id'].values.tolist()


def _select_leaders(profiles: gp.GeoDataFrame, max_leaders: int) -> gp.GeoDataFrame:
    n_leaders = min(len(profiles), max_leaders)

    xlim = (profiles['easting'].min(), profiles['easting'].max())
    ylim = (profiles['northing'].min(), profiles['northing'].max())

    leader_profiles = profiles.copy()
    leader_profiles['i'] = np.arange(len(leader_profiles))

    leader_profiles['group'] = (
        (leader_profiles['i']) / (len(leader_profiles) - 1) * (n_leaders)
    ).astype(int)
    leader_profiles = leader_profiles.groupby('group').first().reset_index().drop(columns='group')

    def make_leader(row):
        right_northing = ylim[0] + (ylim[1] - ylim[0]) / (len(profiles) - 1) * row.i
        return shapely.LineString([row.geometry.coords[-1], (xlim[1] + (xlim[1] - xlim[0]) * 0.1, right_northing)])
    leader_profiles['leader'] = leader_profiles.apply(make_leader, axis=1)

    return leader_profiles


def _plot_multi_transect(
    loader: transect_loaders.TransectLoader,
    transects: list[str],
    max_leaders: int,
    show_full_coast: bool,
    leader_mode: 'lines' | 'labels',
    title: str = '',
) -> pn.ggplot:
    profiles, shoreline = _get_profiles_shoreline()

    xlim = (profiles['easting'].min(), profiles['easting'].max())
    ylim = (profiles['northing'].min(), profiles['northing'].max())

    selected_profiles = profiles.loc[transects].reset_index(names='id')
    if len(selected_profiles) == 0:
        return pn.ggplot()
    leader_profiles = _select_leaders(selected_profiles, max_leaders)
    n_transects = len(selected_profiles)

    if not show_full_coast:
        xlim = (selected_profiles['easting'].min(), selected_profiles['easting'].max())
        ylim = (selected_profiles['northing'].min(), selected_profiles['northing'].max())

    dfs = []

    min_power = 0
    max_power = 0
    wave = None
    for i, transect in enumerate(selected_profiles['id'].values):
        w = loader.get_wavelet(transect)
        wave = wave or w
        if w is None:
            continue
        power = w.global_power(dropnan=True)
        if len(power) == 0:
            continue
        min_power = min(min_power, np.min(power))
        max_power = max(max_power, np.max(power))
        period = w.period()[:len(power)]
        df = pd.DataFrame({
            'id': transect,
            'northing': ylim[0] + (ylim[1] - ylim[0]) / (n_transects - 1) * i,
            'period_log': np.log2(period),
            'period': period,
            'power': power,
        })
        dfs.append(df)

    cbar_limits = wave.colorbar_limits()
    if cbar_limits is not None:
        min_power, max_power = cbar_limits

    if len(dfs) == 0:
        return pn.ggplot()

    global_powers = pd.concat(dfs).set_index('id')

    leader_profiles['x'] = df['period'].max() * 1.1
    northing = global_powers['northing'][~global_powers.index.duplicated()]
    leader_profiles['y'] = leader_profiles['id'].map(northing)

    tile_height = (ylim[1] - ylim[0]) / n_transects * 0.8

    theme = pn.theme_classic()

    leader_lines = pn.geom_blank()
    leader_labels = pn.geom_blank()
    spectrum_labels = pn.geom_blank()
    if leader_mode == 'lines':
        leader_lines = pn.geom_map(
            pn.aes(geometry='leader'),
            data=leader_profiles,
            linetype='dashed',
            color='black',
            size=0.5,
        )
    elif leader_mode == 'labels':
        leader_labels = pn.geom_label(
            pn.aes('easting', 'northing', label='i'),
            data=leader_profiles,
            size=8,
            ha='center',
            #nudge_x=-1000,
        )
        spectrum_labels = pn.geom_label(
            pn.aes('x', 'y', label='i'),
            data=leader_profiles,
            size=8,
            ha='left',
        )

    return ((
        pn.ggplot()
        + pn.geom_map(data=shoreline, fill='gray', color='k')
        + pn.geom_map(data=profiles, size=0.2)
        + leader_lines
        + pn.geom_point(pn.aes('easting', 'northing'), data=selected_profiles, color='tab:red')
        + leader_labels
        + pn.scale_x_continuous(expand=(0.1, 0))
        + pn.coord_cartesian(xlim=xlim, ylim=ylim)
        + pn.labs(title=loader.get_title() + (f' ({title})' if title else ''))
        + theme
        + pn.theme(
            axis_line=pn.element_blank(),
            axis_text=pn.element_blank(),
            axis_title=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            plot_margin_right=0,
            plot_title=pn.element_text(ha='left'),
        )
    ) | (
        pn.ggplot(global_powers)
        + pn.geom_tile(pn.aes('period', 'northing', fill='power', height=tile_height))
        + spectrum_labels
        + pn.scale_fill_continuous(
            limits=wave.colorbar_limits(),
            breaks=np.linspace(min_power, max_power, 5),
            cmap_name='viridis',
        )
        + pn.scale_x_log10(
            breaks=[2, 17, 31*3, 367, 366*4, 366*14],
            labels=lambda breaks: list(map(wavelet.format_period, breaks)),
            expand=(0, 0, 0.1, 0),
        )
        + pn.coord_cartesian(ylim=ylim)
        + pn.labs(title='', x='Period', fill=wave.colorbar_label())
        + theme
        + pn.theme(
            axis_line_y=pn.element_blank(),
            axis_text_y=pn.element_blank(),
            axis_title_y=pn.element_blank(),
            axis_ticks_y=pn.element_blank(),
            plot_margin_left=0,
        )
    )) + pn.theme(figure_size=(10, 8))


def plot_multi_global_sampled(
    loader: transect_loaders.TransectLoader,
    n_transects: int = 80,
    max_leaders: int = 80,
) -> pn.ggplot:
    profiles, _shoreline = _get_profiles_shoreline()
    all_transects = loader.get_transects()
    transects = _select_profiles(profiles, all_transects, n_transects)

    return _plot_multi_transect(loader, transects, max_leaders, show_full_coast=True, leader_mode='lines')


def plot_multi_global_dense(
    loader: transect_loaders.TransectLoader,
    offset: int,
    n_transects: int = 80,
    max_leaders: int = 80,
    stride: int = 1,
    title: str = '',
) -> pn.ggplot:
    all_transects = loader.get_transects()
    transects = all_transects[offset:offset+n_transects*stride:stride]

    return _plot_multi_transect(loader, transects, max_leaders, show_full_coast=False, leader_mode='labels', title=title)
