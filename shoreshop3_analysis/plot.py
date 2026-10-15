import numpy as np
import pandas as pd
import geopandas as gp
import plotnine as pn
import shapely

from shoreshop3_analysis import data, wct, wavelet, transect_loaders


def select_profiles(
    profiles: gp.GeoDataFrame,
    n_transects: int,
    max_leaders: int,
) -> tuple[gp.GeoDataFrame, gp.GeoDataFrame, gp.GeoDataFrame]:
    

    return profiles, selected_profiles, leader_profiles


def plot_multi_global(
    loader: transect_loaders.TransectLoader,
    n_transects: int = 50,
    max_leaders: int = 50,
) -> pn.ggplot:
    profiles = data.load_profiles()
    profiles['id'] = profiles['id'].apply(lambda id: id.removeprefix('usa_NC_'))
    profiles = profiles.drop(columns=['site_id', 'cil', 'ciu', 'trend', 'beach_slope', 'n_points', 'n_points_nonan'])
    profiles = profiles.set_index('id')

    all_transects = loader.get_transects()
    profiles = profiles[profiles.index.isin(all_transects)]

    easting = profiles.geometry.apply(lambda geom: geom.coords[0][0])
    northing = profiles.geometry.apply(lambda geom: geom.coords[0][1])

    xlim = (easting.min(), easting.max())
    ylim = (northing.min(), northing.max())

    center = ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2)
    profiles.geometry = profiles.geometry.rotate(40, origin=center)

    profiles['easting'] = profiles.geometry.apply(lambda geom: geom.coords[0][0])
    profiles['northing'] = profiles.geometry.apply(lambda geom: geom.coords[0][1])

    xlim = (profiles['easting'].min(), profiles['easting'].max())
    ylim = (profiles['northing'].min(), profiles['northing'].max())

    selected_profiles = profiles.copy().reset_index(names='id').sort_values('northing')
    selected_profiles['group'] = (
        (selected_profiles['northing'] - ylim[0]) / (ylim[1] - ylim[0]) * (n_transects - 1)
    ).astype(int)
    #def pick_midpoint(group):
    #    i = len(group) // 2
    #    return group.iloc[i] if len(group) > 0 else None
    # .apply(pick_midpoint)
    selected_profiles = selected_profiles.groupby('group').first().reset_index().drop(columns='group')

    leader_profiles = selected_profiles.copy()
    leader_profiles['group'] = (
        (leader_profiles['northing'] - ylim[0]) / (ylim[1] - ylim[0]) * (min(n_transects, max_leaders) - 1)
    ).astype(int)
    leader_profiles = leader_profiles.groupby('group').first().reset_index().drop(columns='group')

    def make_leader(geom):
        return shapely.LineString([geom.coords[0], (1e6, geom.coords[0][1])])
    leader_profiles['leader'] = leader_profiles.geometry.apply(make_leader)

    dfs = []

    min_power = 0
    max_power = 0
    for transect in selected_profiles['id'].values:
        w = loader.get_wavelet(transect)
        if w is None:
            continue
        power = w.global_power(dropnan=True)
        min_power = min(min_power, np.min(power))
        max_power = max(max_power, np.max(power))
        period = w.period()[:len(power)]
        df = pd.DataFrame({
            'id': transect,
            'northing': profiles.loc[transect, 'northing'],
            'period_log': np.log2(period),
            'period': period,
            'power': power,
        })
        dfs.append(df)

    if len(dfs) == 0:
        return pn.ggplot()

    global_powers = pd.concat(dfs)

    tile_height = (ylim[1] - ylim[0]) / n_transects * 0.8

    theme = pn.theme_classic()
    leader_color = 'tab:blue'

    return ((
        pn.ggplot()
        + pn.geom_map(data=profiles)
        + pn.geom_map(
            pn.aes(geometry='leader'),
            data=leader_profiles,
            linetype='dashed',
            color=leader_color,
            size=0.7,
        )
        + pn.geom_point(pn.aes('easting', 'northing'), data=leader_profiles, color=leader_color)
        + pn.coord_cartesian(xlim=xlim, ylim=ylim)
        + pn.labs(title=loader.get_title())
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
        + pn.scale_fill_continuous(breaks=np.linspace(min_power, max_power, 5), cmap_name='Blues')
        + pn.scale_x_log10(
            labels=lambda breaks: list(map(wavelet.format_period, breaks)),
            expand=(0, 0, 0.1, 0),
        )
        + pn.coord_cartesian(ylim=ylim)
        + pn.labs(title='', x='Period', fill=w.colorbar_label())
        + theme
        + pn.theme(
            axis_line_y=pn.element_blank(),
            axis_text_y=pn.element_blank(),
            axis_title_y=pn.element_blank(),
            axis_ticks_y=pn.element_blank(),
            plot_margin_left=0,
        )
    )) + pn.theme(figure_size=(10, 5))

