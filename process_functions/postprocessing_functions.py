import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches   # for manual legend entries

import config


# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)



def plot_rasterize_2d_maps_reg(abs_link_Processed_tif, lv_l_arr, do_satellite, name, limits, pixel_size):
    """
    Creates geotagged tif-file from the land value array

    :param abs_link_Processed_tif: location where tif is stored to
    :param lv_l_arr: array containing the land value/satellite image classification
    :param do_satellite: boolean if array with satellite classification is to be stored
    :param name: name of the approach, used for the filename
    :param limits: not used currently, possible for clusterring
    """
    fig = plt.figure(figsize=(2.75, 2.4))


    raster_in = abs_link_Processed_tif + 'IMU2000_new.tif'

    if do_satellite == 1:
        raster_out2 = abs_link_Processed_tif + 'Satellite_prediction_level_' + name + '.tif'
    else:
        raster_out2 = abs_link_Processed_tif + 'Land_value_level_' + name + '.tif'

    with rio.open(raster_in) as src:
        profile = src.profile.copy()

        with rio.open(raster_out2, 'w', **profile) as dst:
            dst.write(np.float32(lv_l_arr), 1)
            # plt.show()

    # create new colormap with group colours in the middle of the borders
    if len(limits) == 4:
        colorlist = [(0., 0., 0.536), (0., 0.504, 1.), (0.49, 1., 0.478), (1., 0.582, 0.), (0.536, 0., 0.)]


        limits_list = [0, 0.236, 0.404, 0.685, 1]
        cm2 = LinearSegmentedColormap.from_list('jet_own', zip(limits_list, colorlist))
        cm2.set_over(colorlist[-1])
        cm2.set_under(colorlist[0])
        cm = cm2

    else:
        cm = 'jet'



    fig = plt.figure(figsize=(2.75, 2.4))
    im = plt.imshow(np.flipud(-lv_l_arr), interpolation=None, cmap=cm.reversed(), alpha=0.9, vmin=-1.5, vmax=1.5)  # norm=LogNorm()

    # create colorbar
    cbar = plt.colorbar(im, shrink=0.75, ticks=[-1.5] + [0] + [1.5])
    cbar.ax.set_ylabel('Land Value', labelpad=0)
    cbar.set_label("Land Value", fontsize=8, x=1, y=.5, rotation=90)

    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_tick_params(pad=1, length=0)
    cs = plt.contour(np.flipud(lv_l_arr), limits, colors='k', alpha=0.9, linestyles='solid', linewidths=0.3)

    # add axes labels and change ticks
    ax = plt.gca()
    ticks = (ax.get_xticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_xticklabels(ticks)
    ax.xaxis.set_tick_params(pad=-1)
    ticks = (ax.get_yticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_yticklabels(ticks)
    ax.yaxis.set_tick_params(pad=-1)
    plt.xlabel('East in km')
    plt.ylabel('North in km')
    plt.gca().invert_yaxis()

    plt.gca().set_aspect(aspect='equal', adjustable='box')

    plt.savefig(config.global_path + '0_Land_Value_Map_level_closest_2000_' + name + '.pdf', dpi=300, bbox_inches="tight")

