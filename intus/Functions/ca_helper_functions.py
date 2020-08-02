import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches   # for manual legend entries
from matplotlib.lines import Line2D
import rasterio as rio
from osgeo import gdal
import gc

SMALLER_SIZE = 6
SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALLER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_lv_map(land_value, result_dir = None, pixel_size=30.0):
    '''
    Plots and saves the land value map

    :param land_value: land_value map as numpy array
    :param result_dir: directory where figure is saved to
    '''
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1

    #fig = plt.figure(figsize=(land_value.shape[0] / 160. + 0.5, land_value.shape[1] / 160.), dpi=300)
    fig = plt.figure(figsize=(2.75, 2.4), dpi=300)

    # new colormap
    colorlist = [(0., 0., 0.536), (0., 0.504, 1.), (0.49, 1., 0.478), (1., 0.582, 0.), (0.536, 0., 0.)]
    limits_list = [0, 0.236, 0.404, 0.685, 1]
    cm2 = LinearSegmentedColormap.from_list('jet_own', zip(limits_list, colorlist))
    cm2.set_over(colorlist[-1])
    cm2.set_under(colorlist[0])
    cm = cm2


    # plot image with title
    im = plt.imshow(np.flipud(-land_value), interpolation=None, cmap=cm.reversed(), alpha=0.9, vmin=-1.5, vmax=1.5)  # norm=LogNorm()
    #plt.title('Land Value Map Level')

    # specify colorbar
    limits = [-0.95978, -0.62325, 0.04980, 1.05937]
    cbar = plt.colorbar(im, shrink=0.6, fraction=0.1, ticks=[-1.5] + limits + [1.5])
    cbar.ax.set_ylabel('land value', fontsize=SMALL_SIZE)
    cbar.set_label("land value", fontsize=SMALL_SIZE, y=0.5, rotation=90)
    cbar.ax.text(-1.6, 0.09, "G1", fontsize=SMALL_SIZE, rotation=0, va='center')
    cbar.ax.text(-1.6, 0.22, "G2", fontsize=SMALL_SIZE, rotation=0, va='center')
    cbar.ax.text(-1.6, 0.4, "G3", fontsize=SMALL_SIZE, rotation=0, va='center')
    cbar.ax.text(-1.6, 0.68, "G4", fontsize=SMALL_SIZE, rotation=0, va='center')
    cbar.ax.text(-1.6, 0.9, "G5", fontsize=SMALL_SIZE, rotation=0, va='center')
    cbar.ax.yaxis.set_label_position('right')

    plt.gca().set_aspect(aspect='equal', adjustable='box')
    plt.gca().invert_yaxis()
    ax = plt.gca()

    ticks = (ax.get_xticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_xticklabels(ticks)
    ax.xaxis.set_tick_params(pad=-1)

    ticks = (ax.get_yticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_yticklabels(ticks)
    ax.yaxis.set_tick_params(pad=-1)
    plt.xlabel('East in km')
    plt.ylabel('North in km')

    plt.tight_layout()

    if result_dir != None:
        plt.savefig(result_dir + '/_land_value_map.pdf', bbox_inches = "tight", dpi=300)

    # plt.show()
    plt.close(fig)
    gc.collect()

    return



def plot_ca_map(self, se_levels_growth = None, target = None, result_dir = None, year = 0, without_lv = 0, pixel_size=30.0, name = None):
    '''
    plots either the reference IMU distribution or the metropolitan area with socioeconomic levels and the input layers

    :param self: Simulation object which incorporates the city with the layers
    :param se_levels_growth: gives the socioeconomic levels as array as additional input and determines if they should be shown
    :param target: array of the reference IMU
    :param result_dir: directory where figure is saved to
    :param year: index of simulation year/growth cycle for numbering in the filename
    :param without_lv: if lv shall be omitted in the first year
    '''
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1

    #fig = plt.figure(figsize=(self.VALUE.shape[0] / 160. + 1 , self.VALUE.shape[1] / 160.), dpi=300)
    fig = plt.figure(figsize=(2.75, 2.4), dpi=300)

    # new colormap
    colorlist = [(0., 0., 0.536), (0., 0.504, 1.), (0.49, 1., 0.478), (1., 0.582, 0.), (0.536, 0., 0.)]
    limits_list = [0, 0.236, 0.404, 0.685, 1]
    cm2 = LinearSegmentedColormap.from_list('jet_own', zip(limits_list, colorlist))
    cm2.set_over(colorlist[-1])
    cm2.set_under(colorlist[0])
    cm = cm2
    my_cmap = plt.cm.get_cmap('jet')

    # plot sleuth layers
    if target is None and year == 0 and without_lv == 0:
        im = plt.imshow(-self.true_LandValue[::-1], interpolation=None, cmap=cm.reversed(), alpha=0.9, vmin=-1.5, vmax=1.5, origin='upper')

    plt.imshow(self.SLOPE, cmap='Greys', alpha=0.8)
    EXCLUDED_alpha = np.copy(self.EXCLUDED).astype('float32')
    EXCLUDED_alpha[EXCLUDED_alpha == 0] = np.nan
    plt.imshow(EXCLUDED_alpha, cmap='Greys', vmin=0, vmax=5)
    TRANSPORT_alpha = np.copy(self.TRANSPORT).astype('float32')
    TRANSPORT_alpha[TRANSPORT_alpha == 0] = np.nan
    im3 = plt.imshow(TRANSPORT_alpha, cmap='Greys_r')

    cbar = plt.colorbar(im3, shrink=0.2, drawedges=False)
    cbar.remove()

    if target is not None and (name=='IMU2005_new' or name=='IMU2010_new'):
        Target_alpha = np.copy(target).astype('float32')
        Target_alpha[Target_alpha == 0] = np.nan
        im2 = plt.imshow(Target_alpha, cmap=my_cmap, vmin=1, vmax=5)

        # calculate share of urbanized
        urbanized = np.array(np.copy(target).astype('int'))
        urban_seed = np.array(np.copy(self.URBAN).astype('int'))
        urbanized[urban_seed == 1] = 0
        added = np.bincount(urbanized.flatten())[1:6]
        added = added.astype('float32')
        total_added = sum(added)
        print 'Number of added se-groups', name, ':', added
        print 'Share of added se-groups', name, ':', added / total_added

    if se_levels_growth is not None and year > 0:
        SE_alpha = np.copy(se_levels_growth).astype('float32')
        SE_alpha[SE_alpha == [0]] = np.nan
        plt.imshow(SE_alpha, cmap=my_cmap, vmin=1, vmax=5)

    URBAN_alpha = np.copy(self.URBAN).astype('float32')
    URBAN_alpha[URBAN_alpha == 0] = np.nan
    plt.imshow(URBAN_alpha, cmap='Greys_r')

    if target is None:
        # create legend
        g0 = Line2D([0], [0], marker='3', color='gray', label='Slope',
                    markerfacecolor='white', markersize=10, linestyle='None')
        g01 = mpatches.Patch(color='black', label='Urban Seed')
        g02 = mpatches.Patch(color='lightgray', label='Protected')
        g03 = Line2D([0], [0], color='black', lw=1, label='Roads')
        g1 = mpatches.Patch(color=colorlist[0], label='Group 1')
        g2 = mpatches.Patch(color=colorlist[1], label='Group 2')
        g3 = mpatches.Patch(color=colorlist[2], label='Group 3')
        g4 = mpatches.Patch(color=colorlist[3], label='Group 4')
        g5 = mpatches.Patch(color=colorlist[4], label='Group 5')
        plt.legend(handles=[g0, g01, g02, g03, g1, g2, g3, g4, g5], loc=[1.02, 0])

    # plot options
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(aspect='equal', adjustable='box')
    ax = plt.gca()

    ticks = (ax.get_xticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_xticklabels(ticks)
    ax.xaxis.set_tick_params(pad=2, length=0)

    ticks = (ax.get_yticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_yticklabels(ticks)
    ax.yaxis.set_tick_params(pad=2, length=0)
    plt.xlabel('East in km')
    plt.ylabel('North in km')
    # plt.title('Set Up of Cellular Automata')

    # create colorbar
    if target is None and year == 0 and not without_lv:
        cb_ax = fig.add_axes([0.9, 0.62, 0.02, 0.25])

        # create colorbar
        cbar = plt.colorbar(im, shrink=0.6, drawedges=False, ticks=[-1.5] + [0] + [1.5], cax=cb_ax)
        cbar.outline.set_linewidth(0)
        cbar.ax.set_ylabel('Land Value', labelpad=0)
        cbar.set_label("Land Value", fontsize=7, x=1, y=.5, rotation=90)
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.yaxis.set_tick_params(pad=1, length=0)


    if result_dir is not None:
        if target is not None:
            if name is None:
                plt.savefig(result_dir + '/_IMU2010.pdf', bbox_inches="tight")
            else:
                plt.savefig(result_dir + '/_' + name + '.pdf', bbox_inches="tight")
        if target is None and without_lv:
            plt.savefig(result_dir + '/' + str(year) + '_se_group_init.pdf', bbox_inches="tight")
        else:
            plt.savefig(result_dir + '/' + str(year) + '_se_group.pdf', bbox_inches="tight")

    plt.close(fig)
    gc.collect()

    return




def plot_ca_colors(array, result_dir = None, year = 0, pixel_size=30.0):
    '''
    Plots the colors specified by the input array, this function can be used to plot the results of the CA with the growth type as colors

    :param array: array with rgb color values
    :param result_dir: directory where figure is saved to
    :param year: index of simulation year/growth cycle for numbering in the filename
    '''
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1

    fig = plt.figure(figsize=(2.75, 2.4), dpi=900)
    # plot sleuth layers
    im = plt.imshow(array, interpolation=None)
    cbar = plt.colorbar(im, shrink=0.2, drawedges=False)
    cbar.remove()

    # plot sleuth layers
    color_spontaneous = np.array([153, 0, 153]) / 255.  # violett
    color_new_spread = np.array([248, 119, 117]) / 255.  # salmon
    color_edge = np.array([255,221,51]) / 255.  # dark yellow
    color_road = np.array([51, 153, 0]) / 255.  # green



    # create legend
    g01 = mpatches.Patch(color='black', label='Urban Seed')
    g02 = mpatches.Patch(color='lightgray', label='Protected')
    g03 = Line2D([0], [0], color='black', lw=1, label='Roads')

    g1 = mpatches.Patch(color=color_spontaneous, label='Spontaneous')
    g2 = mpatches.Patch(color=color_new_spread, label='Spreading')
    g3 = mpatches.Patch(color=color_edge, label='Edge')
    g4 = mpatches.Patch(color=color_road, label='Road')
    plt.legend(handles=[g01, g02, g03, g1, g2, g3, g4], loc=[1.02, 0])


    # plot options
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(aspect='equal', adjustable='box')
    ax = plt.gca()

    ticks = (ax.get_xticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_xticklabels(ticks)
    ax.xaxis.set_tick_params(pad=2, length=0)

    ticks = (ax.get_yticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_yticklabels(ticks)
    ax.yaxis.set_tick_params(pad=2, length=0)
    plt.xlabel('East in km')
    plt.ylabel('North in km')

    if result_dir != None:
        plt.savefig(result_dir + '/' + str(year) + '_growth_type.pdf', bbox_inches="tight")

    # plt.show()
    plt.close(fig)
    gc.collect()

    return




def plot_grow_history(n_spont, n_spread, n_edge, n_road, result_dir = None):
    '''
    Creates a graph with the number of newly urbanized cells per growth cycle split into growth type
    :param n_spont: array of grown cells for each growth cycle of the type spontaneous growth
    :param n_spread: array of grown cells for each growth cycle of the type spreading center growth
    :param n_edge: array of grown cells for each growth cycle of the type edge growth
    :param n_road: array of grown cells for each growth cycle of the type road-influenced growth
    :param result_dir: directory where figure is saved to
    '''
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1

    #fig = plt.figure(figsize=(7,5), dpi=300)
    fig = plt.figure(figsize=(2.75, 2.4), dpi=300)
    ax = plt.gca()
    ax.set_position([0, 0, 0.65, 0.75])

    # plot sleuth layers
    color_spontaneous = np.array([153, 0, 153]) / 255.  # violett
    color_new_spread = np.array([248, 119, 117]) / 255.  # salmon
    color_edge = np.array([255, 221, 51]) / 255.  # dark yellow
    color_road = np.array([51, 153, 0]) / 255.  # green


    plt.plot(np.array(range(n_spont.size)), n_spont, c=color_spontaneous)
    plt.plot(np.array(range(n_spread.size)), n_spread, c=color_new_spread)
    plt.plot(np.array(range(n_edge.size)), n_edge, c=color_edge)
    plt.plot(np.array(range(n_road.size)), n_road, c=color_road)

    n_total = n_spont + n_spread + n_edge + n_road
    plt.plot(np.array(range(n_spont.size)), n_total, c='black')


    # create legend
    g1 = mpatches.Patch(color= color_spontaneous, label='Spontaneous')
    g2 = mpatches.Patch(color= color_new_spread, label='Spreading')
    g3 = mpatches.Patch(color= color_edge, label='Edge')
    g4 = mpatches.Patch(color= color_road, label='Road')
    g5 = mpatches.Patch(color='black', label='Total')
    plt.legend(handles=[g5, g1, g2, g3, g4], loc='upper left')


    # plot options
    plt.xlabel('Growth Cycle')
    plt.ylabel('Urbanized Cells')


    if result_dir != None:
        plt.savefig(result_dir + '/' + 'Growth_types_overview.pdf', bbox_inches="tight", dpi=300)


    # plt.show()
    plt.close(fig)
    gc.collect()

    return





def create_tif_from_array(fp_tif, fp_out, year, array, NoData_value = -9999):
    '''
    Creates a geotagged tif (geotiff) from an array with the same dimensions of a reference geotiff

    :param fp_tif: geotiff as source
    :param fp_out: directory where new tif is saved to
    :param year: year which can be included in the filename
    :param array: array to be saved as tif
    :param NoData_value: no data value in the tif
    :return:
    '''
    # create empty tif with same dimensions as DEM
    print 'Creating', year, 'from array as geotif.'
    raster_in = fp_tif
    raster_out = fp_out + 'tif/' + year + '.tif'

    # open source and copy properties
    with rio.open(raster_in) as src:
        profile = src.profile.copy()
        source_tif = gdal.Open(raster_in)
        source_tif_band = source_tif.GetRasterBand(1)
        source_array = source_tif_band.ReadAsArray()

        # save destination with properties from source
        with rio.open(raster_out, 'w', **profile) as dst:
            dst.write(array, 1)
            plt.show()

    return raster_out
