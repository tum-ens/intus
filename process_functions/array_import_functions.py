import matplotlib.pyplot as plt
import copy
import rasterio as rio
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker

# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)


def import_IMU(abs_link_Processed_tif, labels_se_levels):
    '''
    Imports the tif files with data on IMU (both groups and level) for the years and merges them into one numpy array

    :param abs_link_Processed_tif: link to the tif with the IMU
    :param labels_se_levels: labels specifying the filenames
    :return:
        IMU: array with the IMUs for the years
        n_row: number of rows
        n_col: number of columns
        n_elements: number of elements for one year
    '''
    tif = rio.open(abs_link_Processed_tif + labels_se_levels[0])
    IMU2000_group = tif.read(1)

    n_row, n_col = IMU2000_group.shape
    n_row = n_row - 2
    n_col = n_col - 2

    IMU = np.zeros([6, n_row, n_col])

    # 2000 se-group
    IMU2000_group = IMU2000_group[1:-1, 1:-1]
    n_elements = IMU2000_group.size
    IMU[0, :, :] = IMU2000_group

    # 2000 level
    tif = rio.open(abs_link_Processed_tif + labels_se_levels[3])
    tif = tif.read(1)
    IMU[3, :, :] = tif[1:-1, 1:-1]

    # 2005 se-group
    tif = rio.open(abs_link_Processed_tif + labels_se_levels[1])
    tif = tif.read(1)
    IMU[1, :, :] = tif[1:-1, 1:-1]

    # 2005 level
    tif = rio.open(abs_link_Processed_tif + labels_se_levels[4])
    tif = tif.read(1)
    IMU[4, :, :] = tif[1:-1, 1:-1]

    # 2010 se-group
    tif = rio.open(abs_link_Processed_tif + labels_se_levels[2])
    tif = tif.read(1)
    IMU[2, :, :] = tif[1:-1, 1:-1]

    # 2010 level
    tif = rio.open(abs_link_Processed_tif + labels_se_levels[5])
    tif = tif.read(1)
    IMU[5, :, :] = tif[1:-1, 1:-1]

    return IMU, n_row, n_col, n_elements




def plot_factors(factors_arr, labels_factors, abs_link_Processed_graph):
    """
    Plot the environmental characteristics with suitable colormaps

    :param factors_arr: array consisting of 2d matrices for each feature
    :param labels_factors: names of the features
    :param abs_link_Processed_graph: location to save the figure to
    """
    n_factors = labels_factors.__len__()

    # shorten names for features
    feature_list_replacement = {
            'Dem_SRTM_30.tif': 'DEM',
            'Wells_15_18_dist.tif': 'Wells dist',
            'Residual_Water_15_18_dist.tif': 'Residual dist',
            'Water1997_dist.tif': 'Water dist',
            'Highways_1985_dist.tif': 'Highways dist',
            'Urban1997_dist.tif': 'Urban97 dist',
            'closest_group': 'Closest group',
    }
    sns.set(font_scale=0.65)
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(5.5, 5.5),
                            subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.05, wspace=0.1)

    for ax, infl_factor in zip(axs.flat, range(0, n_factors)):
        if infl_factor < 6:
            im = ax.imshow(factors_arr[infl_factor, :, :], interpolation=None, cmap='gist_earth')
        elif infl_factor == 6:
            im = ax.imshow(factors_arr[infl_factor, :, :], interpolation=None, cmap='jet')

        # create colorbar
        cbbox = inset_axes(ax, '25%', '90%', loc=7)
        [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
        cbbox.tick_params(axis='both', left='False', top='False', right='False', bottom='False', labelleft='False', labeltop='False',
                          labelright='False',
                          labelbottom='False')
        cbbox.set_facecolor([1, 1, 1, 0.7])
        cbaxes = inset_axes(cbbox, '35%', '95%', loc=6)
        cb = fig.colorbar(im, cax=cbaxes)  # make colorbar

        # reduce ticks
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.locator = tick_locator
        cb.update_ticks()


        title = feature_list_replacement[labels_factors[infl_factor]]
        ax.set_title(title)

    plt.show()
    fig.savefig(abs_link_Processed_graph + 'factors.pdf', bbox_inches="tight")


