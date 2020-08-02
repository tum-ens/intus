import numpy as np
import matplotlib.pyplot as plt

from prettytable import PrettyTable
import scipy.ndimage

import cv2
import matplotlib.patches as mpatches   # for manual legend entries
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)



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





def create_circular_mask(h, w, center, radius):
    '''
    Create a circular mask of predefined size

    :param h: height if the center is not defined for determining the middle of the image
    :param w: width if the center is not defined for determining the middle of the image
    :param center: predefined center of the circle
    :param radius: radius of the circle
    :return:
    '''

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask




def calculate_metrices(arr_urban_init, arr_urbanized, do_downsampling, gc, result_dir=None, pixel_size=90.0):
    '''
    Calculate the spatial metrices from the Atlas of Urban Extension

    :param arr_urban_init: array of the init urban area
    :param arr_urbanized: array of the urbanized area
    :param do_downsampling: ff downsampling is undertaken (from 30*30 to 60*60 meters cell size)
    :param gc: growth cycle
    :param result_dir: directory for saving the figure
    :return: spatial metrices (more info in the Atlas of Urban Extension)
    '''
    arr = np.maximum(arr_urban_init, arr_urbanized)
    arr[arr>0] = 1

    if do_downsampling == 60:
        pixel_area = 60*60
        r = round(584 / 60)
    elif do_downsampling == 300:
        pixel_area = 300*300
        r = round(584 / 300)
    else:
        pixel_area = pixel_size*pixel_size
        r = round(584 / pixel_size)

    # Dummy set for testing
    # x = np.arange(0, 600)
    # y = np.arange(0, 500)
    # arr = np.zeros((y.size, x.size))
    #
    # cx = 400.
    # cy = 200.
    # r = 40.
    # mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    # arr[mask] = 1.
    #
    # cx = 400.
    # cy = 200.
    # r = 20.
    # mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    # arr[mask] = 0.
    #
    # cx = 350.
    # cy = 200.
    # r = 15.
    # mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    # arr[mask] = 1.
    #
    # cx = 320.
    # cy = 200.
    # r = 3.
    # mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    # arr[mask] = 1.

    ratio_built_up = np.zeros(arr.shape)
    type_built_up = np.zeros(arr.shape, 'int')
    potential_open_space_arr = np.zeros(arr.shape)
    i = 0

    h, w = arr.shape[:2]


    ## ------ ARRAY PREPARATIONS -------
    urban = np.argwhere(arr)



    for u in urban:
        # check how high the ratio of urbanized pixels is in a 1km2 circle around the built-up pixel
        center = u  # center coordinates need x,y order
        mask = create_circular_mask(h, w, center=center, radius=r)

        n_pixels = arr[mask].size

        # determine built-up ratio
        cur_ratio_built_up = sum(arr[mask]) / float(n_pixels)
        ratio_built_up[u[0],u[1]] = cur_ratio_built_up

        # assign type according to built-up ratio
        if ratio_built_up[u[0],u[1]] < 0.25:
            type = 1  # rural
        elif ratio_built_up[u[0],u[1]] < 0.5:
            type = 2  # suburban
        else:
            type = 3  # urban
        type_built_up[u[0],u[1]] = type

        i += 1
        if i%1000 == 0:
            print 'calculated urban metrices, urban cell', i, '/', len(urban)



    # ----- create contours with OpenCV
    ret, thresh = cv2.threshold(arr, 0.5, 255, 0)
    thresh = cv2.convertScaleAbs(thresh)

    if cv2.__version__ == '4.0.0':
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
    else:
        _, contours, hierarchy = cv2.findContours(thresh, 1, 2)     # support older cv2 version as on the virtual machine

    val_max_contour = max([len(c) for c in contours])
    idx_max_contour = 0

    # get biggest contour
    for i in range(0, len(contours)):
        if len(contours[i]) == val_max_contour:
            idx_max_contour = i
            break
    cnt = contours[idx_max_contour]

    # get area of the biggest contour
    area = cv2.contourArea(cnt)
    print 'area', area*pixel_area/1e6
    img = np.zeros((arr.shape[0], arr.shape[1]))
    filled_contour = cv2.fillPoly(img, pts=[cnt], color=(1, 1, 1))



    # approximate contour
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    img2 = np.zeros((arr.shape[0], arr.shape[1]))
    poly2 = cv2.fillPoly(img2, pts=[approx], color=(1, 1, 1))


    # remove built-up to only leave urban open space
    built_up = np.copy(type_built_up)
    built_up[built_up == 1] = 0         # exclude rural
    built_up[built_up > 1] = 1

    open_space_arr = np.copy(filled_contour)
    open_space_arr[built_up > 0] = 0

    # merge built-up cells and urban open space
    urban_ext_arr = np.maximum(built_up, open_space_arr)

    # get built-up cells within urban extension
    built_up_within_arr = np.multiply(built_up, urban_ext_arr)


    # get avg share of open space pixels within walking distance of built-up pixel (for fragmentation, openness)
    share_open_space = []
    share_open_space_arr = np.zeros(arr.shape)


    built_up_in = np.argwhere(built_up_within_arr)
    i = 0

    for b in built_up_in:
        center = [b[0], b[1]]  # center coordinates need x,y order
        mask = create_circular_mask(h, w, center=center, radius=r)

        n_pixels = built_up_within_arr[mask].size
        cur_share_open_space = float(sum(open_space_arr[mask])) / n_pixels

        share_open_space_arr[b[0], b[1]] = cur_share_open_space

        i += 1
        if i%1000 == 0:
            print 'calculated urban metrices, urban cell', i, '/', len(built_up_in)



    ## -------- CALCULATE METRICES

    # built-up cells
    built_up_total = float((type_built_up > 0).sum())
    built_up_urban = float((type_built_up == 3).sum())
    built_up_suburban = float((type_built_up == 2).sum())
    built_up_rural = float((type_built_up == 1).sum())


    # open space
    urbanized_open_space = float((open_space_arr > 0).sum())
    urban_ext = float((urban_ext_arr > 0).sum())


    # fragmentation
    built_up_within = float((built_up_within_arr > 0).sum())

    saturation = built_up_within / urban_ext

    openness = np.mean(share_open_space_arr[np.nonzero(share_open_space_arr)])

    # openness = np.mean(share_open_space)

    # compactness
    urban_area = pixel_area * urban_ext

    if do_downsampling == 60:
        cell_length = 60
    if do_downsampling == 300:
        cell_length = 300
    else:
        cell_length = pixel_size

    equal_radius = round(np.sqrt(urban_area/np.pi)/cell_length)
    equal_center = scipy.ndimage.measurements.center_of_mass(urban_ext_arr)
    mask = create_circular_mask(h,w, center=equal_center, radius=equal_radius)
    mask_arr = mask.astype('int')

    cells_inside_eac = []
    cells_inside_eac_arr = np.empty((0, 2), int)

    cells_urban_extent = []
    cells_urban_extent_arr = np.empty((0, 2), int)


    built_up_in = np.argwhere(built_up_within_arr)
    i = 0

    for b in built_up_in:
        if built_up_within_arr[b[0],b[1]] > 0:
            # separate urban cells which are in and out of the equal area circle
            if mask[b[0],b[1]] == True:
                cells_inside_eac_arr = np.concatenate((cells_inside_eac_arr, [[b[0],b[1]]]), axis=0)

            cells_urban_extent_arr = np.concatenate((cells_urban_extent_arr, [[b[0], b[1]]]), axis=0)

        i += 1
        if i%1000 == 0:
            print 'determined in or out equal area circle urban cell', i, '/', len(built_up_in)


    cells_inside_eac = np.asarray(cells_inside_eac)
    cells_urban_extent = np.asarray(cells_urban_extent)

    cells_inside_eac = cells_inside_eac_arr
    cells_urban_extent = cells_urban_extent_arr

    # calculate all the distances from the points in the equal area circle to its center
    dist_in     = np.linalg.norm(cells_inside_eac   - np.asarray(equal_center), axis=1)
    dist_urb_ex = np.linalg.norm(cells_urban_extent - np.asarray(equal_center), axis=1)

    avg_dist_in = np.mean(dist_in)
    avg_dist_urb_ex = np.mean(dist_urb_ex)

    proximity = avg_dist_in / avg_dist_urb_ex


    # Cohesion
    # calculate all the distances between the points in the equal area circle and the urban extent
    # pdist_in = np.mean(scipy.spatial.distance_matrix(cells_inside_eac,cells_inside_eac))
    # pdist_urb_ex = np.mean(scipy.spatial.distance_matrix(cells_urban_extent,cells_urban_extent))
    # cohesion = pdist_in/pdist_urb_ex

    cohesion = 0


    ## -------- DISPLAY VALUES -------
    built_up_total_area = built_up_total*pixel_area/1e6
    built_up_urban_area = built_up_urban*pixel_area/1e6
    built_up_suburban_area =  built_up_suburban*pixel_area/1e6
    built_up_rural_area = built_up_rural*pixel_area/1e6

    urbanized_open_space_area = urbanized_open_space*pixel_area/1e6
    urban_ext_area = urban_ext*pixel_area/1e6



    t = PrettyTable(['Metric', 'Pixels/Index', 'Area [sqkm]'])

    t.add_row(['Built_up_Total', built_up_total, built_up_total_area])
    t.add_row(['Built_up_Urban', built_up_urban, built_up_urban_area])
    t.add_row(['Built_up_Suburban', built_up_suburban, built_up_suburban_area])
    t.add_row(['Built_up_Rural', built_up_rural, built_up_rural_area])

    t.add_row(['Urbanized_Open_Space', urbanized_open_space, urbanized_open_space_area])
    t.add_row(['Urban_Extension', urban_ext, urban_ext_area])

    t.add_row(['Saturation', '{0:.3f}'.format(saturation), '-'])
    t.add_row(['Openness', '{0:.3f}'.format(openness), '-'])

    t.add_row(['Proximity', '{0:.3f}'.format(proximity), '-'])
    t.add_row(['Cohesion', '{0:.3f}'.format(cohesion), '-'])

    print t



    ## ---------- PLOT ----------
    fig = plt.figure(figsize=(5.5, 4.5))
    cm = plt.cm.get_cmap('viridis_r')

    # Urbanized Cell
    ax1 = plt.subplot(231)
    plt.axis('off')
    arr = arr.astype('float')
    arr[arr == 0] = np.nan
    plt.imshow(arr, cmap=cm.reversed())
    plt.title('Urbanized Cells')
    plt.gca().set_aspect(aspect='equal', adjustable='box')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    for axis in ['top', 'bottom', 'left', 'right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])

    # Ratio Built-Up
    plt.subplot(232, sharex=ax1, sharey=ax1)
    plt.axis('off')
    ratio_built_up[ratio_built_up == 0] = np.nan
    plt.imshow(ratio_built_up, cmap=cm)
    plt.colorbar(shrink=0.6)
    plt.title('Ratio Built-Up')
    plt.gca().set_aspect(aspect='equal', adjustable='box')

    # Type Built-Up
    ax3 = plt.subplot(233, sharex=ax1, sharey=ax1)
    plt.axis('off')
    type_built_up = type_built_up.astype('float')
    type_built_up[type_built_up == 0] = np.nan
    plt.imshow(type_built_up, cmap=cm)
    # plt.colorbar()
    plt.title('Type Built-Up')
    plt.gca().set_aspect(aspect='equal', adjustable='box')


    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    # cax.set_axis_bgcolor('none')
    for axis in ['top', 'bottom', 'left', 'right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])

    # create legend
    g1 = mpatches.Patch(color=cm(0.33), label='Rural')
    g2 = mpatches.Patch(color=cm(0.66), label='Suburban')
    g3 = mpatches.Patch(color=cm(0.99), label='Urban')
    plt.legend(handles=[g1, g2, g3], loc=[1.02, 0])

    # Open Space
    ax4 = plt.subplot(234, sharex=ax1, sharey=ax1)
    plt.axis('off')
    open_space_arr = open_space_arr.astype('float')
    open_space_arr[open_space_arr == 0] = np.nan
    plt.imshow(open_space_arr, cmap=cm.reversed())
    plt.title('Open Space')
    plt.gca().set_aspect(aspect='equal', adjustable='box')

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    # cax.set_axis_bgcolor('none')
    for axis in ['top', 'bottom', 'left', 'right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])

    # Urban Extension Contour Original
    ax5 = plt.subplot(235, sharex=ax1, sharey=ax1)
    plt.axis('off')
    filled_contour = filled_contour.astype('float')
    filled_contour[filled_contour == 0] = np.nan
    plt.imshow(filled_contour, cmap=cm.reversed())
    plt.title('Urban Extension')
    plt.gca().set_aspect(aspect='equal', adjustable='box')

    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    # cax.set_axis_bgcolor('none')
    for axis in ['top', 'bottom', 'left', 'right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])


    # Share Open Space
    plt.subplot(236, sharex=ax1, sharey=ax1)
    plt.axis('off')
    share_open_space_arr[share_open_space_arr == 0] = np.nan
    plt.imshow(share_open_space_arr, cmap=cm)
    plt.colorbar(shrink=0.6)
    plt.title('Share Open Space')
    plt.gca().set_aspect(aspect='equal', adjustable='box')
    if result_dir != None:
        figname = '{}/{}.jpg'.format(result_dir, 'SpatialMetrices')
        plt.savefig(figname, dpi=600, bbox_inches="tight")
        plt.savefig(result_dir + '/SpatialMetrices.pdf', bbox_inches="tight", dpi=600)
        data = t.get_string()
        with open(result_dir + '/parameters.txt', 'a') as f:
            f.write('\n\n')
            f.write(str(gc))
            f.write('\n')
            f.write(data)
    # plt.show()
    plt.close(fig)
    # gc.collect()


    return built_up_total_area, built_up_urban_area, built_up_suburban_area, built_up_rural_area, \
           urbanized_open_space_area, urban_ext_area,\
           saturation, openness, proximity, cohesion, built_up_total


def calculate_correct(se_level, IMU, urban, result_dir):
    '''
    Calculates the share of correctly predicted cells (location) and correctly predicted socioeconomic level for the cells that are in a
    location with data on the AGEBs in the data of the IMU

    :param se_level: socioeconomic level of the predicted cells
    :param IMU: reference socioeconomic level from the IMU
    :param urban: urban_seed for calculating number of urban cells
    :param result_dir: directory for saving the calculated values in a textfile
    :return:
        percent_correct_all: share of predicted cells in a correct location
        percent_correct_only_same_location: share of predicted cells with a correct socioeconomic level
    '''
    urban_seed = np.copy(urban)
    n_urban_seed = np.count_nonzero(urban_seed)

    trueIMU = np.copy(IMU)
    n_urbanized_IMU = np.count_nonzero(trueIMU)

    trueIMU[trueIMU == 0] = 10

    # calculate percentage of all correct predicted locations and se-level
    n_identical  = np.count_nonzero(se_level == trueIMU)
    n_urbanized_sim = np.count_nonzero(se_level)
    percent_correct_all = n_identical/float(n_urbanized_sim)

    # calculate percentage of all correct predicted se-levels for the ones where IMU settlement exists
    se_level_same_location = np.zeros_like(se_level)
    se_level_same_location[(IMU > 0) & (se_level > 0)] = 1
    n_urbanized_same_location = np.count_nonzero(se_level_same_location)

    percent_correct_only_same_location = n_identical/float(n_urbanized_same_location)

    percent_correct_location = n_urbanized_same_location/float(n_urbanized_sim)

    print 'N urban IMU:', n_urbanized_IMU
    print 'N urban sim:', n_urbanized_sim + n_urban_seed
    print 'Percentage of correct location:', percent_correct_location
    print 'Percentage of correct se:', percent_correct_all
    print 'Percentage of correct se only corresponding cells:', percent_correct_only_same_location

    with open(result_dir + '/parameters.txt', 'a') as f:
        f.write('\n\n N urban IMU: ')
        f.write(str(n_urbanized_IMU))
        f.write('\n\n N urban sim: ')
        f.write(str(n_urbanized_sim + n_urban_seed))
        f.write('\n\n Percentage of correct location: ')
        f.write(str(percent_correct_location))
        f.write('\n\n Percentage of correct se: ')
        f.write(str(percent_correct_all))
        f.write('\n\n Percentage of correct se for corresponding cells: ')
        f.write(str(percent_correct_only_same_location))

    return percent_correct_all, percent_correct_only_same_location, n_urbanized_IMU, n_urbanized_sim

