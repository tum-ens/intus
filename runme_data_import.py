import numpy as np
from scipy import signal
import scipy

import rasterio as rio

import os

# --------------
from process_functions.array_import_functions import import_IMU, plot_factors


# This runfile imports the geotagged tif files and creates numpy arrays for further processing.
# The script also extracts the new settlements between the years 2000 and 2005 as well as
# 2005 and 2010.

# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)



scenarios = [0, 1]

for scenario in scenarios:

    # CHANGE SCENARIO  ---------------------------------
    do_newly_urbanized = scenario

    # --------------- SPECIFY PATH OF DIRECTORIES --------------------------------
    script_dir = os.path.dirname(__file__)
    rel_path = 'input/mexico_city/'
    abs_link_Processed = os.path.join(script_dir, rel_path)


    abs_link_Processed_tif = abs_link_Processed + 'tif/'
    abs_link_Processed_graph = abs_link_Processed + 'graph/'
    abs_link_Processed_array = abs_link_Processed + 'array/'



    # -----------------------------------------------
    labels_se_levels = ['IMU2000_new.tif', 'IMU2005_new.tif', 'IMU2010_new.tif',
                        'IMU2000_new_level.tif', 'IMU2005_new_level.tif', 'IMU2010_new_level.tif']


    labels_features = ['Dem_SRTM_30.tif', 'Wells_15_18_dist.tif', 'Residual_Water_15_18_dist.tif', 'Water1997_dist.tif',
                       'Highways_1985_dist.tif', 'Urban1997_dist.tif']



    # -----------------------------------------------
    do_plot_overview = 1

    do_filter_all = 0

    years = [2000, 2005, 2010]





    # ################################################################################
    # -----------------------  IMPORT TIF AS ARRAYS ----------------------------------

    print('Importing tif files as arrays...')

    # -----------------------------------------------------------------------

    n_factors = labels_features.__len__()
    n_se_levels = labels_se_levels.__len__()


    # A) GET SOCIO-ECONOMIC LEVELS
    # -----------------------------------------------------------------------
    IMU, n_row, n_col, n_elements = import_IMU(abs_link_Processed_tif, labels_se_levels)




    # B1) GET INFLUENCING FACTORS
    # -----------------------------------------------------------------------
    factors_mat = np.zeros([n_factors+3, n_row, n_col])
    index = -1

    for i in range(0, n_factors):
        raster_fn = abs_link_Processed_tif + labels_features[i]
        try:
            tif = rio.open(raster_fn)
            array_from_tif = tif.read(1)
            array_from_tif = array_from_tif[1:-1, 1:-1]  # crop outermost row and column to remove artifacts (eg slope calculation)
            array_from_tif[array_from_tif == -9999] = -1


            index += 1
            factors_mat[index, :, :] = array_from_tif
        except:
            print 'WARNING:', labels_features[i], 'not found, leaving array empty with zeros.'
            index += 1



    # average neighboring IMU-classes from 2000 as feature '+1'
    to_filter = np.copy(IMU[0, :, :])
    indices = np.where( np.logical_and( to_filter > 0, to_filter < 255))
    values = to_filter[indices]
    grid_x, grid_y = np.mgrid[0:to_filter.shape[0], 0:to_filter.shape[1]]

    closest_group = scipy.interpolate.griddata(indices, values, (grid_x, grid_y), method='nearest')

    factors_mat[-3, :, :] = closest_group
    labels_features = np.append(labels_features, ['closest_group'])

    n_factors += 1      # for closest group as new factor

    print '... ' + str(n_factors) + ' files imported'


    # x and y location as feature '+2' and '+3'
    factors_mat[-2:, :, :] = np.indices((n_row, n_col))





    # B2) PLOT INFLUENCING FACTORS
    # -----------------------------------------------------------------------
    if do_plot_overview:
        plot_factors(factors_mat, labels_features, abs_link_Processed_graph)







    # ################################################################################
    # ------------------  CLEANING DATA (FLATTEN, REMOVE INVALID) ------------------
    print 'Cleaning data...'

    # -----------------------------------------------------------------------

    factors = np.zeros([n_elements, n_factors+2])
    se_levels = np.zeros([n_elements, n_se_levels])


    # A) FLATTEN
    # -----------------------------------------------------------------------

    # factors and labels
    for i in range(0, n_factors+2):
        if do_filter_all and i < n_factors:
            factors_mat[i, :, :] = signal.medfilt2d(factors_mat[i, :, :], kernel_size=9)

        factors[:, i] = factors_mat[i, :, :].flatten('F')
        if i < n_factors:
            print labels_features[i], ': ', factors[:, i].min(), factors[:, i].max()
        elif i == n_factors:
            print 'rows', ': ', factors[:, i].min(), factors[:, i].max()
        elif i == n_factors+1:
            print 'columns', ': ', factors[:, i].min(), factors[:, i].max()


    # se-levels
    for i in range(0, n_se_levels):
        se_levels[:, i] = IMU[i, :].flatten('F')
        print labels_se_levels[i], ': ', se_levels[:, i].min(), se_levels[:, i].max()


    print '...importing, cropping and flattening finished...'





    # B) REMOVE INVALID (= NO DATA ABOUT SE-LEVEL) FOR A CERTAIN YEAR
    # -----------------------------------------------------------------------
    for year in years:
        print 'Processing year', year
        if year == 2000:
            se_raw = se_levels[:, 0]
            se_level_raw = se_levels[:, 3]

        elif year == 2005:
            se_raw = se_levels[:, 1]
            se_level_raw = se_levels[:, 4]

        elif year == 2010:
            se_raw = se_levels[:, 2]
            se_level_raw = se_levels[:, 5]

        print 'Remove invalid...'

        if do_newly_urbanized:
            if year == 2000:
                continue

            print 'Remove already existing...'

            if year == 2005:
                se_raw_before = se_levels[:, 0]
                se_level_raw_before = se_levels[:, 3]
            elif year == 2010:
                se_raw_before = se_levels[:, 1]
                se_level_raw_before = se_levels[:, 4]

            # remove cells that were urbanized in previous year
            i_existing_before = np.squeeze(np.where(se_level_raw_before != 255))

            se_level_urbanized = np.delete(se_level_raw, i_existing_before)
            se_group_urbanized = np.delete(se_raw, i_existing_before)

            clean_factors_urbanized = np.zeros([se_group_urbanized.size, n_factors+2])
            for i in range(0, n_factors+2):
                clean_factors_urbanized[:, i] = np.delete(factors[:, i], i_existing_before)

            # remove cells which were not urbanized in this step
            i_invalid_urbanized = np.squeeze(np.concatenate((np.where(se_level_urbanized == 255), np.where(se_level_urbanized == 0)), axis=1))

            se_level = np.delete(se_level_urbanized, i_invalid_urbanized)
            se_group = np.delete(se_group_urbanized, i_invalid_urbanized)

            clean_factors = np.zeros([se_group.size, n_factors+2])
            for i in range(0, n_factors+2):
                clean_factors[:, i] = np.delete(clean_factors_urbanized[:, i], i_invalid_urbanized)

        else:
            i_invalid = np.squeeze(np.concatenate((np.where(se_level_raw == 255), np.where(se_level_raw == 0)), axis=1))

            se_level = np.delete(se_level_raw, i_invalid)
            se_group = np.delete(se_raw, i_invalid)

            clean_factors = np.zeros([n_elements - i_invalid.size, n_factors+2])

            for i in range(0, n_factors+2):
                clean_factors[:, i] = np.delete(factors[:, i], i_invalid)


        mean_se = se_level.mean()
        mean_se_group = se_group.mean()

        print '... preprocessing finished.'



        # save cleaned factors and respective se-group and se-level for the year
        if do_newly_urbanized == 0:
            indicator = str(year) + '_all_'
        else:
            indicator = str(year) + '_new_'

        np.save(abs_link_Processed_array + indicator + 'clean_factors', clean_factors)
        np.save(abs_link_Processed_array + indicator + 'se_group', se_group)
        np.save(abs_link_Processed_array + indicator + 'se_level', se_level)

        np.save(abs_link_Processed_array + indicator + 'labels_factors', labels_features)

        # save whole feature matrices
        if do_newly_urbanized == 0 and year == 2000:
            np.save(abs_link_Processed_array + indicator + 'factors_mat', factors_mat)
            np.save(abs_link_Processed_array + indicator + 'dimensions', [n_elements, n_factors+2])
            np.save(abs_link_Processed_array + indicator + 'mat_dim', np.array(factors_mat.shape[1:]))


