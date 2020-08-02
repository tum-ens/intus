import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


import matplotlib.pyplot as plt
import seaborn as sns

from prettytable import PrettyTable

import config
import sys
import os


# --------------
from process_functions.classification_functions import plot_2d_belief, plot_projected, plot_projected_cont, \
                                                        plot_loading_score, print_RF_stats

from process_functions.regression_functions import lmu_to_class, assess_regression_accuracy, \
                                                    plot_misclassified_reg, rfr_param_selection, rfr_param_selection_test

from process_functions.preprocessing_functions import preprocess_data_2d_predict, preprocessing_data

from process_functions.postprocessing_functions import plot_rasterize_2d_maps_reg



# This runfile takes the arrays from the runme_data_import.py and incorporates all the machine learning parts such as training and assessing
# the RFR and PLS plus predicting the land value map and saving it as an geotagged tif file for the cellular automata.


# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)



SMALLER_SIZE = 7
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 11
plt.rc('font', size=SMALLER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# SET LOCATION  ---------------------------------
location = 'mexico_city'


# CHANGE SCENARIO  ---------------------------------
do_only_land_value_map = 0      # create land value map and then stop
do_satellite = 0               # analyze satellite images

do_scenario_05_10_00 = 0
do_scenario_00_05_10 = 0
do_scenario_05_10 = 1


pixel_size = 300.0

# Specify steps for subsample in option 'sparse_downsampling'
sampling_steps = 1


# --------------- SPECIFY PATH OF DIRECTORIES --------------------------------
script_dir = os.path.dirname(__file__)
rel_path = 'input/mexico_city/'
abs_link_Processed = os.path.join(script_dir, rel_path)

abs_link_Processed_tif = abs_link_Processed + 'tif/'
abs_link_Processed_graph = abs_link_Processed + 'graph/'
abs_link_Processed_array = abs_link_Processed + 'array/'



# -----------------------------------------------
config.global_path = abs_link_Processed + 'land_value/'
# -----------------------------------------------


# --------
do_PLS = 1
do_RFR = 1
# --------



print_class_report = 0
do_plot_tree = 0
do_plot_2d = 1
do_plot_features_accuracy = 0

do_polynomials = 0




# ------- SAMPLING PARAMETERS ---------
# Specify parameters for sampling type ('none', 'uniform', 'max_samples', 'sparse_downsampling')
sampling = 'sparse_downsampling'                        # sampling for training the model
sampling_prediction = 'none'                            # sampling for predicton from the trained model


# specify maximum of samples that limit the samples in 'uniform' and 'max_samples'
max_samples = 2000
do_train_with_all = 0






# ------ PARAMETERS ------
chosen_groups = [1, 2, 3, 4, 5]



labels_factors = ['Dem_SRTM_30.tif'
                  'Wells_15_18_dist.tif', 'Residual_Water_15_18_dist.tif', 'Water1997_dist.tif',
                  'Highways_1985_dist.tif',
                  'Urban1997_dist.tif'
                  ]

chosen_factors = np.array([0, 1, 2, 3, 4, 5, 6])


# ------------
shorten = 0


scaler = preprocessing.StandardScaler()                # scale to unit variance

sns.set(font_scale= .65)
sns.set_style("dark")
plt.style.use('fast')   # https://matplotlib.org/tutorials/introductory/customizing.html




# ----- Initializations ----

sys.stdout = open(config.global_path + 'console_out.txt', 'w')

print 'Start to process', location, 'with sampling methods', sampling, sampling_prediction, '\n'


pls = []
rfr = []


X_train = []
y_train = []
y_train_l = []
X_test = []

if do_scenario_05_10_00 or do_scenario_05_10:
    order = [3, 1, 2]
elif do_scenario_00_05_10:
    order = [1, 2, 3]


for i_year in range(1,4):

    config.global_figure_count = 0

    if i_year == order[0]:
        year = [2000]
        config.global_year = str(i_year) + '_' + str(2000)
        scenario = '_all_'
        if do_scenario_05_10:
            break
    elif i_year == order[1]:
        year = [2005]
        config.global_year = str(i_year) + '_' + str(2005)
        scenario = '_new_'
    elif i_year == order[2]:
        year = [2010]
        config.global_year = str(i_year) + '_' + str(2010)
        scenario = '_new_'

    # delete already existing in 2000 when they are used as training labels (last feature in matrix)
    if i_year == 1 and scenario == '_all_':
        chosen_factors = np.delete(chosen_factors, chosen_factors.shape[0] - 1)

    if do_satellite:
        year = [2010]
        config.global_year = 'satellite_prediction' + '_' + str(2010)
        scenario = '_all_'
        chosen_factors = None
        if i_year == 2:
            break

    if i_year == 2 or i_year == 3:
        # allow different sampling for years with prediction
        sampling = sampling_prediction




    #   ------------------------------------------------------------
    #   #############  PREPROCESSING  #############
    #   ------------------------------------------------------------
    fig = plt.figure()
    fig.suptitle('PREPROCESSING', fontsize=25, fontweight='bold', y=0.6)
    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_PREPROCESSING' + '.pdf')
    # plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_PREPROCESSING' + '.pgf')
    config.global_figure_count += 1
    plt.show()


    indicator = str(year[0]) + scenario
    abs_link_data = abs_link_Processed_array + indicator

    X, X_idx, y, y_l, feature_list, n_chosen_factors, n_groups, limits, scaler = preprocessing_data(abs_link_data,
                                                              chosen_factors, chosen_groups, max_samples,
                                                              scaler, do_polynomials, sampling, sampling_steps, i_year)

    print '... DATA PREPARATION FOR ML FINISHED.'






    #   ------------------------------------------------------------
    #   ###############  PREPARE DATA (SPLIT, TRANSFORM) ###########
    #   ------------------------------------------------------------
    if i_year == 1:
        # randomly split into training and test data and set up the model
        X_train_temp, X_test_temp, X_train_idx_temp, X_test_idx_temp, y_train_temp, y_test_temp, y_train_l_temp, y_test_l_temp = \
            train_test_split(X, X_idx, y, y_l, test_size=0.2, random_state=None)

        X_train = np.copy(X_train_temp)
        X_train_idx = np.copy(X_train_idx_temp)

        y_train = np.copy(y_train_temp)
        y_train_l = np.copy(y_train_l_temp)

        X_test = np.copy(X_test_temp)
        X_test_idx = np.copy(X_test_idx_temp)

        y_test = np.copy(y_test_temp)
        y_test_l = np.copy(y_test_l_temp)

    elif i_year == 2 or i_year == 3:
        X_test = np.copy(X)
        X_test_idx = np.copy(X_idx)

        y_test = np.copy(y)
        y_test_l = np.copy(y_l)



    #   ------------------------------------------------------------
    #   CREATE Data for VALUE MAP FROM MODEL BY EVALUATING THE WHOLE GRID
    #   ------------------------------------------------------------
    if i_year == 1:
        if do_satellite:
            factors_mat = np.load(abs_link_Processed_array + '2010_all_satellite_mat.npy')
        else:
            factors_mat = np.load(abs_link_Processed_array + '2000_all_' + 'factors_mat.npy')

        X_2d, feature_list_2d, n_chosen_factors_2d = preprocess_data_2d_predict(abs_link_data, abs_link_Processed_array, chosen_factors,
                                                                       scaler, do_polynomials, do_satellite)


    fig = plt.figure()
    fig.suptitle('REGRESSION ' + str(year), fontsize=25, fontweight='bold', y=0.6)
    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_REGRESSION' + '.pdf')
    config.global_figure_count += 1
    plt.show()


    #   ------------------------------------------------------------
    #   #############  PARTIAL LEAST SQUARES REGRESSION  ###########
    #   ------------------------------------------------------------
    if do_PLS:
        print '\n\n\n\n#######################################################'
        print year, '------ PLS on the data... --- \n'
        name = 'PLS'

        if i_year == 1:
            # --------------------------------- TRAIN ----------------------------------------
            pls = PLS(n_components=2, scale=True)
            pls = pls.fit(X_train, y_train_l)

            print feature_list
            print 'PLS x_loadings:\n', pls.x_loadings_

            # test the model with cross validation
            scores = cross_val_score(pls, X, y_l, cv=5, scoring='r2')
            print("R2 accuracy PLS cross-valid: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            print scores

        # --------------------------------- TEST ----------------------------------------

        acc_PLS, prec_PLS = assess_regression_accuracy(pls, y_test_l, X_test, name, print_class_report, n_groups, limits)


        # ----------------------- FURTHER ACCURACY ASSESSMENT ---------------------------
        # Create land value map
        if i_year == 1:
            lv_l_arr = np.reshape(pls.predict(X_2d), factors_mat.shape[1:3], order='F')
            plot_rasterize_2d_maps_reg(abs_link_Processed_tif, lv_l_arr, do_satellite, 'PLS', limits, pixel_size)

        # plot location of misclassified/mispredicted
        plot_misclassified_reg(abs_link_Processed, X_test_idx, y_test_l, np.transpose(pls.predict(X_test)).flatten(), name, limits,
                               n_groups, do_satellite, pixel_size)

        if X.shape[1] == 2 and do_plot_2d:
            plot_2d_belief(X, y, X_train, y_train, X_test, y_test, pls, name)

        # Score and loading plots
        X_pls = pls.transform(X)
        predictions = pls.predict(X_test).flatten()
        predictions_class = lmu_to_class(pls.predict(X_test).flatten(), limits)


        plot_projected(X_pls, y, name)
        plot_projected(pls.transform(X_test), predictions_class, name + ' test predict')
        plot_projected_cont(X_pls, y_l, name)

        plot_loading_score(X_pls, np.transpose(pls.x_loadings_), feature_list, y)

        print '--- ... PLS finished. ---'





    #   ------------------------------------------------------------
    #   #############  RANDOM FOREST REGRESSION  #############
    #   ------------------------------------------------------------

    if do_RFR:
        print '\n\n\n\n#######################################################'
        print year, '------ Random Forest Regression on the data... --- \n'
        name = 'RFR'

        if i_year == 1:
            # --------------------------------- TRAIN ----------------------------------------
            # hyperparameter optimization through simple grid search
            cv_score = rfr_param_selection_test(X_train, y_train_l, X_test, y_test_l, limits)

            params = rfr_param_selection(X_train, y_train_l, 5)
            n_trees = params['n_estimators']
            depth_max = params['max_depth']

            # Instantiate model
            rfr = RandomForestRegressor(n_jobs=1, n_estimators=n_trees, max_depth=depth_max, oob_score=True)
            rfr.fit(X_train, y_train_l)

            print_RF_stats(rfr, X, feature_list, name)

            # cross-validation
            scores = cross_val_score(rfr, X, y_l, cv=5, scoring='r2')
            print("R2 accuracy RFR cross-valid: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            print scores
        else:
            cv_score = rfr_param_selection_test(X_train, y_train_l, X_test, y_test_l, limits)

        # --------------------------------- TEST ----------------------------------------
        acc_RFR, prec_RFR = assess_regression_accuracy(rfr, y_test_l, X_test, name, print_class_report, n_groups, limits)


        # ----------------------- FURTHER ACCURACY ASSESSMENT ---------------------------
        # Create land value map
        if i_year == 1:
            lv_l_arr = np.reshape(rfr.predict(X_2d), factors_mat.shape[1:3], order='F')
            plot_rasterize_2d_maps_reg(abs_link_Processed_tif, lv_l_arr, do_satellite, 'RFR', limits, pixel_size)


        # plot location of misclassified/mispredicted
        plot_misclassified_reg(abs_link_Processed, X_test_idx, y_test_l, rfr.predict(X_test), name, limits, n_groups, do_satellite, pixel_size)

        print '--- ... finished Random Forest Regression. ---'


    if do_train_with_all and i_year == 1:
        pls.fit(X, y_l)
        rfr.fit(X, y_l)



    print '\n\n#######################################################'
    print '#######################################################'
    # classification accuracies and precisions
    t = PrettyTable(['Year', 'Method', 'Accuracy', 'Precision'])


    # regression accuracies and precisions
    if do_PLS:
        t.add_row([year, 'acc_PLS', '{0:.3f}'.format(acc_PLS), prec_PLS])
    if do_RFR:
        t.add_row([year, 'acc_RFR', '{0:.3f}'.format(acc_RFR), prec_RFR])

    print t

    print '... finished with processing year', year
    print '#######################################################'
    print '#######################################################\n\n'



    if do_only_land_value_map == 1:
        break


print '... processing finished.'








