from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

import numpy as np
from scipy import stats
import pandas as pd


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

import seaborn as sns
import matplotlib.patches as mpatches   # for manual legend entries

import rasterio as rio

import config


# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)




#   ------------------------------------------------------------
#   #############  GENERAL HELPER FUNCTIONS  #############
#   ------------------------------------------------------------
def lmu_to_class(predictions, limits = [-0.95978, -0.62325, 0.04980, 1.05937]):
    """
    Create classes according to user-defined limits (or standard limits in the IMU of 2010)

    :param predictions: predicted continuous values
    :param limits: limits for splitting the groups
    :return: predicted classes (after grouping)
    """
    predictions_class = np.zeros_like(predictions)

    if len(limits) == 4:
        for j in range(0, len(predictions)):
            if predictions[j] < limits[0]:
                predictions_class[j] = 1     # very low marginisation = very high socio-economic level
            elif predictions[j] < limits[1]:
                predictions_class[j] = 2     # low
            elif predictions[j] < limits[2]:
                predictions_class[j] = 3     # medium
            elif predictions[j] < limits[3]:
                predictions_class[j] = 4     # high
            else:
                predictions_class[j] = 5     # very high
    elif len(limits) == 2:
        for j in range(0, len(predictions)):
            if predictions[j] < limits[0]:
                predictions_class[j] = 1  # low marginisation = high socio-economic level
            elif predictions[j] < limits[1]:
                predictions_class[j] = 3  # medium
            else:
                predictions_class[j] = 5  # high = low socio-economic level (rank = 5)

    return predictions_class




def assess_regression_accuracy(reg, y_test_l, X_test, name, print_class_report, n_groups, limits, no_plot=False):
    """
    Several functions to assess the accuracy of a regressor based on test data such as R2, accuracy, precision, mean absolute error after
    classification

    :param reg: regressor object from scikit-learn
    :param y_test_l: labels of test set
    :param X_test: features of test set
    :param name: name of the classifier
    :param print_class_report: boolean if classification report should be printed
    :param n_groups: number of groups for the classification
    :param limits: classification borders
    :parma no_plot: supress plots if true
    :return:
        acc_reg: accuracy of the regressor
        prec_reg: array with precisions with elements as groups
    """
    # ------ REGRESSION ------
    expected = y_test_l
    predictions = reg.predict(X_test)

    if np.size(predictions.shape) > 1:
        predictions = predictions.flatten()

    if no_plot is False:
        plot_regression_scatter(expected, predictions, name)

    errors_l = abs(predictions - expected)
    print '\nMean Absolute Error:', round(np.mean(errors_l), 2), 'lmu.'
    print 'Explained variance:', '{0:.3f}'.format(metrics.explained_variance_score(expected, predictions))

    print '\nPearson Test_score_whole_data [-1, _0_, 1]:', '{0:.3f}'.format(stats.pearsonr(expected, predictions)[0])
    print 'R2 Test_score_whole_data [-x, _1_]:', '{0:.3f}'.format(metrics.r2_score(expected, predictions))
    print 'MeanAbsError whole_uniform_data:', '{0:.3f}'.format(metrics.mean_absolute_error(expected, predictions))
    print 'MeanSqEr whole_uniform_data:', '{0:.3f}'.format(metrics.mean_squared_error(expected, predictions))


    # ------ CLASSIFICATION ------
    # create classes
    expected_class = lmu_to_class(expected, limits)
    predictions_class = lmu_to_class(predictions, limits)

    if n_groups == 2:
        group_divider = np.mean([expected_class.max(), expected_class.min()])
        predictions_class[predictions_class > group_divider] = expected_class.max()
        predictions_class[predictions_class < group_divider] = expected_class.min()


    # Accuracy assessment (Jaccard Similarity Score)
    acc_reg = metrics.accuracy_score(expected_class, predictions_class)
    print '\nAccuracy classes:', acc_reg

    # Get precision of each of the classes
    prec_reg = metrics.precision_score(expected_class, predictions_class, average=None)
    print 'Precision classes:', prec_reg

    # Calculate the absolute errors and print the mean absolute error (mae)
    errors = abs(predictions_class - expected_class)
    print 'Mean Absolute Error:', round(np.mean(errors), 2), 'classes.'


    if print_class_report:
        print("Classification report for classifier %s:\n%s\n"
              % (name + ' with grid search', metrics.classification_report(expected_class, predictions_class)))

    print("\nConfusion matrix:\n%s" % metrics.confusion_matrix(expected_class, predictions_class))
    print '\n'


    # check if probability is available and calculate if not yet given
    if hasattr(reg, "predict_proba"):
        belief = reg.predict_proba(X_test)
    elif hasattr(reg, "decision_function"):  # use decision function
        belief = reg.decision_function(X_test)
        belief = (belief - belief.min()) / (belief.max() - belief.min())
    else:
        print 'No method for calculating the belief found.'
        belief = None

    # plot classification overviews
    if no_plot is False:
        if belief is None:
            plot_classify_analysis_reg(X_test, expected_class, predictions_class, name, None)
        elif np.size(belief.shape) == 1:
            plot_classify_analysis_reg(X_test, expected_class, predictions_class, name, belief)
        elif np.size(belief.shape) > 1:
            plot_classify_analysis_reg(X_test, expected_class, predictions_class, name, np.amax(belief, axis=1))

    return acc_reg, prec_reg



def plot_regression_scatter(expected, predictions, name):
    """
    Plot regression plot of divergence expected vs. predicted IMU

    :param expected: true IMU
    :param predictions: predicted IMU
    :param name: method
    """
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    sns.set_style("white")

    # plot scatter of regression
    fig = plt.figure(figsize=(2.75, 2.4))
    MAE = metrics.mean_absolute_error(expected, predictions)
    str_acc = '{:.3f}'.format(MAE)
    perfect = np.linspace(expected.min(), expected.max(), 100)
    plt.xlim(expected.min(), expected.max())
    plt.ylim(expected.min(), expected.max())
    plt.plot(perfect, perfect, c='black', linewidth=0.8)
    ax = sns.kdeplot(predictions, expected, shade=True)
    ax.collections[0].set_alpha(0)

    plt.scatter(predictions, expected, s=1, c='black', alpha=0.5, edgecolors=None, marker='.')

    plt.xlabel('Predicted')  # + ' (Mean abs err=' + str_acc +')')
    plt.ylabel('Expected')
    plt.xlim(expected.min(), expected.max())
    plt.ylim(expected.min(), expected.max())

    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches="tight")
    config.global_figure_count += 1





def plot_classify_analysis_reg(X_test, expected, predictions, name, belief):
    '''
    Plot several analyses of the classifier such as confusion matrix

    :param X_test: Test features
    :param expected: Test labels
    :param predictions: Predicted labels
    :param name: Name of the classifier
    :param belief: Belief if the classifier features to predict its probability for a correct prediction
    '''
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    accuracy = metrics.accuracy_score(expected,predictions)
    fig = plt.figure(figsize=(2.75, 2.4))

    # Plot violinplot of predictions
    sns.violinplot(predictions, expected, scale="count", orient='h')
    plt.xlabel('predictions')
    plt.ylabel('expected')
    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches = "tight")
    config.global_figure_count += 1

    # plot confusion matrix
    fig = plt.figure(figsize=(2.75, 2.4))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1

    confusion_matrix = metrics.confusion_matrix(expected, predictions)
    confusion_matrix_p = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    char_l = np.chararray(confusion_matrix.shape)
    char_l[:] = '('
    perc = np.core.defchararray.add(char_l, confusion_matrix_p.astype('|S4'))

    char_l = np.chararray(confusion_matrix.shape)
    char_l[:] = '\n'
    perc = np.core.defchararray.add(char_l, perc)

    char_r = np.chararray(confusion_matrix.shape)
    char_r[:] = ')'
    perc = np.core.defchararray.add(perc, char_r)

    cm_label = np.core.defchararray.add(confusion_matrix.astype(str), perc)

    labels = np.unique(expected).astype(int)

    sns.set_style("white")

    ax = sns.heatmap(confusion_matrix_p, annot=cm_label, fmt='s', cmap='Greys', xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 7})
    ax.add_patch(patches.Rectangle((0, 0), 5, 5, linewidth=1, edgecolor='black', fill=False))

    # color x and y labels according to group colour
    my_cmap = plt.cm.get_cmap('jet')
    colors = my_cmap([0.01, 0.25, 0.5, 0.75, 0.99])
    ax.get_xticklabels()[0].set_color(colors[0, :])
    ax.get_xticklabels()[1].set_color(colors[1, :])
    ax.get_xticklabels()[2].set_color(colors[2, :])
    ax.get_xticklabels()[3].set_color(colors[3, :])
    ax.get_xticklabels()[4].set_color(colors[4, :])
    ax.get_yticklabels()[0].set_color(colors[0, :])
    ax.get_yticklabels()[1].set_color(colors[1, :])
    ax.get_yticklabels()[2].set_color(colors[2, :])
    ax.get_yticklabels()[3].set_color(colors[3, :])
    ax.get_yticklabels()[4].set_color(colors[4, :])
    str_acc = '{:.3f}'.format(accuracy)
    plt.xlabel('Predicted group')  # , (Accuracy= ' + str_acc + ')')
    plt.ylabel('True group')
    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches = "tight")
    config.global_figure_count += 1



    # Plot belief and misprediction
    if belief is not None:
        fig = plt.figure(figsize=(2.75, 2.4))
        my_cmap = plt.cm.get_cmap('jet')

        colors = my_cmap([0.01, 0.25, 0.5, 0.75, 0.99])

        g1 = mpatches.Patch(color=my_cmap(0.01), label='Group 1')
        g2 = mpatches.Patch(color=my_cmap(0.25), label='Group 2')
        g3 = mpatches.Patch(color=my_cmap(0.5), label='Group 3')
        g4 = mpatches.Patch(color=my_cmap(0.75), label='Group 4')
        g5 = mpatches.Patch(color=my_cmap(0.99), label='Group 5')
        plt.legend(handles=[g1, g2, g3, g4, g5])

        for g in range(1, 6):
            indices = np.array(np.where(expected == g)).astype(int).flatten()
            expected_g = expected[indices]
            predictions_g = predictions[indices]
            belief_g = belief[indices]
            plt.scatter(belief_g, predictions_g - expected_g + np.random.normal(0, 0.05, len(predictions_g)), c=colors[g - 1], alpha=0.3,
                        s=4)

        plt.xlabel('belief')
        plt.ylabel('misprediction')
        plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches = "tight")
        config.global_figure_count += 1



def plot_misclassified_reg(abs_link_Processed, X_idx, expected, predictions, name, limits, n_groups, do_satellite, pixel_size):
    '''
    Plot the test samples depending on wheter the predictions where correct or wrong in two subplots after grouping according to limits

    :param abs_link_Processed: location where file is saved to
    :param X_idx: Index of settlements in the array
    :param expected: Expected socioeconomic groups
    :param predictions: Predicted socioeconomic groups
    :param name: Name of the classifier
    :param limits: limits for grouping the IMU into the classes
    :param n_groups: number of groups can be set to 2 for further assessment
    :param do_satellite: boolean if satellite image classification is undertaken
    '''
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1

    dim = np.load(abs_link_Processed + 'array/2000_all_mat_dim.npy')

    if do_satellite:
        land_value = rio.open(abs_link_Processed + 'tif/Satellite_prediction_level_' + name + '.tif').read(1)
    else:
        land_value = rio.open(abs_link_Processed + 'tif/Land_value_level_' + name + '.tif').read(1)

    # ------ CLASSIFICATION ------

    # create classes
    expected_class = lmu_to_class(expected, limits)
    predictions_class = lmu_to_class(predictions, limits)
    if n_groups == 2:
        group_divider = np.mean([expected_class.max(), expected_class.min()])
        predictions_class[predictions_class > group_divider] = expected_class.max()
        predictions_class[predictions_class < group_divider] = expected_class.min()
    my_cmap = plt.cm.get_cmap('jet')
    colors = my_cmap([0.01, 0.25, 0.5, 0.75, 0.99])

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

    f = plt.figure(figsize=(5.5, 2.4))




    ## -----  create first subplot with correct predicted
    ax = f.add_subplot(121)
    pos1 = ax.get_position()  # get the original position
    pos2 = [pos1.x0 + 0.02, pos1.y0, pos1.width, pos1.height]
    ax.set_position(pos2)  # set a new position
    # contour line and land value

    cs = plt.contour(np.flipud(land_value), limits, colors='k', alpha=0.8, linestyles='solid', linewidths=0.3)

    im = plt.imshow(np.flipud(land_value), interpolation=None, cmap=cm, alpha=0.3, vmin=-1.5, vmax=1.5, zorder=1)
    # correctly predicted urbanized regions

    # plot existing urban structure

    imu_exist = rio.open(abs_link_Processed + 'tif/IMU2000_new.tif').read(1)

    imu_exist[imu_exist == 255] = np.nan
    if do_satellite == 0:
        plt.imshow(np.flipud(imu_exist), alpha=1, interpolation=None, cmap='Greys', zorder=2)
    plt.xlim((0, dim[1]))
    plt.ylim((0, dim[0]))

    for g in range(1, 6):
        X_idx_correct = X_idx[(expected_class == g) & (expected_class == predictions_class), :]
        scat = plt.scatter(X_idx_correct[:, 1], dim[0] - X_idx_correct[:, 0], c=colors[g - 1], alpha=1, s=3, marker='s', zorder=3)


    plt.gca().set_aspect(aspect='equal', adjustable='box')
    ticks = (ax.get_xticks() * (pixel_size / 1000)).astype('int')  # get ticks in km
    ax.set_xticklabels(ticks)
    ax.xaxis.set_tick_params(pad=-1)
    ticks = (ax.get_yticks() * (pixel_size / 1000)).astype('int')  # get ticks in km
    ax.set_yticklabels(ticks)
    ax.yaxis.set_tick_params(pad=-1)
    plt.xlabel('East in km')
    plt.ylabel('North in km')



    ## -----  create second subplot with wrongly predicted
    ax2 = f.add_subplot(122, sharex=ax, sharey=ax)
    pos1 = ax2.get_position()  # get the original position
    pos2 = [pos1.x0 + 0.02, pos1.y0, pos1.width, pos1.height]
    ax2.set_position(pos2)  # set a new position

    # contour line and land value
    if do_satellite:
        cs = plt.contour(np.flipud(land_value), limits, colors='k', alpha=0., linestyles='solid', linewidths=0.05)
    else:
        cs = plt.contour(np.flipud(land_value), limits, colors='k', alpha=0.8, linestyles='solid', linewidths=0.3)

    im = plt.imshow(np.flipud(land_value), interpolation=None, cmap=cm, alpha=0.3, vmin=-1.5, vmax=1.5, zorder=1)

    # plot existing urban structure
    if do_satellite == 0:
        plt.imshow(np.flipud(imu_exist), alpha=1, interpolation=None, cmap='Greys', zorder=2)
    plt.xlim((0, dim[1]))
    plt.ylim((0, dim[0]))

    # wrongly predicted urbanized regions
    for g in range(1, 6):
        X_idx_wrong = X_idx[(expected_class == g) & (expected_class != predictions_class), :]
        plt.scatter(X_idx_wrong[:, 1], dim[0] - X_idx_wrong[:, 0], c=colors[g - 1], alpha=1, s=2, marker='s', zorder=3)


    plt.gca().set_aspect(aspect='equal', adjustable='box')
    ticks = (ax.get_xticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_xticklabels(ticks)
    ax2.xaxis.set_tick_params(pad=-1)
    ticks = (ax.get_yticks() * (pixel_size/1000)).astype('int')  # get ticks in km
    ax.set_yticklabels(ticks)
    ax2.yaxis.set_tick_params(pad=-1)

    plt.xlabel('East in km')
    plt.ylabel('North in km')



    # create legend
    if do_satellite == 0:
        g0 = mpatches.Patch(color='white', label='Urban 2000')
    g1 = mpatches.Patch(color=my_cmap(0.01), label='Group 1')
    g2 = mpatches.Patch(color=my_cmap(0.25), label='Group 2')
    g3 = mpatches.Patch(color=my_cmap(0.5), label='Group 3')
    g4 = mpatches.Patch(color=my_cmap(0.75), label='Group 4')
    g5 = mpatches.Patch(color=my_cmap(0.99), label='Group 5')
    if do_satellite == 0:
        plt.legend(handles=[g0, g1, g2, g3, g4, g5], loc=[1.02, -0.1])

        # plot existing urban structure
        plt.imshow(np.flipud(imu_exist), alpha=1, interpolation=None, cmap='Greys', zorder=2)
    else:
        plt.legend(handles=[g1, g2, g3, g4, g5], loc=[1.02, -0.1])


    # create colorbar
    cb_ax = f.add_axes([0.95, 0.45, 0.02, 0.45])
    cbar = plt.colorbar(im, shrink=0.6, fraction=0.1, ticks=[-1.5] + limits + [1.5], cax=cb_ax)
    cbar.ax.yaxis.set_tick_params(pad=1, length=0)
    cbar.ax.set_ylabel('Predicted IMU')
    cbar.set_label("Predicted IMU", fontsize=8, y=0.5, rotation=90)
    cbar.ax.text(-1.3, 0.09, "G1", fontsize=7, rotation=0, va='center')
    cbar.ax.text(-1.3, 0.22, "G2", fontsize=7, rotation=0, va='center')
    cbar.ax.text(-1.3, 0.4, "G3", fontsize=7, rotation=0, va='center')
    cbar.ax.text(-1.3, 0.68, "G4", fontsize=7, rotation=0, va='center')
    cbar.ax.text(-1.3, 0.9, "G5", fontsize=7, rotation=0, va='center')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.solids.set_edgecolor("face")
    cbar.add_lines(cs)  # Add the contour line levels to the colorbar


    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', dpi=300,
                bbox_inches="tight")
    config.global_figure_count += 1

    # plt.show()





#   ------------------------------------------------------------
#   #############  RFR HELPER FUNCTIONS  #############
#   ------------------------------------------------------------

def rfr_param_selection(X, y, nfolds):
    """
    Calls the Grid Search CV for the parameter selection of a random forest

    :param X: Features
    :param y: Labels
    :param nfolds: number n of n-fold validation
    :return: best parameters
    """
    n_trees =   [50, 100, 150, 200]
    depth_max = [50, 100, 150, 200]
    n_trees =   [100]
    depth_max = [100]
    param_grid = {'n_estimators': n_trees, 'max_depth' : depth_max}
    grid_search = GridSearchCV(RandomForestRegressor(oob_score=True), param_grid, cv=nfolds, return_train_score=True)
    grid_search.fit(X, y)
    grid_search.best_params_

    print("R2 RFR CV Grid Search: %0.2f (+/- %0.2f)" % (grid_search.best_score_, grid_search.cv_results_['std_test_score'][grid_search.best_index_] * 2))
    print 'RFR Mean CV score: %.4g' % grid_search.best_score_


    df_gridsearch = pd.DataFrame(grid_search.cv_results_)
    max_scores = df_gridsearch.groupby(['param_n_estimators',
                                        'param_max_depth']).max()
    max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]



    print 'RFR parameter selection finished with %d trees and %d depth.\n' \
          % (grid_search.best_params_['n_estimators'], grid_search.best_params_['max_depth'])
    return grid_search.best_params_



def rfr_param_selection_test(X, y, X_test, y_test, limits):
    """
    Calls the Grid Search CV for the parameter selection of a random forest

    :param X: Features
    :param y: Labels
    :param nfolds: number n of n-fold validation
    :return: best parameters
    """
    n_trees =   [50, 100, 150, 200]
    depth_max = [50, 100, 150, 200]
    n_trees =   [100]
    depth_max = [100]
    accuracy = np.zeros((len(n_trees), len(depth_max)))

    for idx0, trees in enumerate(n_trees):
        for idx1, depth in enumerate(depth_max):
            rfr = RandomForestRegressor(n_jobs=1, n_estimators=trees, max_depth=depth)
            rfr.fit(X, y)
            new_accuracy, new_precision = assess_regression_accuracy(rfr, y_test, X_test, 'RFR', 0, 5, limits, no_plot=True)
            accuracy[idx0, idx1] = new_accuracy

    df = pd.DataFrame(accuracy, index=n_trees, columns=depth_max)

    plt.show()

    return 0

