import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches   # for manual legend entries
from mpl_toolkits.mplot3d import Axes3D

import config


# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)




def plot_2d_belief(X, y, X_train, y_train, X_test, y_test, svc, name):
    '''
    Plot the belief of a classifier together with the location of training and test sets, not used anymore

    :param X: Complete feature set
    :param y: Complete labels
    :param X_train: Training features
    :param y_train: Training labels
    :param X_test: Test features
    :param y_test: Test labels
    :param svc: Draws the decision function for a support vector classifier
    :param name: Name of the classifier
    '''
    plt.figure(figsize=(2.75, 2.4))
    plt.clf()
    sparse = 1
    sparse_X = 1

    # test data as +
    plt.scatter(X_test[::sparse, 0], X_test[::sparse, 1], c=y_test[::sparse], zorder=10, cmap='jet', s=60, alpha=1, marker='+')
    # train data as *
    plt.scatter(X_train[::sparse, 0], X_train[::sparse, 1], c=y_train[::sparse], zorder=10, edgecolor='grey', cmap='jet', s=5, alpha=0.5,
                marker='*')
    # all data as .
    plt.scatter(X[::sparse_X, 0], X[::sparse_X, 1], c=y[::sparse_X], zorder=10, cmap='jet', s=2, alpha=0.2)
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    h = 0.02
    XX, YY = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    if hasattr(svc, "predict_proba"):
        Z = svc.predict_proba(np.c_[XX.ravel(), YY.ravel()])
    elif hasattr(svc, "decision_function"):  # use decision function
        Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = (Z - Z.min()) / (Z.max() - Z.min())
    else:
        print 'No probability measure'
        return

    try:
        Z = np.amax(Z, axis=1)
    except:
        print

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z, cmap='jet', alpha=0.1)


    plt.legend
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches = "tight")
    config.global_figure_count += 1




#   ------------------------------------------------------------
#   #############  PCA/LDA HELPER FUNCTIONS  #############
#   ------------------------------------------------------------
def print_PCA_stats(pca, feature_list):
    """
    Print statistics of a Principal Component Analysis

    :param pca: object of the PCA
    :param feature_list: names of the features for showing the feature importance
    """
    print feature_list
    print 'PCA Components:'
    print np.transpose(pca.components_)
    print 'Singular Values:', pca.singular_values_
    print 'Explained Variance Ratio:', pca.explained_variance_ratio_
    print 'Explained Variance:', pca.explained_variance_
    print 'Noise Variance', pca.noise_variance_


def print_LDA_stats(lda, feature_list):
    """
    Print statistics of a Linear Discriminant Analysis

    :param lda: object of the LDA
    :param feature_list: names of the features for showing the feature importance
    """
    print feature_list
    print 'LDA Coefficients:'
    print np.transpose(lda.coef_)
    print 'Means:'
    print np.transpose(lda.means_)
    print 'Explained Variance Ratio:', lda.explained_variance_ratio_
    print 'Feature Scaling:', lda.scalings_



def plot_projected(X_pca, y, name):
    """
    Plot projected feature values in 2d or 3d with colors depending on the group number

    :param X_pca: Transformed variables in the space of the principal components
    :param y: Group labels
    :param name: Name for the filename
    """
    fig = plt.figure(figsize=(2.75, 2.4))
    dimension = X_pca.shape[1]


    if dimension == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='jet', s=3, alpha=0.2, vmin=1, vmax=5)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

    elif dimension == 3:
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y.astype(int), cmap='jet', s=3, alpha=0.5, vmin=1, vmax=5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    my_cmap = plt.cm.get_cmap('jet')

    g1 = mpatches.Patch(color=my_cmap(0.01), label='Group 1')
    g2 = mpatches.Patch(color=my_cmap(0.25), label='Group 2')
    g3 = mpatches.Patch(color=my_cmap(0.5), label='Group 3')
    g4 = mpatches.Patch(color=my_cmap(0.75), label='Group 4')
    g5 = mpatches.Patch(color=my_cmap(0.99), label='Group 5')
    plt.legend(handles=[g1, g2, g3, g4, g5])

    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches = "tight")
    config.global_figure_count += 1



def plot_projected_cont(X_pca, y_l, name):
    """
    Plot projected feature values in 2d or 3d with colors depending on the socioeconomic level (IMU)

    :param X_pca: Transformed variables in the space of the principal components
    :param y_l: IMU as labels
    :param name: Name for the filename
    """
    dimension = X_pca.shape[1]
    fig = plt.figure(1)
    fig = plt.figure(figsize=(2.75, 2.4))

    if dimension == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_l, cmap='jet', s=3, alpha=0.2)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

    elif dimension == 3:
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_l, cmap='jet', s=3, alpha=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    if dimension == 2:
        plt.colorbar()

    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches = "tight")
    config.global_figure_count += 1



def plot_loading_score(X_pca, components, feature_list, y):
    """
    Creates a 2d loading and score plot for PCA and PLS

    :param X_pca: Transformed features as scores
    :param components: Transformed features as vectors in the new coordinate system
    :param feature_list: Names of the features
    :param y: Labels
    """
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1

    # 0,1 denote PC1 and PC2; change values for other PCs
    xs = X_pca[:, 0]
    ys = X_pca[:, 1]

    xvector = components[0]
    yvector = components[1]

    # Loading and score plot in one

    # a) loading
    plt.figure(figsize=(5.5, 2.4))
    ax1 = plt.subplot(121)

    loading_scale = 0.7*np.mean([max(xs),max(ys)])

    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        plt.arrow(0, 0, xvector[i] * loading_scale, yvector[i] * loading_scale, color='k', width=0.0005, head_width=0.1)
        plt.text(xvector[i] * (loading_scale*1.1 + 0.5), yvector[i] * (loading_scale*1.1 + 0.5), feature_list[i], color='k')

    plt.xlabel('PC1')
    plt.ylabel('PC2', labelpad=-2)


    # b) score
    plt.subplot(122, sharex=ax1, sharey=ax1)
    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        plt.arrow(0, 0, xvector[i] * loading_scale, yvector[i] * loading_scale, color='k', width=0.0005, head_width=0.1)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='jet', s=1, edgecolor=None, marker='.', alpha=0.3, vmin=1, vmax=5)
    plt.xlabel('PC1')
    plt.ylabel('PC2', labelpad=-2)


    my_cmap = plt.cm.get_cmap('jet')
    g1 = mpatches.Patch(color=my_cmap(0.01), label='Group 1')
    g2 = mpatches.Patch(color=my_cmap(0.25), label='Group 2')
    g3 = mpatches.Patch(color=my_cmap(0.5), label='Group 3')
    g4 = mpatches.Patch(color=my_cmap(0.75), label='Group 4')
    g5 = mpatches.Patch(color=my_cmap(0.99), label='Group 5')
    plt.legend(handles=[g1, g2, g3, g4, g5],  loc=[1.02, 0])


    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + '.pdf', bbox_inches = "tight")
    config.global_figure_count += 1








#   ------------------------------------------------------------
#   #############  RANDOM FOREST HELPER FUNCTIONS  #############
#   ------------------------------------------------------------


def print_RF_stats(rfc, X, feature_list, name):
    """
    Print statistics of a random forest

    :param rfc: object of the random forest
    :param X: features
    :param feature_list: names of the features
    :param name: name used for ploting the feature importances
    """
    print 'OOB_score:', rfc.oob_score_

    # Print and plot the feature ranking
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
                 axis=0)

    print("Feature ranking:")
    indices = np.argsort(importances)[::-1]

    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, feature_list[indices[f]], importances[indices[f]]))

    plot_feature_importance(feature_list, importances, std, name)




def plot_feature_importance(feature_list, importances, std, name):
    """
    Plot the feature importance of a random forest both in natural and sorted order from high to low

    :param feature_list: names of the features
    :param importances: importance in percentage of the particular feature
    :param std: standard deviation of the feature importance based on the different trees
    :param name: name used for ploting the feature importances
    """
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1

    indices = np.argsort(importances)[::-1]
    x_values = list(range(len(importances)))

    # Plot sorted bar chart with feature importance
    fig = plt.figure(figsize=(2.75, 2.4))
    plt.bar(x_values, importances[indices], color='grey', yerr=std[indices], orientation='vertical')
    plt.xticks(x_values, feature_list[indices], rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches="tight")
    config.global_figure_count += 1

    # Plot bar chart with feature importance
    fig = plt.figure(figsize=(2.75, 2.4))
    plt.bar(x_values, importances, yerr=std, color='grey', orientation='vertical')
    plt.xticks(x_values, list(feature_list), rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.savefig(config.global_path + config.global_year + '_' + str(config.global_figure_count) + '_' + name + '.pdf', bbox_inches="tight")
    config.global_figure_count += 1



