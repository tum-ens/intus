import numpy as np
from sklearn import preprocessing
from regression_functions import lmu_to_class

import matplotlib.pyplot as plt
import pandas as pd

# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)


# -----------------------

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



def preprocess_data_2d_predict(link, abs_link_Processed_array, chosen_factors, scaler, do_polynomials, do_satellite):
    """
    Preprocessing the data for predicting the whole grid, this means that no sampling is undertaken but apart from that the same 
    preprocessing as for the training process

    :param link: absolute link to the data
    :param abs_link_Processed_array: absolute link to the folder with the processed arrays
    :param chosen_factors: indices for the chosen factors
    :param scaler: object for scaling the features
    :param do_polynomials: boolean if polyonmials from the features are created
    :param do_satellite: boolean if satellite images are analyzed
    :return:
    """
    # load arrays
    if do_satellite == 1:
        factors = np.load(link + 'satellite_arr.npy')
        labels_factors = np.load(link + 'satellite_labels.npy')
    else:
        factors_mat = np.load(abs_link_Processed_array + '2000_all_factors_mat.npy')
        [n_elements, n_factors] = np.load(abs_link_Processed_array + '2000_all_dimensions.npy')
        factors = np.zeros([n_elements, n_factors])

        for i in range(0, n_factors):
            factors[:, i] = factors_mat[i, :, :].flatten('F')

        labels_factors = np.load(abs_link_Processed_array + '2000_all_labels_factors.npy')



    data = factors
    n_factors = factors.shape[1]-2  # removing the row and column indices

    if chosen_factors is not None:
        n_chosen_factors = len(chosen_factors)
    else:
        n_chosen_factors = n_factors

    # initialize variables
    print 'amount of samples:', factors.shape[0]


    # split up x by extracting row and column indices to X_idx
    X = np.copy(data)
    X_idx = X[:, n_factors:n_factors+2]
    X = np.delete(X, [n_factors,n_factors+1], axis=1)


    # select factors to use
    if chosen_factors is not None:
        X = X[:, chosen_factors]
        feature_list = np.array(labels_factors)
        feature_list = feature_list[np.array(chosen_factors)]
    else:
        feature_list = np.array(labels_factors)

    # scale variables
    X = scaler.transform(X)


    # create polynomials of features
    if do_polynomials:
        poly = preprocessing.PolynomialFeatures(2)
        X = poly.fit_transform(X)
        n_chosen_factors = X.shape[1]
        feature_list = np.array(poly.get_feature_names(feature_list))

    return X, feature_list, n_chosen_factors






def preprocessing_data(link, chosen_factors, chosen_groups, max_samples, scaler, do_polynomials, sampling, step, i_year):
    """
    Preprocessing the data for predicting the whole grid, this means that no sampling is undertaken but apart from that the same 
    preprocessing as for the training process

    :param link: absolute link to the data
    :param chosen_factors: indices for the chosen factors
    :param chosen_groups: create new classifications by merging the five original groups
    :param max_samples: maximum number of samples per group if sampling is not sparse sampling
    :param scaler: object for scaling the features
    :param do_polynomials: boolean if polyonmials from the features are created
    :param sampling: type of sampling to be used
    :param step: if sampling is sparse downsampling, defines the step-th sample to be kept in the sparse set
    :param i_year: index of the year (1, 2 or 3 in our case)
    :return:
    """

    #  -------------------- LOAD ARRAYS --------------------
    clean_factors = np.load(link + 'clean_factors.npy')
    labels_factors = np.load(link + 'labels_factors.npy')

    se_group = np.load(link + 'se_group.npy')
    se_level = np.load(link + 'se_level.npy')

    data = np.column_stack((clean_factors, se_level, se_group))


    # -------------- PREPARE FEATURE LIST ---------------

    # select factors to use
    if chosen_factors is not None:
        feature_list = np.array(labels_factors)
        feature_list = feature_list[np.array(chosen_factors)]
    else:
        feature_list = np.array(labels_factors)

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


    feature_list = np.array([feature_list_replacement[old_name] for old_name in feature_list])



    # ---- print info about data ----
    feature_list_stats = np.append(feature_list, ['se_level', 'se_group'])

    if chosen_factors is not None:
        X_stats = np.column_stack((clean_factors[:, chosen_factors], se_level, se_group))
    else:
        X_stats = np.column_stack((clean_factors, se_level, se_group))

    df = pd.DataFrame(X_stats, columns=feature_list_stats)

    np.round(df.describe(), 2).T[['count', 'mean', 'std', 'min', 'max']].to_csv(link + 'summary_stats.csv', sep=';', decimal=',')



    #  -------------------- INITIALIZE VARIABLES --------------------
    n_values_per_group = np.bincount(data[:, -1].astype(int))
    print 'Number of datapoints per selected group:', n_values_per_group[1:]

    n_factors = clean_factors.shape[1]-2    # remove row and col indices

    if chosen_factors is not None:
        n_chosen_factors = len(chosen_factors)
    else:
        n_chosen_factors = n_factors


    n_samples_max = np.minimum(n_values_per_group[1:], max_samples) # set maximum numbers of samples
    n_samples_min = min(n_values_per_group[1:])

    y_class_before = np.copy(data[:, -1])



    #  -------------------- CREATE NEW CLASSIFICAITON --------------------
    if chosen_groups == [1,3,5]:
        # assign new groups with mean of previous group borders
        limits = [ (-0.95978 + (-0.62325))/2, (0.04980 + 1.05937)/2 ]
        n_groups = 3
        data[:, -1] = lmu_to_class(data[:, -2], limits)

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        ax1.bar(np.arange(5) + 1, np.bincount(y_class_before.astype(int))[1:])
        ax1.set_title('All groups')

        ax2.bar(np.arange(5) + 1, np.bincount(data[:, -1].astype(int))[1:])
        ax2.set_title('Distribute group 2 and 4 to surrounding')
        plt.show()

    elif chosen_groups == [2,3,4]:
        # assign new groups with 1 integrated to 2 and 5 integrated to 4
        limits = [-0.62325, 0.04980]
        n_groups = 3

        groups_out = lmu_to_class(data[:, -2], limits)
        groups_out[groups_out == 1] = 2
        groups_out[groups_out == 5] = 4

        data[:, -1] = groups_out

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        ax1.bar(np.arange(5) + 1, np.bincount(y_class_before.astype(int))[1:])
        ax1.set_title('All groups')

        ax2.bar(np.arange(5) + 1, np.append(np.bincount(data[:, -1].astype(int))[1:],[0]))
        plt.show()

    elif chosen_groups == [1,5]:
        # only use outermost groups
        limits = [-0.95978, -0.62325, 0.04980, 1.05937]
        n_groups = len(chosen_groups)

        data_of_group1 = data[np.where(data[:, -1] == 1)]
        data_of_group5 = data[np.where(data[:, -1] == 5)]

        data = np.row_stack((data_of_group1, data_of_group5))

    else:
        limits = [-0.95978, -0.62325, 0.04980, 1.05937]
        n_groups = len(chosen_groups)


    n_values_per_group = np.bincount(data[:, -1].astype(int))
    print 'Number of datapoints per selected group after new grouping:', n_values_per_group[1:]




    # ---------------------- SAMPLING -----------------------
    #
    X = np.empty((0, data.shape[1]-2))
    y = np.empty((0))
    y_l = np.empty((0))


    if sampling == 'none':
        X = data[:, :-2]
        y = data[:, -1]
        y_l = data[:, -2]

    elif sampling == 'sparse_downsampling':
        np.random.shuffle(data)

        X = data[::step, :-2]
        y = data[::step, -1]
        y_l = data[::step, -2]

    elif sampling == 'uniform' or sampling == 'max_samples':
        for i in range(0,n_groups):
            g = chosen_groups[i]
            data_of_group = data[np.where(data[:,-1]==g)]
            np.random.shuffle(data_of_group)

            # uniform sampling for equal number of samples per groups
            if sampling == 'uniform':
                n_samples_g = n_samples_min  # limit samples to minimum group
                if i == 0:
                    print 'Param max samples:', max_samples
                    print 'Min number of samples per group', n_samples_min

            # allow more samples for groups with more number of sample points
            if sampling == 'max_samples':
                n_samples_g = n_samples_max[i]  # allow more samples per group than minimal
                if i == 0:
                    print 'Param max samples:', max_samples
                    print 'Max number of samples per group', n_samples_max

            X = np.append(X, data_of_group[0:n_samples_g, :-2], axis=0)
            y = np.append(y, data_of_group[0:n_samples_g, -1].astype(np.float), axis=0)
            y_l = np.append(y_l, data_of_group[0:n_samples_g, -2].astype(np.float), axis=0)


    # extract row and column indices
    X_idx = X[:, n_factors:n_factors+2]
    X = np.delete(X, [n_factors,n_factors+1], axis=1)

    # select factors to use
    if chosen_factors is not None:
        X = X[:, chosen_factors]


    # scale variables
    if i_year == 1:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    np.set_printoptions(precision=3)
    print 'Scaled X: '
    print(X[1:3, :])


    # create polynomials of features
    if do_polynomials:
        poly = preprocessing.PolynomialFeatures(2)
        X = poly.fit_transform(X)
        n_chosen_factors = X.shape[1]
        feature_list = np.array(poly.get_feature_names(feature_list))




    return X, X_idx, y, y_l, feature_list, n_chosen_factors, n_groups, limits, scaler
