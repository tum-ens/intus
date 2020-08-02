"""
Simulation.py
intus
AMC @ TUM ENS
"""

import os
import numpy as np
import scipy.ndimage
from datetime import datetime
import matplotlib.pyplot as plt

import intus.Functions as intus
from .City import City
from process_functions.spatial_metrices import calculate_metrices, calculate_correct
from intus.Functions.ca_helper_functions import plot_lv_map, plot_ca_map, plot_ca_colors, plot_grow_history


class Simulation():
    # --------------------------------------------------------------------------------
    def __init__(self, city_name, sim_name, R, GC, start_date, stop_date, grid_size,
                 pixel_size, N_type, k, urban_seeds, urban_center, _debug, _save):

        # General
        self.city_name = city_name  # City name
        self.sim_name = sim_name    # Simulation name
        self.R = R                  # Number of runs
        self.GC = GC                # Length of growth cycle
        self.start_date = start_date    # Simulation start date (year)
        self.stop_date = stop_date      # Simulation stop date day (year)

        self.grid_size = [grid_size[0] + 2, grid_size[1] + 2]  # (rows, cols)
        self.N_type = N_type        # Neighborhood type

        self.my_dir = os.getcwd()   # Current directory
        self.result_dir = None      # Directory to save results

        # Input data
        self.SLOPE =        np.zeros(self.grid_size)
        self.LANDUSE =      np.ones(self.grid_size)
        self.EXCLUDED =     np.zeros(self.grid_size)
        self.URBAN =        np.zeros(self.grid_size)
        self.TRANSPORT =    np.zeros(self.grid_size)
        self.HILLSHADE =    np.zeros(self.grid_size)
        self.VALUE =        np.zeros(self.grid_size)
        self.SE_LEVEL =     np.zeros(self.grid_size)
        self.IMU =          np.zeros(self.grid_size)
        self.k = k  # Growth coefficients
        self.urban_seeds = urban_seeds
        self.urban_center = urban_center
        self.true_LandValue = None

        self.pixel_size = pixel_size

        #
        self._debug = _debug  # Debug level
        self._save = _save




    def run(self):
        """
        Runs a series of R runs of growth cycles for the defined city beginning at self.stop_date
        and completing at self.stop_date
        """
        # Result directory
        self.prepare_result_directory()

        # Input data
        self.read_input_data()


        # Urban growth simulations
        for run in range(self.R):

            # create City
            self.create_City()

            # calculate length of Growth Cycle (GC) (if not given)
            if not self.GC:
                self.GC = self.stop_date - self.start_date
            if not self.start_date:
                self.start_date = 0

            # Plot initial state
            self.plot_city(self.start_date)
            with open(self.result_dir + '/parameters.txt', 'a') as f:
                f.write('[k_dispersion, k_breed, k_spread, k_slopeResistance, k_roadGravity]:' + str(self.k))


            n_spont = np.zeros(self.GC).astype('int')
            n_spread = np.zeros(self.GC).astype('int')
            n_edge = np.zeros(self.GC).astype('int')
            n_road = np.zeros(self.GC).astype('int')


            # run Growth Cycles (GC)
            for gc in range(self.GC):
                # run growth cycle
                print '\n --------- Growth Cycle ', gc+1, ' ----------\n '

                print '[k_dispersion, k_breed, k_spread, k_slopeResistance, k_roadGravity]:', self.k

                n_spont[gc], n_spread[gc], n_edge[gc], n_road[gc] = self.my_city.run_growth_cycle()

                print '\n n_spont:', n_spont[gc], 'n_spread:', n_spread[gc], 'n_edge:', n_edge[gc], 'n_road:', n_road[gc]

                N_urbanized = sum(n_spont) + sum(n_spread) + sum(n_edge) + sum(n_road)
                plot_grow_history(n_spont, n_spread, n_edge, n_road, self.result_dir)

                # calculate urban metrices for every 10th finished growth cycle
                if (gc+1) % self.GC == 0 and gc>0:
                    percent_correct, percent_correct_location, n_urbanized_IMU, n_urbanized_sim = calculate_correct(self.my_city.se_level,
                                                                                                                    self.my_city.IMU,
                                                                                                                    self.my_city.URBAN,
                                                                                                                    self.result_dir)
                    A_built, A_urban, A_suburban, A_rural, \
                    A_urban_open, A_urban_ext, \
                    saturation, openness, proximity, cohesion, \
                    N_built = calculate_metrices(self.my_city.URBAN, self.my_city.se_level,
                                                        gc+1, self.result_dir, pixel_size=self.pixel_size)

                # plot city
                self.plot_city(self.start_date + gc + 1)




    def prepare_result_directory(self):
        """
        Creates a time stamped directory within the result folder.
        Returns path as string.
        """

        result_dir_ = self.my_dir + '/results'
        now = datetime.now().strftime('%Y%m%d %H%M')
        if self.sim_name:
            name = self.city_name + '_' + self.sim_name
        else:
            name = self.city_name
        self.result_dir = os.path.join(result_dir_, '{} - {}'.format(name, now))
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        if self._debug != 0:
            print 'Results directory'
            print '  ' + '{}\n'.format(self.result_dir)





    def read_input_data(self):
        """
        Returns input data as numpy arrays with adjusted ranges
        """

        input_dir = self.my_dir + '/input/'

        if self._debug != 0:
            print 'INPUT'
            print '------------------------------------------------------------'
            print 'Image files (tif) at ' + input_dir
            print 'Reading'

        # ENVIRONMENT
        # Layer	Description
        # 0		slope
        # 1		land use
        # 2		exclude
        # 3		urban
        # 4		transport
        # 5		hillshade

        LAYER = {0: 'Slope', 1: 'Land_Use_1997', 2: 'Excluded', 3: 'Urban1997', 4: 'Highways_1985', 5: 'Aspect',
                 6: 'Land_value_level_RFR', 7: 'IMU2010_new', 8: 'Urban1997_dist', 9: 'IMU2000_new', 10: 'IMU2005_new' }

        IMU2000 = None
        IMU2005 = None

        for layer in LAYER:

            # Filename
            img = input_dir + self.city_name + '/tif/' + LAYER[layer] + '.tif'

            # check if file exists
            if not os.path.isfile(img):
                print '  *WARNING: File ...' + LAYER[layer] + '.gif does not exist'

            else:
                try:
                    imarray = scipy.ndimage.imread(img, 'L')

                    if self._debug != 0:
                        print '   ' + LAYER[layer] + '.tif'

                    # resize to fit grid size
                    imarray = self.resize_image_array(imarray, method='max')    # ! contains a BUG: only columns are resized, rows are cut

                    # assign array to variable and scale according to CA requirements (= 1 means true)
                    if layer == 0:
                        imarray[imarray < 0] = 0
                        self.SLOPE = (np.flipud(imarray) / 100.).astype('float32')
                    elif layer == 1:
                        self.LANDUSE = (np.flipud(imarray)).astype('int8')
                    elif layer == 2:
                        self.EXCLUDED = (np.flipud(imarray) / 255).astype('int8')       # 0 - 1 (excluded)
                    elif layer == 3:
                        imarray[imarray == 255] = 1
                        imarray[imarray == 0] = 255
                        imarray[imarray == 1] = 0
                        self.URBAN = (np.flipud(imarray) / 255).astype('int8')          # 0 - 1 (urban)
                    elif layer == 4:
                        imarray[imarray == 255] = 1
                        imarray[imarray == 0] = 255
                        imarray[imarray == 1] = 0
                        self.TRANSPORT = (np.flipud(imarray) / 255).astype('int8')      # 0 - 1 (highways)
                    elif layer == 5:
                        self.HILLSHADE = (np.flipud(imarray) / 255.).astype('float32')
                    elif layer == 6:
                        plot_lv_map(imarray, result_dir=self.result_dir, pixel_size=self.pixel_size)    # plot land value map
                        self.true_LandValue = imarray
                        imarray = 255/(imarray.max()-imarray.min()) * (imarray-imarray.min())   # stretch land value to 0 - 255
                        self.VALUE = abs(255-np.flipud(imarray).astype('int16'))                # 0 - 255 (invert for land value)
                    elif layer == 7:
                        imarray[imarray == 255] = 0
                        self.IMU = np.flipud(imarray)
                        IMU2010 = np.copy(self.IMU)
                    elif layer == 8:
                        self.URBDIST = np.flipud(imarray).astype('float32')
                    elif layer == 9:
                        imarray[imarray == 255] = 0
                        IMU2000 = np.flipud(imarray)
                    elif layer == 10:
                        imarray[imarray == 255] = 0
                        IMU2005 = np.flipud(imarray)

                except:
                    print '  *WARNING: File ...' + LAYER[layer] + 'image could not be read'

        # Plot SLEUTH layers with initial urban seed and target se-groups
        plot_ca_map(self, result_dir=self.result_dir, pixel_size=self.pixel_size)
        plot_ca_map(self, result_dir=self.result_dir, without_lv = 1, pixel_size=self.pixel_size)

        # Plot all IMUs of the years
        plot_ca_map(self, target=IMU2000, result_dir=self.result_dir, pixel_size=self.pixel_size, name = 'IMU2000')
        plot_ca_map(self, target=IMU2005, result_dir=self.result_dir, pixel_size=self.pixel_size, name= 'IMU2005')
        plot_ca_map(self, target=IMU2010, result_dir=self.result_dir, pixel_size=self.pixel_size, name= 'IMU2010')

        # Plot newly urbanized IMUs in 2005 and 2010
        IMU2010[IMU2005 > 0] = 0
        IMU2005[IMU2000 > 0] = 0

        plot_ca_map(self, target=IMU2005, result_dir=self.result_dir, pixel_size=self.pixel_size, name = 'IMU2005_new')
        plot_ca_map(self, target=IMU2010, result_dir=self.result_dir, pixel_size=self.pixel_size, name= 'IMU2010_new')





    def create_City(self):
        """
        Creates city as City object
        """
        se_ratio = np.array([15, 41, 26, 8, 11])  # group 5 to 1

        self.my_city = City(self.SLOPE, self.LANDUSE, self.EXCLUDED, self.URBAN, self.TRANSPORT, self.HILLSHADE, self.VALUE,
                            self.SE_LEVEL, self.IMU, self.URBDIST, self.k, self.grid_size, self.N_type, self.urban_seeds, se_ratio)





    def plot_city(self, year):
        """
        plots the city
		"""
        if self._save != 0:
            plot_ca_map(self, se_levels_growth=self.my_city.se_level, result_dir=self.result_dir, year=year, pixel_size=self.pixel_size)
            plot_ca_colors(self.my_city.color_growth[1:self.grid_size[0] - 1, 1:self.grid_size[1] - 1], result_dir=self.result_dir, year=year, pixel_size=self.pixel_size)




    def resize_image_array(self, array, method='mean'):
        """
		Resizes image arrays to defined grid size using the specified method

		args:
			array
			method		'mean', 'max', 'min'
		"""

        if self._debug != 0:
            print '      array.shape', array.shape
            print '      grid.size', self.grid_size
        orig_shape = array.shape

        # compare size of image and grid
        _resize = False
        ### Rows
        if (self.grid_size[0] < array.shape[0]):
            # calculate group size
            g_row = int(array.shape[0] / self.grid_size[0])
            if g_row * self.grid_size[0] != array.shape[0]:
                g_row = int(np.ceil(array.shape[0] / self.grid_size[0])) - 1
                # crop input file to fit grid
                array = array[0:g_row * self.grid_size[0], :]
            _resize = True

        ### Cols
        if (self.grid_size[1] < array.shape[1]):
            # calculate group size
            g_col = int(array.shape[1] / self.grid_size[1])
            if g_col * self.grid_size[1] != array.shape[1]:
                g_col = int(np.ceil(array.shape[1] / self.grid_size[1])) - 1
                # crop input file to fit grid
                array = array[:, 0:g_col * self.grid_size[1]]
            _resize = True

        if _resize:

            # reshape image to fit grid size
            iii = 0
            jjj = 0
            temp = np.zeros((g_row, self.grid_size[1]))
            new_array = np.zeros(self.grid_size)

            if (method == 'mean'):
                for row in range(array.shape[0]):
                    # reshape columns
                    temp[iii] = array[row, :].reshape(self.grid_size[1], g_col).mean(axis=1)
                    iii += 1
                    if (iii == g_row):
                        # reshape rows
                        new_array[jjj, :] = temp.mean(axis=0)
                        iii = 0
                        jjj += 1

            elif (method == 'max'):
                for row in range(array.shape[0]):
                    # reshape columns
                    temp[iii] = array[row, :].reshape(-1, g_col).max(axis=1)
                    iii += 1
                    if (iii == g_row):
                        # reshape rows
                        new_array[jjj, :] = temp.max(axis=0)
                        iii = 0
                        jjj += 1

            elif (method == 'min'):
                for row in range(array.shape[0]):
                    # reshape columns
                    temp_array = array[row, :]
                    temp[iii] = array[row, :].reshape(-1, g_col).min(axis=1)
                    iii += 1
                    if (iii == g_row):
                        # reshape rows
                        new_array[jjj, :] = temp.min(axis=0)
                        iii = 0
                        jjj += 1

            #
            if self._debug != 0:
                print '      ' + 'image resized: ({},{}) to ({},{})'.format(orig_shape[0], orig_shape[1], \
                                                                            self.grid_size[0], self.grid_size[1])

            # plot resized array
            plt.imshow(new_array, vmin=0, vmax=100)
            plt.colorbar()
            plt.show()

        else:
            new_array = array

        return new_array
