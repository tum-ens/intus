"""
City.py
intus
AMC @ TUM ENS
"""

import numpy as np
from scipy import spatial

import intus.Functions as intus
from .Cell import Cell
from intus.Functions.ca_helper_functions import plot_ca_map


class City():
    # --------------------------------------------------------------------------------
    def __init__(self, SLOPE, LANDUSE, EXCLUDED, URBAN, TRANSPORT, HILLSHADE,
                 VALUE, SE_LEVEL, IMU, URBDIST, k, grid_size, N_type, urban_seeds, se_ratio):

        # Input data
        self.SLOPE =    SLOPE
        self.LANDUSE =  LANDUSE
        self.EXCLUDED = EXCLUDED
        self.URBAN =    URBAN
        self.TRANSPORT = TRANSPORT
        self.HILLSHADE = HILLSHADE
        self.VALUE =    VALUE
        self.SE_LEVEL = SE_LEVEL
        self.IMU = IMU
        self.URBDIST = URBDIST
        self.k = k  # Growth coefficients

        # City
        self.grid_size = grid_size  # (rows, cols)
        self.N_type = N_type  # Neighborhood type
        self.se_ratio = se_ratio

        self.p_G_urbanization_dispersion = intus.calculate_p_G_urbanization_dispersion(self.k[0], self.grid_size)  # spontaneous growth
        self.urban_seeds = urban_seeds

        # Results
        self.landuse = np.zeros(self.grid_size)
        self.color_growth = np.array([[np.array([1., 1., 1.]) for col in range(self.grid_size[1])] for row in range(self.grid_size[0])])
        self.se_level = self.SE_LEVEL   # in case any information on se-level is given (for future work)


        # initialize City
        self.initialize_city()




    def run_growth_cycle(self):
        """
        Runs one growth cycle. A growth cycle represents a year of growth.
        The growth coefficients remain the same through the whole cycle.
        """
        gc_spont = 0
        gc_spread = 0
        gc_edge = 0
        gc_road = 0

        # add colors for each group
        colorlist = [(0., 0., 0.536), (0., 0.504, 1.), (0.49, 1., 0.478), (1., 0.582, 0.), (0.536, 0., 0.)]

        new_urbanized = np.empty((0, 3))

        # Cell loop
        for row in range(self.grid_size[0]):
            if row % 100 == 0:
                print 'run_growth_cycle row', row, '/', self.grid_size[0]
            for col in range(self.grid_size[1]):

                # checks if Cell is available for urbanization
                available = self.check_cell_availability(row, col)

                if available:
                    # define neighborhood
                    Neighborhood = self.define_neighborhood(row, col)

                    # extract neighborhood data
                    ### Urban
                    S_N = self.extract_neighborhood_attributes(Neighborhood, 'urban')
                    ### Land use
                    l_N = self.extract_neighborhood_attributes(Neighborhood, 'land_use')
                    ### Exclude
                    e_N = self.extract_neighborhood_attributes(Neighborhood, 'excluded')
                    ### Transport (roads)
                    r_N = self.extract_neighborhood_attributes(Neighborhood, 'transport')
                    ### Probability of urbanization
                    p_N_urbanization = self.extract_neighborhood_attributes(Neighborhood, 'p_urbanization')
                    ### Colors
                    c_N = self.extract_neighborhood_attributes(Neighborhood, 'colors')
                    ### Land Value
                    v_N = self.extract_neighborhood_attributes(Neighborhood, 'value')

                    # calculate Cell state
                    old_S_N = np.copy(S_N)
                    new_S_C, new_S_N, new_c_C, new_c_N, n_spont, n_spread, n_edge, n_road = self.Grid[row, col].calculate_new_state(
                                                                                        S_N, l_N, e_N, r_N, v_N, p_N_urbanization, c_N)

                    # add new Cell state to result matrix
                    self.landuse[row, col] = new_S_C

                    # add growth rule colors to result matrix
                    self.color_growth[row, col] = new_c_C

                    # update Grid
                    self.update_neighborhood_attributes(row, col, new_S_N, new_c_N)

                    # store urbanized cell for determination of se-level after growth cycle
                    if new_S_C == 1:
                        l_v = np.mean(v_N)
                        new_urbanized = np.append(new_urbanized, np.array([[row, col, l_v]]), axis=0)

                    if (new_S_N != old_S_N).any():
                        change_arr = (new_S_N != old_S_N)
                        l_v = np.mean(v_N)

                        changed = np.array(np.where(change_arr))[0]

                        for i in changed:
                            if i == 0:
                                pos = [max(row - 1, 0),         max(col - 1, 0),                        l_v]
                            elif i == 1:
                                pos = [max(row - 1, 0),         col,                                    l_v]
                            elif i == 2:
                                pos = [max(row - 1, 0),         min(col + 1, self.grid_size[1] - 1),    l_v]
                            elif i == 3:
                                pos = [row,                     max(col - 1, 0),                        l_v]
                            elif i == 4:
                                pos = [row,                     min(col + 1, self.grid_size[1] - 1),    l_v]
                            elif i == 5:
                                pos = [min(row + 1, self.grid_size[0] - 1), max(col - 1, 0),            l_v]
                            elif i == 6:
                                pos = [min(row + 1, self.grid_size[0] - 1), col,                        l_v]
                            elif i == 7:
                                pos = [min(row + 1, self.grid_size[0] - 1), min(col + 1, self.grid_size[1] - 1), l_v]

                            new_urbanized = np.append(new_urbanized, np.array([pos]), axis=0)

                    gc_spont = gc_spont +  n_spont
                    gc_spread = gc_spread + n_spread
                    gc_edge = gc_edge + n_edge
                    gc_road = gc_road + n_road


        # determine se-level depending on land value, percentages
        new_urbanized = new_urbanized[new_urbanized[:, 2].argsort()] # sort in ascending land value
        i_splits = np.array_split(new_urbanized, 100)
        ratios = self.se_ratio

        ratios_cum = np.cumsum(ratios)

        i_groups = np.zeros(5)
        group_counter = 0

        for i in range(0,100):
            if i == ratios_cum[group_counter]:
                group_counter += 1

            i_groups[group_counter] = i_groups[group_counter] +  np.size(i_splits[i], 0)
        i_groups = np.cumsum(i_groups)

        # assign se-level for each cell
        for i in range(0, np.size(new_urbanized, 0)):
            row = int(new_urbanized[i, 0])
            col = int(new_urbanized[i, 1])

            if i < i_groups[0]:
                new_se_C = 5
            elif i < i_groups[1]:
                new_se_C = 4
            elif i < i_groups[2]:
                new_se_C = 3
            elif i < i_groups[3]:
                new_se_C = 2
            elif i < i_groups[4]:
                new_se_C = 1

            self.se_level[row, col] = int(new_se_C)

        return gc_spont, gc_spread, gc_edge, gc_road



    def initialize_city(self):
        """
        Generates grid as array of Cell objects and initializes the Cell values
		"""

        # initialize list of objects
        self.Grid = np.empty((self.grid_size[0], self.grid_size[1]), dtype=object)

        max_dist = np.amax(self.URBDIST)

        # create cells
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                # Cell attributes from input data
                slope =     self.SLOPE[row, col]
                land_use =  self.LANDUSE[row, col]
                excluded =  self.EXCLUDED[row, col]
                urban =     self.URBAN[row, col]
                transport = self.TRANSPORT[row, col]
                value =     self.VALUE[row, col]
                se_level =  self.SE_LEVEL[row, col]

                urbdist =   (-1 / (1 * max_dist)) * self.URBDIST[row, col] + 1

                # create cell object
                self.Grid[row, col] = Cell(slope, land_use, excluded, urban, transport, value, se_level,
                                           self.k, self.p_G_urbanization_dispersion, urbdist)
            if row % 100 == 0:
                print 'initialize city, row', row, '/', self.grid_size[0]

        # add seeds
        self.add_urban_seeds()

        # update colors
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                self.color_growth[row, col] = self.Grid[row, col].c_C

    #
    def add_urban_seeds(self):
        """
        Includes urban seeds depending on the mode selected:
			0	Seeds from URBAN input file
			1	One urban center in the middle of the grid
			2	Random U urban seeds
			3	Urban seeds at specific positions
		"""

        # One urban seed in the city center
        if self.urban_seeds[0] == 1:  # at center
            mid_row = int(np.ceil(self.grid_size[0] / 2))
            mid_col = int(np.ceil(self.grid_size[1] / 2))
            self.Grid[mid_row, mid_col].S_C = 1
            self.landuse[mid_row, mid_col] = 1
            self.Grid[mid_row, mid_col].c_C = np.array([0., 0., 0.])
            self.color_growth[mid_row, mid_col] = np.array([0., 0., 0.])

        # Urban seeds at fixed locations
        elif self.urban_seeds[0] == 2:  # random
            for urban_seed in range(self.urban_seeds[1]):
                row = np.random.randint(0, self.grid_size[0])
                col = np.random.randint(0, self.grid_size[1])
                self.Grid[row, col].S_C = 1
                self.landuse[row, col] = 1
                self.Grid[row, col].c_C = np.array([0., 0., 0.])
                self.color_growth[row, col] = np.array([0., 0., 0.])

        # Urban seeds at fixed locations
        elif self.urban_seeds[0] == 3:  # fixed
            for urban_seed in range(len(self.urban_seeds[1])):
                row = self.urban_seeds[1][urban_seed][0]
                col = self.urban_seeds[1][urban_seed][1]
                self.Grid[row, col].S_C = 1
                self.landuse[row, col] = 1
                self.Grid[row, col].c_C = np.array([0., 0., 0.])
                self.color_growth[row, col] = np.array([0., 0., 0.])




    #
    def check_cell_availability(self, row, col):
        """
        Checks if cell is available for urbanization depending on the layer information.
        Excluded and already urban cells are not available for urbanization.
        """

        if self.Grid[row, col].excluded == 1:
            return False
        elif self.Grid[row, col].S_C == 1:
            return False
        elif self.Grid[row, col].land_use != 1:
            return False
        else:
            return True




    #
    def define_neighborhood(self, row, col):
        """
		Defines Cell neighborhood according to neighborhood type (N_type)
		"""

        if (self.N_type == 'Moore'):
            # initialize list of objects
            Neighborhood = np.empty((8), dtype=object)

            # add neighboring Cell objects
            Neighborhood[0] = self.Grid[max(row - 1, 0), max(col - 1, 0)]
            Neighborhood[1] = self.Grid[max(row - 1, 0), col]
            Neighborhood[2] = self.Grid[max(row - 1, 0), min(col + 1, self.grid_size[1] - 1)]
            Neighborhood[3] = self.Grid[row, max(col - 1, 0)]
            Neighborhood[4] = self.Grid[row, min(col + 1, self.grid_size[1] - 1)]
            Neighborhood[5] = self.Grid[min(row + 1, self.grid_size[0] - 1), max(col - 1, 0)]
            Neighborhood[6] = self.Grid[min(row + 1, self.grid_size[0] - 1), col]
            Neighborhood[7] = self.Grid[min(row + 1, self.grid_size[0] - 1), min(col + 1, self.grid_size[1] - 1)]

        return Neighborhood



    #
    def extract_neighborhood_attributes(self, Neighborhood, attribute):
        """
		Returns an array of neighborhood attribute as array
		"""

        if attribute == 'urban':
            N_parameters = np.array([Cell.S_C for Cell in Neighborhood])

        elif attribute == 'land_use':
            N_parameters = np.array([Cell.land_use for Cell in Neighborhood])

        elif attribute == 'excluded':
            N_parameters = np.array([Cell.excluded for Cell in Neighborhood])

        elif attribute == 'transport':
            N_parameters = np.array([Cell.transport for Cell in Neighborhood])

        elif attribute == 'p_urbanization':
            p_G_urbanization_dispersion = [Cell.p_G_urbanization_dispersion for Cell in Neighborhood]
            p_C_urbanization_slopeResistance = [Cell.p_C_urbanization_slopeResistance for Cell in Neighborhood]
            # N_parameters = np.multiply(p_G_urbanization_dispersion, p_C_urbanization_slopeResistance)
            N_parameters = np.array([Cell.p_C_urbanization_slopeResistance for Cell in Neighborhood])

        elif attribute == 'colors':
            N_parameters = np.array([Cell.c_C for Cell in Neighborhood])

        elif attribute == 'value':
            N_parameters = np.array([Cell.value for Cell in Neighborhood])

        return N_parameters



    #
    def update_neighborhood_attributes(self, row, col, new_values, colors):
        """
		Updates neighborhood attributes based on the new cell states, for new_spreading_center and  road_influenced_growth
		"""

        if self.N_type == 'Moore':
            # Urban values (binary)
            # Update values in grid
            self.Grid[max(row - 1, 0), max(col - 1, 0)].S_C =                       new_values[0]
            self.Grid[max(row - 1, 0), col].S_C =                                   new_values[1]
            self.Grid[max(row - 1, 0), min(col + 1, self.grid_size[1] - 1)].S_C =   new_values[2]

            self.Grid[row, max(col - 1, 0)].S_C =                                   new_values[3]
            self.Grid[row, min(col + 1, self.grid_size[1] - 1)].S_C =               new_values[4]

            self.Grid[min(row + 1, self.grid_size[0] - 1), max(col - 1, 0)].S_C =   new_values[5]
            self.Grid[min(row + 1, self.grid_size[0] - 1), col].S_C = new_values[6]
            self.Grid[min(row + 1, self.grid_size[0] - 1), min(col + 1, self.grid_size[1] - 1)].S_C = new_values[7]

            # Update values in landuse matrix
            self.landuse[max(row - 1, 0), max(col - 1, 0)] =                        new_values[0]
            self.landuse[max(row - 1, 0), col] =                                    new_values[1]
            self.landuse[max(row - 1, 0), min(col + 1, self.grid_size[1] - 1)] =    new_values[2]
            self.landuse[row, max(col - 1, 0)] =                                    new_values[3]
            self.landuse[row, min(col + 1, self.grid_size[1] - 1)] =                new_values[4]
            self.landuse[min(row + 1, self.grid_size[0] - 1), max(col - 1, 0)] =    new_values[5]
            self.landuse[min(row + 1, self.grid_size[0] - 1), col] =                new_values[6]
            self.landuse[min(row + 1, self.grid_size[0] - 1), min(col + 1, self.grid_size[1] - 1)] = new_values[7]

            # Urban colors (RGB)
            # Update values in grid
            self.Grid[max(row - 1, 0), max(col - 1, 0)].c_C =                       colors[0]
            self.Grid[max(row - 1, 0), col].c_C =                                   colors[1]
            self.Grid[max(row - 1, 0), min(col + 1, self.grid_size[1] - 1)].c_C =   colors[2]
            self.Grid[row, max(col - 1, 0)].c_C =                                   colors[3]
            self.Grid[row, min(col + 1, self.grid_size[1] - 1)].c_C =               colors[4]
            self.Grid[min(row + 1, self.grid_size[0] - 1), max(col - 1, 0)].c_C =   colors[5]
            self.Grid[min(row + 1, self.grid_size[0] - 1), col].c_C =               colors[6]
            self.Grid[min(row + 1, self.grid_size[0] - 1), min(col + 1, self.grid_size[1] - 1)].c_C = colors[7]

            # Update in color matrix
            self.color_growth[max(row - 1, 0), max(col - 1, 0)] =                       colors[0]
            self.color_growth[max(row - 1, 0), col] =                                   colors[1]
            self.color_growth[max(row - 1, 0), min(col + 1, self.grid_size[1] - 1)] =   colors[2]
            self.color_growth[row, max(col - 1, 0)] =                                   colors[3]
            self.color_growth[row, min(col + 1, self.grid_size[1] - 1)] =               colors[4]
            self.color_growth[min(row + 1, self.grid_size[0] - 1), max(col - 1, 0)] =   colors[5]
            self.color_growth[min(row + 1, self.grid_size[0] - 1), col] =               colors[6]
            self.color_growth[min(row + 1, self.grid_size[0] - 1), min(col + 1, self.grid_size[1] - 1)] = colors[7]

