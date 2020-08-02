"""
runme.py
intus
AMC @ TUM ENS
"""

import intus.Classes as intus
import os
from osgeo import gdal
import numpy as np
import scipy.ndimage



# This runfile imports the geotagged tif files (SLEUTH layers and land value map) simulates urban growth including the socioeconomic level
# in mexico city

# Last changed: 2 August 2020, Lukas Poehler (lukas.poehler@tum.de)




## Set Parameters


# GENERAL
city_name = 'mexico_city'
sim_name = 'se_level'

N_type = 'Moore'  # Neighborhood type
R = 1  # Number of simulation Runs

start_date = None  # Start year
stop_date = None  # Stop year

pixel_size = 300.0

GC = 60  # Number of growth cycles


# URBAN GROWTH COEFFICIENTS
# The coefficients affect how the rules are applied to the data. Five coefficients
# are used in the SLEUTH model:
# 	k_dispersion:		Affects rule_spontaneous_growth.
# 	k_breed:			Is the probability of urbanization in the
# 						rule_new_spreading_center.
# 	k_spread:			Is the probability of occurrence of organic growth
# 						(rule_edge_growth)
# 	k_slopeResistance:	Affects the influence of terrain slope to urbanization. As
# 						its value increases, the ability to urbanize steepening
# 						slopes decreases.
# 	k_roadGravity:		Affects the influence of existing infrastructure to
# 						urbanization. As its value increases, cells far from
# 						existing roads are more likely to be urbanized.

growth_params = [0.3, 0.5, 0.2, 0.6, 0.12]  # 300

k_dispersion = growth_params[0]
k_breed = growth_params[1]
k_spread = growth_params[2]
k_slopeResistance = growth_params[3]
k_roadGravity = growth_params[4]

k = [k_dispersion, k_breed, k_spread, k_slopeResistance, k_roadGravity]


# URBAN SEEDS
# 	[0]					Seeds from URBAN input file
# 	[1]					One urban seed in the middle of the grid
# 	[2, U]				Random U urban seeds
# 	[3, [[row,col],]]	Urban seeds at specific positions

urban_seeds = [0]
# urban_seeds 			= [3, [[4, 4], [3, 4], [4, 5], [5, 4]]]



## Options
_debug = 0
_save = 1



# Get dimensions and urban center from urban seed file
urban_tif = gdal.Open(os.getcwd() + '/input/' + city_name + '/tif/Urban1997.tif')
urban_arr = urban_tif.GetRasterBand(1).ReadAsArray()
grid_size = np.array(urban_arr.shape) - 2

urban_center = scipy.ndimage.measurements.center_of_mass(urban_arr)



## SIMULATION


# Set up simulation
my_simulation = intus.Simulation(city_name, sim_name, R, GC, start_date, stop_date, grid_size,
                                 pixel_size, N_type, k, urban_seeds, urban_center, _debug, _save)


# Run simulation
print 'Simulating', sim_name, 'in', city_name

my_simulation.run()
