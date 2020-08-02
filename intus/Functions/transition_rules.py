"""
transition_rules.py
AMC @ TUM ENS
"""

import numpy as np
from scipy import spatial


# SLEUTH model
# --------------------------------------------------------------------------------
# LAND COVER VALUES
# Value		Desc		  Color
#	0	Unclassified	0X000000
#	1	Urban			0X8b2323
#	2	Agriculture		0Xffec8b
#	3	Rangeland		0Xee9a49
#	4	Forest			0Xee9a49
#	5	Water			0X104e8b
#	6	Wetland			0X483d8b
#	7	Barren			0Xeec591
#	8	Tundra			0X323232
#	9	Ice & Snow		0XFFFFFF
#
# COEFFICIENTS
# The coefficients affect how the rules are applied to the data. Five coefficients
# are used in the SLEUTH model:
#	k_dispersion:		Affects rule_spontaneous_growth and search distance along
#						the road network as part of the rule_road_influenced_growth
#						attempts.
#	k_breed:			Is the probability of urbanization in the 
#						rule_new_spreading_center and affects the number of 
#						rule_road_influenced_growth attempts.
#	k_spread:			Is the probability of occurrence of organic growth 
#						(rule_edge_growth)
#	k_slopeResistance:	Affects the influence of terrain slope to urbanization. As 
#						its value increases, the ability to urbanize steepening 
#						slopes decreases.
#	k_roadGravity:		Affects the influence of existing infrastructure to
#						urbanization. As its value increases, cells far from
#						existing roads are more likely to be urbanized.


color_spontaneous = np.array([153, 0, 153]) / 255.  # violett
color_new_spread = np.array([248, 119, 117]) / 255.  # salmon
color_edge = np.array([255,221,51]) / 255.  # dark yellow
color_road = np.array([51, 153, 0]) / 255.  # green



# COEFFICIENTS
# --------------------------------------------------------------------------------
def calculate_p_G_urbanization_dispersion(k_dispersion, grid_size):
    """
	Calculates the Grid (global) likelihood of urbanization depending on the 
	k_dispersion and the size of the grid. A maximum of cells (50% of the grid 
	diagonal) are candidates to be spontaneously urbanized.
	
	args:
		k_dispersion
		grid_size		[rows, cols]
		
	returns:
		p_G_urbanization_dispersion
	"""

    # calculate number of potential spontaneous cells
    spontaneous_cells = k_dispersion * np.sqrt(grid_size[0] ** 2 + grid_size[1] ** 2) * 0.5

    # calculate the Grid probability of urbanization due to k_dispersion
    p_G_urbanization_dispersion = spontaneous_cells / (grid_size[0] * grid_size[1])

    return p_G_urbanization_dispersion


#
def calculate_p_C_urbanization_slopeResistance(k_slopeResistance, slope):
    """
	Calculates the weighted Cell probability of urbanization depending on the
	k_slopeResistance. If k_slopeResistance is close to 1, increasingly steeper 
	slopes are less likely to urbanize. As k_slopeResistance gets closer to zero, 
	an increase in local slope has less effect on the likelihood of urbanization.
	
	args:
		k_slopeResistance	 k[3]	Coefficient of resistance to steep urbanization
		slope						Slope of cell (0 [flat] to 1 [critical_slope])
	
	returns:
		p_C_urbanization_slopeResistance
	"""

    # calculate exponent
    exp = (1. / k_slopeResistance) - 1.

    # calculate weight of slope in urbanization
    weight_slope = slope ** exp

    # calculate probability of urbanization due to slope resistance
    p_C_urbanization_slopeResistance = 1 - weight_slope

    return p_C_urbanization_slopeResistance


#
def calculate_p_C_urbanization_roadGravity(k_roadGravity, distance_to_road,
                                           max_influence_distance=5.):
    """
	Calculates the weighted Cell probability of urbanization depending on the
	k_roadGravity. As k_roadGravity gets closer to zero, a shorter local distance 
	has less effect on the likelihood of urbanization. 
	
	args:
		k_roadGravity	 k[4]	Road gravity coefficient
		distance_to_road
	
	returns:
		p_C_urbanization_roadGravity
	"""

    if distance_to_road == 0:
        p_C_urbanization_roadGravity = 1.

    else:
        # normalize distance to road
        distance_to_road = distance_to_road / max_influence_distance

        # calculate exponent
        exp = (1. / k_roadGravity) - 1.

        # calculate weight of distance_to_road in urbanization
        weight_road = distance_to_road ** exp

        # calculate probability of urbanization due to road gravity
        p_C_urbanization_roadGravity = weight_road

    return p_C_urbanization_roadGravity







# GROWTH RULES
# --------------------------------------------------------------------------------
def rule_spontaneous_growth(S_C, p_G_urbanization_dispersion, p_C_urbanization_slopeResistance, c_C, urbdist):
    """
	Defines the occurrence of random urbanization of land based on the prob of
	urbanization due to k_dispersion and k_slopeResistance. 

	args:
		S_C								Current State of Cell S_C(t)
		p_C_urbanization_slopeResistance		
		p_G_urbanization_dispersion

	returns:
		S_C								State of Cell in next time step S_C(t+1)
	"""
    counter = 0

    # calculate probability of urbanization by spontaneous growth
    p_C_urbanization = p_G_urbanization_dispersion * p_C_urbanization_slopeResistance

    # scale probability by distance to urban center by 100% max
    p_C_urbanization = urbdist * p_C_urbanization

    # evaluate if cell is urbanized
    rand_num = np.random.uniform()
    if rand_num <= p_C_urbanization:
        S_C = 1
        # add color
        c_C = color_spontaneous # np.array([0, 101, 189]) / 255.  # TUM blue
        counter = 1


    return S_C, c_C, counter


#
def rule_new_spreading_center(k_breed, S_N, l_N, e_N, p_N_urbanization, c_N, size_spreading_center=2):
    """
	Determines whether any of the new, spontaneously urbanized cells will become
	new urban spreading centers based on k_breed and the land cover value of
	Neighboring cells (N). An urban spreading center is defined as a location with 
	two or more adjacent urbanized cells.
	
	args:
		k_breed
		S_N	 			 	  <np.array>	State of Neighboring cells S_N(t)
		p_N_urbanization 	  <np.array>	Probability of urbanization of adjacent cells
		size_spreading_center
		
	returns:
		S_N				 	  <np.array>	State of adjacent cells in next step S_N(t+1)
	"""
    counter = 0

    # evaluate if spreading centers breeds
    rand_num = np.random.uniform()
    if rand_num <= k_breed:

        # get available cells for urbanization, i.e.:
        # 	land_use = 1    l_N
        #	exclude  = 0    e_N
        #	urban	 = 0    S_N
        available_cells = (l_N == 1) & (e_N == 0) & (S_N == 0)

        # spreading center needs at least two available cells
        if available_cells.sum() >= size_spreading_center:

            # filter non-urban cells
            potential_spreading_center = p_N_urbanization[available_cells]

            # check if p_N_urbanization is the same for all cells
            equal_p_N_urbanization = np.equal.reduce(potential_spreading_center == p_N_urbanization[0])
            if equal_p_N_urbanization:

                # attempt to urbanize random cell
                for iii in range(size_spreading_center):

                    # evaluate if cell is urbanized
                    rand_cell = np.random.randint(0, len(potential_spreading_center))
                    rand_num = np.random.uniform()
                    if rand_num <= potential_spreading_center[rand_cell]:
                        # urbanize cell in S_N array
                        S_N[rand_cell] = 1

                        # add color
                        c_N[rand_cell] = color_new_spread # np.array([100, 160, 200]) / 255.  # Light blue
                        counter += 1

            else:

                # sort non-urban cells per p_C_urbanization
                sorted = np.sort(potential_spreading_center)[::-1]
                index_sorted = np.argsort(potential_spreading_center)[::-1]

                # attempt to urbanize the adjacent cells with the highest likelihood
                # of urbanization
                for iii in range(size_spreading_center):

                    # evaluate if cell is urbanized
                    rand_num = np.random.uniform()
                    if (rand_num <= sorted[iii]):
                        # urbanize cell in S_N array
                        S_N[index_sorted[iii]] = 1

                        # add color
                        c_N[index_sorted[iii]] = color_new_spread # np.array([100, 160, 200]) / 255.  # Light blue
                        counter += 1

    return S_N, c_N, counter


#
def rule_edge_growth(S_C, k_spread, p_C_urbanization_slopeResistance, S_N, c_C, size_urban_center=3):
    """
	Defines the organic growth (growth that stems from existing spreading centers).
	This growth propagates both the new centers from rule_new_spreading_center and
	the more established urban centers from earlier times. If a non-urban cell has
	at least three urbanized Neighboring cells, it has a certain global probability
	(defined by k_spread) to become urban.
	
	args:
		S_C						Current State of Cell S_C(t)
		k_spread
		p_C_urbanization_slopeResistance
		S_N		  <np.array>	Current States of Neighboring cells S_N(t)
	"""
    counter = 0

    # get urban cells in neighborhood
    urban_cells = np.count_nonzero(S_N == 1)

    # edge growth requires an urban center
    if urban_cells >= size_urban_center:

        # calculate probability of urbanization by edge growth
        p_C_urbanization = k_spread * p_C_urbanization_slopeResistance

        # evaluate if cell is urbanized
        rand_num = np.random.uniform()
        if rand_num <= p_C_urbanization:
            S_C = 1

            # add color
            c_C = color_edge    # np.array([227, 114, 34]) / 255.  # Orange
            counter = 1

    return S_C, c_C, counter


#
def rule_road_influenced_growth(S_N, l_N, e_N, r_N,
                                k_roadGravity,
                                p_C_urbanization_slopeResistance, c_N,
                                size_influence_road=8):
    """
	Defines the growth that is influenced by the existing infrastructure.
	This growth propagates the new urban centers created in the same
	time step. If the Neighborhood is intersected by a road, the non-urban
	cells with the shortest distance to the road are urbanized.
	Adaptation. Not exactly as in SLEUTH model.
	
	args:
		S_N						Current State of Neighborhood S_N(t)
		r_N						Roads in Neighborhood
		k_roadGravity			Road gravity coefficient
		size_influence_road		Number of potential urban cells with this rule 
		
	returns:
		S_N				 	  <np.array>	State of adjacent cells in next step S_N(t+1)
	"""
    counter = 0

    # extract roads in Neighborhood
    road_ix = np.argwhere(r_N == 1)

    # check if there is at least one road cell
    if road_ix.size > 0:

        # Dictionary of cell coords in Neighborhood
        coords = {0: (0, 0),
                  1: (0, 1),
                  2: (0, 2),
                  3: (1, 0),
                  4: (1, 2),
                  5: (2, 0),
                  6: (2, 1),
                  7: (2, 2)}

        p_N_urbanization_roadGravity = np.array([-1. for n in range(S_N.size)])

        # for available cells, i.e.:
        # 	land_use = 1
        #	exclude  = 0
        #	urban	 = 0
        for c in range(S_N.size):
            if (l_N[c] == 1) & (e_N[c] == 0) & (S_N[c] == 0):
                # calculate minimum distance to road
                roads = [coords[road_ix[r][0]] for r in range(len(road_ix))]
                distance_to_road = np.argmin(spatial.distance.cdist(np.array([coords[c]]), roads))

                # calculate probability of urbanization due to road gravity and local slope

                p_C_urbanization_roadGravity = calculate_p_C_urbanization_roadGravity(k_roadGravity,
                                                                                      distance_to_road)
                p_N_urbanization_roadGravity[c] = p_C_urbanization_roadGravity * p_C_urbanization_slopeResistance

        # sort non-urban cells per p_C_urbanization
        sorted = np.sort(p_N_urbanization_roadGravity)[::-1]
        index_sorted = np.argsort(p_N_urbanization_roadGravity)[::-1]

        # attempt to urbanize the cells with the highest probability of urbanization
        for iii in range(int(np.ceil(size_influence_road * k_roadGravity))):

            # evaluate if cell is urbanized
            rand_num = np.random.uniform()
            if rand_num <= sorted[iii]:
                # urbanize cell in S_N array
                S_N[index_sorted[iii]] = 1

                # add color
                c_N[index_sorted[iii]] = color_road     # np.array([162, 173, 0]) / 255.  # Green
                counter += 1

    return S_N, c_N, counter

#
def execute_urban_growth_rules(S_C,
                               p_G_urbanization_dispersion, p_C_urbanization_slopeResistance,
                               S_N, l_N, e_N, r_N,
                               p_N_urbanization,
                               k,
                               c_C, c_N, urbdist):
    '''
    Executes the urban growth rules

    :param S_C:
    :param p_G_urbanization_dispersion:
    :param p_C_urbanization_slopeResistance:
    :param S_N:
    :param l_N:
    :param e_N:
    :param r_N:
    :param p_N_urbanization:
    :param k:
    :param c_C:
    :param c_N:
    :param urbdist: distance to the initial urban seed
    :return:
    '''

    # SLEUTH growth coefficients
    k_dispersion = k[0]
    k_breed = k[1]
    k_spread = k[2]
    k_slopeResistance = k[3]
    k_roadGravity = k[4]


    # SLEUTH growth rules
    ### Spontaneous growth
    # evaluate if cell is spontaneously urbanized
    S_C, c_C, n_spont = rule_spontaneous_growth(S_C, p_G_urbanization_dispersion, p_C_urbanization_slopeResistance, c_C, urbdist)

    ### New spreading center
    # if cell was urbanized, evaluate if it becomes a new spreading center
    if S_C == 1:
        S_N, c_N, n_spread = rule_new_spreading_center(k_breed, S_N, l_N, e_N, p_N_urbanization, c_N, size_spreading_center=2)
    else:
        n_spread = 0

    ### Edge growth
    # if cell is not urbanized, evaluate if it becomes urban by organic growth
    S_C, c_C, n_edge = rule_edge_growth(S_C, k_spread, p_C_urbanization_slopeResistance, S_N, c_C, size_urban_center=3)

    ### Road influenced growth
    # if cell was urbanized, evaluate if additional urban cells are generated due to
    # existing transportation infrastructure
    if S_C == 1:
        S_N, c_N, n_road = rule_road_influenced_growth(S_N, l_N, e_N, r_N, k_roadGravity, p_C_urbanization_slopeResistance, c_N)
    else:
        n_road = 0

    return S_C, S_N, c_C, c_N, n_spont, n_spread, n_edge, n_road
