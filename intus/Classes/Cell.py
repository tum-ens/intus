"""
Cell.py
intus
AMC @ TUM ENS
"""

import numpy as np
import intus.Functions as intus


class Cell():
# --------------------------------------------------------------------------------
	def __init__(self, slope, land_use, excluded, urban, transport, value, se_level,
				k, p_G_urbanization_dispersion, urbdist):
		
		# City
		# ------------------------------------------------------------------------
		# Urban growth coefficients
		self.k									= k								# urban growth coefficients
			
		# Cell
		# ------------------------------------------------------------------------
		# Input attributes
		self.slope								= slope
		self.land_use							= land_use
		self.excluded							= excluded
		self.S_C								= urban
		self.transport							= transport
		self.value								= value
		self.se_level							= se_level
		self.urbdist 							= urbdist

		# Probabilities of urbanization
		self.p_G_urbanization_dispersion 		= p_G_urbanization_dispersion	# for spontaneous growth
		self.p_C_urbanization_slopeResistance 	= intus.calculate_p_C_urbanization_slopeResistance(k[3], slope)		
		
		# Output
		self.initialize_cell_colors()



	def calculate_new_state(self, S_N, l_N, e_N, r_N, v_N, p_N_urbanization, c_N):
		"""
		Determines a new the state of the cell based on the urban growth transition
		rules. 
		
		returns:
			self.S_C				S_C(t+1)
			self.color_growth		cell color [R G B]
		"""
		
		# execute urban growth rules
		self.S_C, S_N, self.c_C, c_N, n_spont, n_spread, n_edge, n_road = intus.execute_urban_growth_rules(self.S_C,
												self.p_G_urbanization_dispersion, self.p_C_urbanization_slopeResistance,
												S_N, l_N, e_N, r_N, p_N_urbanization, self.k, self.c_C, c_N, self.urbdist)
		return self.S_C, S_N, self.c_C, c_N, n_spont, n_spread, n_edge, n_road





	#
	def initialize_cell_colors(self):
		"""
		Assign colors for all cells according to initial conditions
		"""
		
		# start with white cells
		self.c_C = np.array([1., 1., 1.])

		if self.land_use  != 1: 	# not suitable for urbanization
			self.c_C = np.array([0.8, 0.8, 0.8])	# bright gray

		if self.excluded  != 0: 	# excluded area
			self.c_C = np.array([0.85, 0.85, 0.85])	# bright gray

		if self.transport != 0: 	# cell with roads
			self.c_C = np.array([0., 0., 0.])		# black
		
		if self.S_C 	  != 0: 	# urban seed
			self.c_C = np.array([0., 0., 0.])		# black