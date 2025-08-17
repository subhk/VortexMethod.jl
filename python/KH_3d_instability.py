"""
This script simulates KH instability using point vortex method

"""

import sys
import os
import numpy as np
import scipy.io as sio
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from matplotlib.colors import LightSource
from scipy.spatial import ConvexHull

from init_3d import _init_meshing_,  _init_eleVorStrgth_, _init_nodeCirculation_
from init_3d import _finding_boundary_elements_, _finding_boundary_nodes_, _tri_coord_, _tri_coord_mod_
from timestep_3d import _Sim_Xpv_gRidp_, _Sim_Xpv_gRidp_AB_, _update_Circulation, _RK2_timestep_, _RK2h_timestep_

from utility_3d_paral import _left_edge_BndryPts_, _cal_Normal_Vector, _SmoothGrid_Generation_
from utility_3d_paral import _allTriangle_Area_, _allTriangle_Centroid_, _test_
from utility_3d_paral import _Enforce_Predic_Cond_on_Vel_, _Enforce_Predic_Cond_on_Gma_, _Enforce_Predic_Cond_on_nodeCoord_

from vor3d_paral_mod_a import _node_Circulation, _cal_nodeVelocity_frm_gridVelocity, _cal_eleVorStrgth_frm_nodeCirculation

from remeshing_3d import _max_edge_length_dection_, _min_edge_length_dection_
from remeshing_3d import _element_merging_below_critical_length_ , _element_splitting_above_critical_length_
from remeshing_3d import _repositioning_boundary_eles_, _construct_elements_in_X_, _construct_elements_in_Y_ 


def print_function_(flag, Nx, Ny, Atg, Max_Iter, dt, ds_max, ds_min, _no_tri_, Lx, Ly):
	
	if flag:
		print('x----------------------------------------------------------x')
		print('         Initialising the 3D vortex sheet method            ')
		print('x----------------------------------------------------------x')
		print('x==========================================================x')
		print('          (A)  Details of the simulation parameters         ')
		print('x==========================================================x')
		print('1. Size of the computational domain [Lx, Ly] - [', Lx, '|', Ly, ']')
		print('2. No of point vortices in x-direction - ', Nx)
		print('3. No of point vortices in y-direction - ', Ny)
		print('4. Atwood number x gravitational acceleration - ', Atg)
		print('5. Maximum simulation iterations - ', Max_Iter)
		print('6. Time step (\delta t) of the simulation - ', dt)
		print('x==========================================================x')
		print('          (B)  Details of the (local) elements remeshing    ')
		print('x==========================================================x')
		print('7. Total number of triangle elements - '  , _no_tri_)
		print('8. Maximum distance for node splitting - ', ds_max)
		print('9. Minimum distance for node merging   - ', ds_min)
		print('x==========================================================x')
		print('              Time stepping of simulation starts           ')
		print('x==========================================================x')
	
	
def _Initialisation_(Nx_vor, Ny_vor):
	gra = 1.
	delta = 0.1
	alpha = 0.
	
	threshold = 1.5
	Ampl = 1.e-2
	delta_ = 0.05
	
	x = np.zeros(Nx_vor)
	y = np.zeros(Ny_vor)
	
#	x = np.cos (np.pi * np.arange(Nx_vor + 1) / Nx_vor)
#	y = np.cos (np.pi * np.arange(Ny_vor + 1) / Ny_vor)
	
	x = np.linspace(0., 1., Nx_vor)
	y = np.linspace(0., 1., Ny_vor)
	
	return gra, delta, alpha, threshold, Ampl, x, y, delta_
	

def _sign_(a, b):
	
	if np.sign(b) == np.sign(a): sgn = np.sign(a)
	if np.sign(b) != np.sign(a): sgn = np.sign(a)
	
	return sgn

def _init_field_creation( _imax_, _jmax_, _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, At, dt ):

	# co-ordinate of each element
	_triXC, _triYC, _triZC = _tri_coord_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
	
	_leng_ = len(_nodeXC)
	_nodeCoord = np.zeros( (_leng_,3) )
	( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
	
#	tri_C = _test_()
	
# --------------------------------------------------------------------
	# initialize Vortex Strength at each element
	_eleGma_ = _init_eleVorStrgth_(_no_tri_, _triXC) 
	
	# solve for node velocity
#	_uXg_, _vYg_, _wZg_, _ = \
#	_cal_nodeVelocity_frm_gridVelocity(_eleGma_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
	
	# move mesh nodes
#	_nodeXC, _nodeYC, _nodeZC = _Sim_Xpv_gRidp_(_nodeXC, _nodeYC, _nodeZC, _uXg_, _vYg_, _wZg_, dt)
	
	#calculate circulation plus adding baroclinc part
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma_)
	_nodeCirc = Obj_._change_in_Circu()
	# (baroclinic part)
#	_nodeCirc = \
#	_update_Circulation(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _nodeCirc, _no_tri_, tri_verti_, At, dt)
	
	_eleGma_[:,0], _eleGma_[:,1], _eleGma_[:,2] = \
	_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC )
# ---------------------------------------------------------------------
	
	return _nodeXC, _nodeYC, _nodeZC, _nodeCirc, _eleGma_


def _Simulation_(Nx_vor, Ny_vor, x, y, At):

	# initialization
	gra, _, _, _, _, x, y, delta_  = _Initialisation_(Nx_vor, Ny_vor)
	
	xg, yg, zg = _SmoothGrid_Generation_()
	del_split = max( abs( xg[1] - xg[0] ), abs( yg[1] - yg[0] ) )
	
	ds_max = 0.80*del_split
	ds_min = 0.05*del_split
	
	_no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC = _init_meshing_(x, y)
	
	_triXC, _triYC, _triZC = _tri_coord_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
	
	_ele_lt, _ele_rt, _ele_dn, _ele_up = \
	_finding_boundary_elements_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, min(x), max(x), min(y), max(y))
	
	_bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up = \
	_finding_boundary_nodes_(_nodeXC, _nodeYC, xmin=min(x), xmax=max(x), ymin=min(y), ymax=max(y))
	
	# left boundary points
	_LeftEdge_BndryPts_ = _left_edge_BndryPts_(_nodeXC, Nx_vor)
	
	euler_dt = 1.e-4 
	_nodeXC, _nodeYC, _nodeZC, _nodeCirc, _eleGma = \
	_init_field_creation(Nx_vor, Ny_vor, _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, At, euler_dt)
	
	length = len(_LeftEdge_BndryPts_)
	xp = np.zeros(length)
	for itr in range(length): xp[itr] = _LeftEdge_BndryPts_[itr]
	
	# time iteration
	time = 0. #euler_dt
	Max_Iter = 1000
	dt = 0.01     # time step
	
#	KE3d = np.zeros(Max_Iter)
#	KE3d[0] = D3_gridKE
	flag = True
	
	print_function_(True, Nx_vor, Ny_vor, At, Max_Iter, dt, ds_max, ds_min, _no_tri_, max(x), max(y))
	
	# !!!!! start time-stepping
	for Iter in range(Max_Iter):
		
		_eleGma, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, dt = \
		_RK2_timestep_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGma, _no_tri_, tri_verti_, At)
		
		time += dt
		
		print('x-------------------------------------------------------------------------x')
		print('Iter : ', Iter,  '| time : ', time, '| dt : ', dt, '|', 'no_tri : ', _no_tri_, '|' )
		print('No of Lagrangian nodes at the boundary - [', len(_bndry_node_lt), '|', len(_bndry_node_rt), '|', len(_bndry_node_dn), '|', len(_bndry_node_up), ']' )
		print( 'max gamma-y = ', np.max(_eleGma[:,1]), '|', 'min gamma-y = ', np.min(_eleGma[:,1]) )
		
		_no_tri_ = np.shape(tri_verti_)[0] 
		
#		_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC = _repositioning_boundary_eles_\
#		(_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up, 1., 1.)
		
# --------------------------
		_ele_max_edge_, _max_edge_ = _max_edge_length_dection_( _triXC, _triYC, _triZC, ds_max )
		if _ele_max_edge_ >= 0:
			print( '---->>> triangles found to have max edge length  ' )
			_Triangle_Area_ = _allTriangle_Area_( _triXC, _triYC, _triZC )
			
			_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _eleGma, \
			_bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up = \
			_element_splitting_above_critical_length_\
			(_triXC, _triYC, _triZC, _ele_max_edge_, _max_edge_, _nodeXC, _nodeYC, _nodeZC, _Triangle_Area_, _eleGma, ds_max, \
			_bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up)
			
			tri_verti_ = tri_verti_.astype(int) 
			_no_tri_ = np.shape(tri_verti_)[0]
			print( 'no of triangle after node merging - ', len(_triXC[:,0]) )
			
#		_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC = _repositioning_boundary_eles_\
#		(_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up, 1., 1.)
		
# --------------------------
		_ele_min_edge_, _min_edge_ = _min_edge_length_dection_( _triXC, _triYC, _triZC, ds_min )
		if _ele_min_edge_ >= 0:
			print( '---->>> triangles found to have min edge length  ' )
			_Triangle_Area_ = _allTriangle_Area_( _triXC, _triYC, _triZC )
			
			_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _eleGma, \
			_bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up = \
			_element_merging_below_critical_length_\
			(_triXC, _triYC, _triZC, _ele_min_edge_, _min_edge_, _nodeXC, _nodeYC, _nodeZC, _Triangle_Area_, _eleGma, ds_min, \
			_bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up)
			
			tri_verti_ = tri_verti_.astype(int) 
			_no_tri_ = np.shape(tri_verti_)[0]
			print( 'no of triangle after scrapping the garbage element(s) - ', len(_triXC[:,0]) )
		
#		_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC = _repositioning_boundary_eles_\
#		(_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up, 1., 1.)
			
		_triXC, _triYC, _triZC, tri_verti_, _eleGma, _nodeXC, _nodeYC, _nodeZC, flagx = \
		_construct_elements_in_X_(_triXC, _triYC, _triZC, _eleGma, _nodeXC, _nodeYC, _nodeZC, 1., 2.0*ds_max, _ele_lt, _ele_rt)
		
		_triXC, _triYC, _triZC, tri_verti_, _eleGma, _nodeXC, _nodeYC, _nodeZC, flagy = \
		_construct_elements_in_Y_(_triXC, _triYC, _triZC, _eleGma, _nodeXC, _nodeYC, _nodeZC, 1., 2.0*ds_max, _ele_dn, _ele_up)
		
#		points = np.column_stack( (_nodeXC, _nodeYC) )
#		hull = ConvexHull(points)
#		iteR = 0
#		for simplex in hull.simplices:
#			print(points[simplex, 0])
#			print(points[simplex, 1])
#			iteR += 1
#		
#		print('iteR = ', iteR)
#		
#		if flagx:
#			points = np.column_stack( (_nodeXC, _nodeYC) )
#			hull = ConvexHull(points)
#		
#		if flagy:
#			points = np.column_stack( (_nodeXC, _nodeYC) )
#			hull = ConvexHull(points)
			
			
			#_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC = _repositioning_boundary_eles_\
			#(_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up, 1., 1.)
		
######	
#		KE3d[Iter] = D3_gridKE
		
		print( 'max._nodeXC = ', np.max(_nodeXC), '| min._nodeXC = ', np.min(_nodeXC) )
		print( 'max._nodeYC = ', np.max(_nodeYC), '| min._nodeYC = ', np.min(_nodeYC) )
		print( 'max._nodeZC = ', np.max(_nodeZC), '| min._nodeZC = ', np.min(_nodeZC) )
		print('x-------------------------------------------------------------------------x')
		
		if Iter%2 == 0:
			sio.savemat( 'nodex%s.mat' %Iter, {'x':_nodeXC} )
			sio.savemat( 'nodey%s.mat' %Iter, {'y':_nodeYC} )
			sio.savemat( 'nodez%s.mat' %Iter, {'z':_nodeZC} )
			sio.savemat( 'tri%s.mat'  %Iter, {'tri':tri_verti_} )
#			sio.savemat( 'ux%s.mat' %Iter, {'ux':_uXg_} )
#			sio.savemat( 'uy%s.mat' %Iter, {'uy':_vYg_} )
#			sio.savemat( 'uz%s.mat' %Iter, {'uz':_wZg_} )
#			sio.savemat( 'trix%s.mat' %Iter, {'tx':_triXC} )
#			sio.savemat( 'triy%s.mat' %Iter, {'ty':_triYC} )
#			sio.savemat( 'triz%s.mat' %Iter, {'tz':_triZC} )
#			sio.savemat( 'eleGma%s.mat' %Iter, {'gma':_eleGma} )
#			sio.savemat('KE3d.mat', {'ke3d':KE3d} )
		

if __name__ == '__main__':
	Nx_vor = 100            # No of Lagrangian nodes in x-direction
	Ny_vor = 100            # No of Lagrangian nodes in y-direction
	
	# Atwood number*gravity
	Atg = 0.
	
	gra, delta, alpha, threshold, Ampl, x, y, delta_ = _Initialisation_(Nx_vor, Ny_vor)
	_Simulation_(Nx_vor, Ny_vor, x, y, Atg)
	
	
	
