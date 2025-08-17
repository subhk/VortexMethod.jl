"""
This script simulates KH instability using point vortex method

"""

import sys
import os
import numpy as np
#from mpi4py import MPI
import scipy.io as sio
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from matplotlib.colors import LightSource

from init_3d import _init_meshing_,  _init_eleVorStrgth_, _tri_coord_, _init_nodeCirculation_
from init_3d import _finding_boundary_elements_, _finding_boundary_nodes_
from timestep_3d import  _Sim_Gma_gRidp_, _Sim_Upv_gRidp_, _Sim_Xpv_gRidp_, _Sim_Xpv_gRidp_AB_, _update_Circulation

from utility_3d_paral import _left_edge_BndryPts_, _cal_Normal_Vector
from utility_3d_paral import _allTriangle_Area_, _allTriangle_Centroid_, _test_
from utility_3d_paral import _Enforce_Predic_Cond_on_Vel_, _Enforce_Predic_Cond_on_Gma_, _Enforce_Predic_Cond_on_nodeCoord_


#from remeshing_3d import _elements_splitting_
from vor3d_paral_mod_a import _node_Circulation, _cal_nodeVelocity_frm_gridVelocity, _cal_eleVorStrgth_frm_nodeCirculation

#_cal_nodeVoricity_frm_eleVorStrgth_ #, 

#from vel3d_paral_mod_a import Vel3d_


def _Initialisation_(Nx_vor, Ny_vor):
	gra = 1.
	delta = 0.1
	alpha = 0.
	
	threshold = 1.5
	Ampl = 1.e-2
	delta_ = 0.05
	
	x = np.zeros(Nx_vor)
	y = np.zeros(Ny_vor)
	
	x = np.linspace(0., 1.0, Nx_vor)
	y = np.linspace(0., 1.0, Ny_vor)
	
	return gra, delta, alpha, threshold, Ampl, x, y, delta_
	

def _sign_(a, b):
	
	if np.sign(b) == np.sign(a): sgn = np.sign(a)
	if np.sign(b) != np.sign(a): sgn = np.sign(a)
	
	return sgn

def _init_field_creation( _imax_, _jmax_, _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, del_split ):

	# co-ordinate of each element
	_triXC, _triYC, _triZC = _tri_coord_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
	_Triangle_Area_ = _allTriangle_Area_( _triXC, _triYC, _triZC, _no_tri_, tri_verti_ )
	_Centroid_ = _allTriangle_Centroid_( _triXC, _triYC, _triZC, _no_tri_, tri_verti_ )
	
	#sio.savemat('xT.mat', {'xT':_triXC})
	#sio.savemat('yT.mat', {'yT':_triYC})
	#sio.savemat('zT.mat', {'zT':_triZC})
	
	_leng_ = len(_nodeXC)
	_nodeCoord = np.zeros( (_leng_,3) )
	_nodeCoord[:,0] = _nodeXC
	_nodeCoord[:,1] = _nodeYC
	_nodeCoord[:,2] = _nodeYC
	
	tri_C = _test_()
# --------------------------------------------------------------------
	# initialize circulation at each node of triangle
#	_nodeCirc  = _init_nodeCirculation_(_triXC, tri_verti_, _nodeXC)
	

	# calculating Vortex Strength at each element
#	_eleGmaX_, _eleGmaY_, _eleGmaZ_ = \
#	_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _triXC, _triYC, _triZC )
	
#	_leng_ = len(_triXC[:,0])
#	_eleGma = np.zeros( (_leng_,3) )
#	_eleGma[:,0] = _eleGmaX_
#	_eleGma[:,1] = _eleGmaY_
#	_eleGma[:,2] = _eleGmaZ_
# --------------------------------------------------------------------
	
# --------------------------------------------------------------------
#	# initialize Vortex Strength at each element
	_eleGma_ = _init_eleVorStrgth_(_no_tri_, _triXC) 
	
	#initialize circulation at each node 	
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma_, del_split)
	_nodeCirc = Obj_._change_in_Circu()
	
	( a0, a1, a2 ) = ( _nodeCirc[0,0], _nodeCirc[0,1], _nodeCirc[0,2] )
	print(a0, a1, a2)
# ----------------------------------------------------------------------	
	
	_uXg_, _vYg_, _wZg_, D3_gridKE = \
	_cal_nodeVelocity_frm_gridVelocity(_eleGma_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
	
	return _uXg_, _vYg_, _wZg_, D3_gridKE, _eleGma_, _nodeCirc


def _Simulation_(Nx_vor, Ny_vor, x, y, At):

	# initialization
	gra, _, _, _, _, x, y, delta_  = _Initialisation_(Nx_vor, Ny_vor)
	
	dx_t0 = abs( x[1] - x[0] )
	dy_t0 = abs( y[1] - y[0] )
	
	_no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC = _init_meshing_(x, y)
		
	_bndry_ele_lt, _bndry_ele_rt, _bndry_ele_dn, _bndry_ele_up = \
	_finding_boundary_elements_(_nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, xmin=min(x), xmax=max(x), ymin=min(y), ymax=max(y))
	
	_bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up = \
	_finding_boundary_nodes_(_nodeXC, _nodeYC, xmin=min(x), xmax=max(x), ymin=min(y), ymax=max(y))
	
#	sio.savemat( 'west.mat',  {'west':_bndry_node_lt} )
#	sio.savemat( 'east.mat',  {'east':_bndry_node_rt} )
#	sio.savemat( 'south.mat', {'south':_bndry_node_dn} )
#	sio.savemat( 'north.mat', {'north':_bndry_node_up} )
	
	# calculating the unit normal vector of each triangle
	Obj_ = _cal_Normal_Vector(_nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_)
	_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()	
	_edge_len_ = Obj_._edge_length_()
	_del_split = 2.5*_edge_len_
	 
	
	# left boundary points
	_LeftEdge_BndryPts_ = _left_edge_BndryPts_(_nodeXC, Nx_vor)
	
	_uXg_, _vYg_, _wZg_, D3_gridKE, _eleGma, _nodeCirc = \
	_init_field_creation(Nx_vor, Ny_vor, _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, _del_split)
	
	length = len(_LeftEdge_BndryPts_)
	xp = np.zeros(length)
	for itr in range(length): xp[itr] = _LeftEdge_BndryPts_[itr]
	
	sio.savemat('xleft.mat', {'xp':xp})
	
	# time iteration
	Max_Iter = 500
	
	KE3d = np.zeros(Max_Iter)
	KE3d[0] = D3_gridKE
	
	#fig = plt.figure()
	#ax = fig.add_subplot(1, 1, 1, projection='3d')
	#ax.plot_trisurf(_nodeXC, _nodeYC, _uXg_,  triangles=tri_verti_, lw=0.2, edgecolor="black", color="grey", alpha=0.5 ) 
	#plt.show() 
	
	time = 0.
	dt = 0.0005     # time step	
	flag = True
	
	# !!!!! start time-stepping
	for Iter in range(Max_Iter):
		time += dt
		print('Iter = ', Iter,  'time = ', time, 'dt = ', dt)
				
	#	if flag:
	#		Obj_ = _elements_splitting_(_nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, _del_split)
	#		_node_share_cmn_eles_, _leng_ = Obj_._cal_common_elements_each_node_(flag)
			#print(_node_share_cmn_eles_)
	#		flag = False
		
		if Iter	== 0:
			_nodeXC, _nodeYC, _nodeZC = _Sim_Xpv_gRidp_(_nodeXC, _nodeYC, _nodeZC, _uXg_, _vYg_, _wZg_, dt)
		else:
			_nodeXC, _nodeYC, _nodeZC = _Sim_Xpv_gRidp_AB_(_nodeXC, _nodeYC, _nodeZC, _uXg_, _vYg_, _wZg_, uxold, vyold, wzold, dt)
		
#		_nodeXC, _nodeYC, _nodeZC = \
#		_Enforce_Predic_Cond_on_nodeCoord_(_nodeXC, _nodeYC, _nodeZC, Nx_vor, Ny_vor, _LeftEdge_BndryPts_, 1.0, 1.0)
		
		_nodeXC, _nodeYC, _nodeZC = \
		_Enforce_Predic_Cond_on_nodeCoord_(_nodeXC, _nodeYC, _nodeZC, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up, 1.0, 1.0)
		print('Step 1 is done')

		_triXC, _triYC, _triZC = _tri_coord_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
		_Triangle_Area_ = _allTriangle_Area_( _triXC, _triYC, _triZC, _no_tri_, tri_verti_ )
		_Centroid_ = _allTriangle_Centroid_( _triXC, _triYC, _triZC, _no_tri_, tri_verti_ )
		
		_nodeCirc = \
		_update_Circulation(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _nodeCirc, _no_tri_, tri_verti_, At, dt, _del_split)
		print('Step 2 is done')
		
		_eleGma[:,0], _eleGma[:,1], _eleGma[:,2] = \
		_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _del_split )
		
#		_eleGma[:,0], _eleGma[:,1], _eleGma[:,2] = \
#		_Enforce_Predic_Cond_on_Gma_(_eleGma[:,0], _eleGma[:,1], _eleGma[:,2], _bndry_ele_lt, _bndry_ele_rt, _bndry_ele_dn, _bndry_ele_up)
		print('Step 3 is done')
		
		_uXg_, _vYg_, _wZg_, D3_gridKE = \
		_cal_nodeVelocity_frm_gridVelocity( _eleGma, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC )
		
# no periodic boundary conditions ->
#		_uXg_, _vYg_, _wZg_ = \
#		_Enforce_Predic_Cond_on_Vel_( _uXg_, _vYg_, _wZg_, Nx_vor, Ny_vor, _LeftEdge_BndryPts_ )
		
#		_uXg_, _vYg_, _wZg_ = \
#		_Enforce_Predic_Cond_on_Vel_( _uXg_, _vYg_, _wZg_, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up )
		print('Step 4 is done')	
		
		uxold = np.copy(_uXg_)
		vyold = np.copy(_vYg_)
		wzold = np.copy(_wZg_)
		
		KE3d[Iter] = D3_gridKE
		
		print( 'max._nodeZC = ', np.max(_nodeZC) )	
		
		if Iter%5 == 0:
			sio.savemat('nodex%s.mat' %Iter, {'x':_nodeXC})
			sio.savemat('nodey%s.mat' %Iter, {'y':_nodeYC})
			sio.savemat('nodez%s.mat' %Iter, {'z':_nodeZC})
			sio.savemat('ux.mat', {'ux':_uXg_})
			sio.savemat('uz.mat', {'uz':_wZg_})
			#sio.savemat('gy.mat', {'gy':_eleGma[:,1]})
			sio.savemat('tri.mat', {'tri':tri_verti_})
			sio.savemat('KE3d.mat', {'ke3d':KE3d} )
		

if __name__ == '__main__':
	Nx_vor = 200           # No of Lagrangian nodes in x-direction
	Ny_vor = 40           # No of Lagrangian nodes in y-direction
	
	#Lx = 1.         # length of the domain ( useless )
	#Ly = 0.5
	
	# Atwood number
	At = 0.3
	
	gra, delta, alpha, threshold, Ampl, x, y, delta_ = _Initialisation_(Nx_vor, Ny_vor)
	_Simulation_(Nx_vor, Ny_vor, x, y, At)
	
	
	
