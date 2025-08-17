#import sys
#import os
import numpy as np

import numpy as np
#import matplotlib.cm as cm
from scipy.interpolate import Rbf

from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

#from timestep_3d import _Sim_Upv_gRidp_
#from utility_3d_paral import _Triangle_Area_


#def _init_velocity_(_xg, _yg, _zg, _GmaXg_, _GmaYg_, _GmaZg_, delta_):
#
#	_uxg_, _vyg_, _wzg_ = _Sim_Upv_gRidp_(_xg, _yg, _zg, _GmaXg_, _GmaYg_, _GmaZg_, delta_)		
#		
#	return _uxg_, _vyg_, _wzg_
	

def _init_eleVorStrgth_(_no_tri_, _triXC):
		
	_eleGma_ = np.zeros( (_no_tri_,3) )

	for _tri in range(_no_tri_):
		_eleGma_[_tri,0] = 0.0 
		_eleGma_[_tri,1] = 1.0 #+ 0.1*np.sin(2.*np.pi*_triXC[_tri,1]) 
		_eleGma_[_tri,2] = 0.0 

	return _eleGma_
	
	
def _init_nodeCirculation_(_triXC, tri_verti_, _nodeXC):

	_tmp_ = np.zeros(len(_nodeXC))
	
	for itr in range(len(_nodeXC)):
		_tmp_[itr] = 0.01*np.sin(2.*np.pi*_nodeXC[itr])
		

	_no_tri_ = len(_triXC[:,0])	
	_Circ = np.zeros( (_no_tri_,3) )
	
	for _tri in range(_no_tri_):
		
		_Circ[_tri,0] = _tmp_[tri_verti_[_tri,0]]
		_Circ[_tri,1] = 0. 
		_Circ[_tri,2] = -_tmp_[tri_verti_[_tri,0]]
	
	return _Circ
		

def _init_meshing_(x, y):
	
	x, y = np.meshgrid(x, y)
	x = x.flatten()
	y = y.flatten()
	
	#evaluate the parameterization at the flattened x and y
	_xgrid = x + 0.01*np.sin(2.*np.pi*x)
	_ygrid = y
	_zgrid = 0.01*np.sin(2.*np.pi*x) + 0.001*np.sin(4.*np.pi*y) #+ 0.01*np.sin(4.*np.pi*_ygrid)
	
	#print(np.shape(_zgrid))
	
	#define 2D points, as input data for the Delaunay triangulation 
	points2D = np.vstack([_xgrid,_ygrid]).T
	tri = Delaunay(points2D)
	tri_vertices_ = tri.simplices  # vertices of each triangle
	_no_triangle_ = np.shape(tri_vertices_)[0]
	print('no of elements  = ', _no_triangle_)
	
	#print('length = ', len(_xgrid))
	
	hull = ConvexHull(points2D)
	boundaryList = hull.vertices
	#print(np.shape(boundaryList))
	
	# store the boundary points
	#boundary = (points2D[tri.convex_hull]).flatten()
	#bp_x = boundary[0:-3:3]
	#bp_y = boundary[1:-2:3]
	#bp_z = boundary[2:-1:3]
	
	#tst = tri.find_simplex(p)
	#print('tst = ', tst)
	
	#sio.savemat('boundary.mat', {'b':boundary})
	
	#points3D = np.vstack((x,y,z)).T
	#tri_vertices = map(lambda index: points3D[index], simplices)
	#zmean = [np.mean(tri[:,2]) for tri in tri_vertices ]
	
	#min_zmean = np.min(zmean)
	#max_zmean = np.max(zmean)
	#print(min_zmean)
	#print(max_zmean)
	#I, J, K = tri_indices(simplices)
	#lists_coord=[[[T[k%3][c] for k in range(4)]+[ None]   for T in tri_vertices]  for c in range(3)]
	
	return _no_triangle_, tri_vertices_, _xgrid, _ygrid, _zgrid #, bp_x, bp_y, bp_z


def tri_indices(simplices):
	#simplices is a numpy array defining the simplices of the triangularization
	#returns the lists of indices i, j, k
	
	return ([triplet[c] for triplet in simplices] for c in range(3))	
			

def _tri_coord_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ ):

	_triXC = np.zeros( (_no_tri_,3) )
	_triYC = np.zeros( (_no_tri_,3) )
	_triZC = np.zeros( (_no_tri_,3) )
	
	_triXC[0:_no_tri_,0:3] = _nodeXC[tri_verti_[0:_no_tri_,0:3]]
	_triYC[0:_no_tri_,0:3] = _nodeYC[tri_verti_[0:_no_tri_,0:3]]
	_triZC[0:_no_tri_,0:3] = _nodeZC[tri_verti_[0:_no_tri_,0:3]]
	
	return _triXC, _triYC, _triZC
	
	
def _tri_coord_mod_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ ):

	_triXC = np.zeros( (_no_tri_,3) )
	_triYC = np.zeros( (_no_tri_,3) )
	_triZC = np.zeros( (_no_tri_,3) )
	
	for itr in range(_no_tri_):
		( _triXC[itr,0], _triXC[itr,1], _triXC[itr,2] ) = \
		( _nodeXC[tri_verti_[itr,0]], _nodeXC[tri_verti_[itr,1]], _nodeXC[tri_verti_[itr,2]] )
		
		( _triYC[itr,0], _triYC[itr,1], _triYC[itr,2] ) = \
		( _nodeYC[tri_verti_[itr,0]], _nodeYC[tri_verti_[itr,1]], _nodeYC[tri_verti_[itr,2]] )
		
		( _triZC[itr,0], _triZC[itr,1], _triZC[itr,2] ) = \
		( _nodeZC[tri_verti_[itr,0]], _nodeZC[tri_verti_[itr,1]], _nodeZC[tri_verti_[itr,2]] )
		
	return _triXC, _triYC, _triZC
	
	
	
def _boundary_elements_(_Coord, _no_tri_, bundry_val, tol):
	
	_bndry_ele_ = []
	
	for itr in range(_no_tri_):
		flag = True
		
		if _Coord[itr,0] < bundry_val + tol and _Coord[itr,0] > bundry_val - tol:
			flag = False
			_bndry_ele_.append(itr)
		
		if flag:
			if _Coord[itr,1] < bundry_val + tol and _Coord[itr,1] > bundry_val - tol:
				flag = False
				_bndry_ele_.append(itr)	
		
		if flag:
			if _Coord[itr,2] < bundry_val + tol and _Coord[itr,2] > bundry_val - tol:
				_bndry_ele_.append(itr)
	
	_bndry_ele_ = np.unique(_bndry_ele_).tolist()
	
	return _bndry_ele_
	
	
def _finding_boundary_elements_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, xmin, xmax, ymin, ymax ):

	_triXC, _triYC, _triZC = _tri_coord_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
	
	tol = 1.e-3
	
	# finding elements on left boundary -> x = xmin
	_bndry_ele_l = _boundary_elements_( _triXC, _no_tri_, xmin, tol )
				
	# finding elements on right boundary -> x = xmax 
	_bndry_ele_r = _boundary_elements_( _triXC, _no_tri_, xmax, tol )
	
    # finding elements on lower boundary -> y = ymin 	
	_bndry_ele_d = _boundary_elements_( _triYC, _no_tri_, ymin, tol )
		
	# finding elements on upper boundary -> y = ymax 
	_bndry_ele_u = _boundary_elements_( _triYC, _no_tri_, ymax, tol )
	
	return _bndry_ele_l, _bndry_ele_r, _bndry_ele_d, _bndry_ele_u
	
	
def _boundary_nodes_(_Coord, bundry_val, tol):
	
	_bndry_node_ = []
	
	for itr in range( len(_Coord) ):
		
		if _Coord[itr] < bundry_val + tol and _Coord[itr] > bundry_val - tol:
			_bndry_node_.append(itr)
	
	_bndry_node_ = np.unique(_bndry_node_).tolist()
	
	return _bndry_node_
	
	
def _finding_boundary_nodes_( _nodeXC, _nodeYC, xmin, xmax, ymin, ymax ):
	
	tol = 1.e-3
	
	# finding nodes on left boundary -> x = xmin
	_bndry_node_l = _boundary_nodes_( _nodeXC, xmin, tol )
	
	# finding nodes on right boundary -> x = xmax 
	_bndry_node_r = _boundary_nodes_( _nodeXC, xmax, tol )
	
	# finding nodes on lower boundary -> y = ymin 	
	_bndry_node_d = _boundary_nodes_( _nodeYC, ymin, tol )
		
	# finding nodes on upper boundary -> y = ymax 
	_bndry_node_u = _boundary_nodes_( _nodeYC, ymax, tol )
	
	return _bndry_node_l, _bndry_node_r, _bndry_node_d, _bndry_node_u

	
