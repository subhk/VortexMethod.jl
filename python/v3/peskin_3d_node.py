import numpy as np
from scipy import linalg
import scipy.io as sio
import multiprocessing as mp
from numpy.fft import fft, ifft, fftfreq
import unittest
import ctypes
from scipy.linalg import lstsq

from utility_3d_paral import _allTriangle_Area_, _D1_Peskin_function_, _SmoothGrid_Generation_, _allTriangle_Centroid_
from utility_3d_paral import _centroid_16subTriangle_, _centroid_4subTriangle_, Domain, _allTriangle_Area_ 

#from utility_3d_paral import _chk_GridPoint_close_boundary, _chk_GridPoint_on_boundary
#from utility_3d_paral import _chk_GridPoint_on_corner_pt_, _chk_GridPoint_on_corner_line_, _chk_GridPoint_cl_corner_line_
#from utility_3d_paral import _find_elements_nearby_3dgrid_, _chk_GridPoint_cl_wall_corner_

from utility_3d_paral import _find_elements_nearby_3dgrid_

def _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0):
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	
	for _len in range( len(_list_) ):
		
		_grid_num_ = _list_[_len]
		
		dx = coord[0] - _grid_coord_[_grid_num_,0]
		dy = coord[1] - _grid_coord_[_grid_num_,1]
		dz = coord[2] - _grid_coord_[_grid_num_,2]
		_tmp_ = ( 1. + np.cos(np.pi*dx/eps0) )*( 1. + np.cos(np.pi*dy/eps0) )*( 1. + np.cos(np.pi*dz/eps0) )/(8.*delr**3.)
		
		sum_x += _gridVelX_[_grid_num_]*_tmp_
		sum_y += _gridVelY_[_grid_num_]*_tmp_
		sum_z += _gridVelZ_[_grid_num_]*_tmp_
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	 
	# middle box
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_W_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	_grid_coord0_ = np.copy(_grid_coord_) 
	
	# western box = middle box - Lx
	_leng_ = len(_grid_coord_[:,0])
	_grid_coord0_[0:_leng_, 0] = _grid_coord_[0:_leng_, 0] - Lx
	_grid_coord0_[0:_leng_, 0][_grid_coord0_[0:_leng_, 0] == 0.] = 10.
	
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord0_ ) 
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_E_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	_grid_coord0_ = np.copy(_grid_coord_) 
	
	# eastern box = middle box + Lx
	_leng_ = len(_grid_coord_[:,0])
	_grid_coord0_[0:_leng_, 0] = _grid_coord_[0:_leng_, 0] + Lx
	_grid_coord0_[0:_leng_, 0][_grid_coord0_[0:_leng_, 0] == Lx] = 10.
	
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_S_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	_grid_coord0_ = np.copy(_grid_coord_) 
	
	# southern box = middle box - Lx
	_leng_ = len(_grid_coord_[:,0])
	_grid_coord0_[0:_leng_, 1] = _grid_coord_[0:_leng_, 1] - Ly
	_grid_coord0_[0:_leng_, 1][_grid_coord0_[0:_leng_, 1] == 0.] = 10.
		
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_N_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	_grid_coord0_ = np.copy(_grid_coord_) 
	
	# northern box = middle box + Ly
	_leng_ = len(_grid_coord_[:,0])
	_grid_coord0_[0:_leng_, 1] = _grid_coord_[0:_leng_, 1] + Ly
	_grid_coord0_[0:_leng_, 1][_grid_coord0_[0:_leng_, 1] == Ly] = 10.
	
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_WS_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	_grid_coord0_ = np.copy(_grid_coord_)
	
	# west-southern box = middle box - Lx, - Ly
	_leng_ = len(_grid_coord_[:,0])
	_grid_coord0_[0:_leng_, 0] = _grid_coord_[0:_leng_, 0] - Lx
	_grid_coord0_[0:_leng_, 1] = _grid_coord_[0:_leng_, 1] + Ly
	_grid_coord0_[0:_leng_, 0][_grid_coord0_[0:_leng_, 0] == 0.] = 10.
	
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_WN_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	_grid_coord0_ = np.copy(_grid_coord_)
	
	# west-northern box = middle box - Lx, + Ly 
	_leng_ = len(_grid_coord_[:,0])
	_grid_coord0_[0:_leng_, 0] = _grid_coord_[0:_leng_, 0] - Lx
	_grid_coord0_[0:_leng_, 1] = _grid_coord_[0:_leng_, 1] + Ly
	_grid_coord0_[0:_leng_, 0][_grid_coord0_[0:_leng_, 0] == 0.] = 10.
	
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_ES_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	_grid_coord0_ = np.copy(_grid_coord_)
	
	# east-southern box = middle box + Lx, - Ly
	_leng_ = len(_grid_coord_[:,0])
	_grid_coord0_[0:_leng_, 0] = _grid_coord_[0:_leng_, 0] + Lx
	_grid_coord0_[0:_leng_, 1] = _grid_coord_[0:_leng_, 1] - Ly
	_grid_coord0_[0:_leng_, 0][_grid_coord0_[0:_leng_, 0] == Lx] = 10.
	
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
def _D3_Peskin_function_EN_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	_grid_coord0_ = np.copy(_grid_coord_)
	
	# east-northern box = middle box + Lx, + Ly
	_leng_ = len(_grid_coord_[:,0])
	_grid_coord0_[0:_leng_, 0] = _grid_coord_[0:_leng_, 0] + Lx
	_grid_coord0_[0:_leng_, 1] = _grid_coord_[0:_leng_, 1] + Ly
	_grid_coord0_[0:_leng_, 0][_grid_coord0_[0:_leng_, 0] == Lx] = 10.
	
	_list_ = _find_elements_nearby_3dgrid_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _grid_coord0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_gridVelX_, _gridVelY_, _gridVelZ_, _grid_coord_, coord, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3b_Peskin_function_toget_nodeVel_( _gridVelX_, _gridVelY_, _gridVelZ_, _nodeC, x3d, y3d, z3d, ds ):
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	
	coord = np.array( [ _nodeC[0], _nodeC[1], _nodeC[2]  ] )
	_grid_coord_ = np.column_stack( (x3d, y3d, z3d) )
	
	delr = 4.
	eps = delr*ds[0]
	
	eps0 = np.array( [ eps, eps, eps ] )
	
	sum_x0, sum_y0, sum_z0 = _D3_Peskin_function_       (_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_E_box_ (_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_S_box_ (_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	sum_x4, sum_y4, sum_z4 = _D3_Peskin_function_N_box_ (_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	
	sum_x5, sum_y5, sum_z5 = _D3_Peskin_function_WN_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	sum_x6, sum_y6, sum_z6 = _D3_Peskin_function_EN_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	sum_x7, sum_y7, sum_z7 = _D3_Peskin_function_ES_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	sum_x8, sum_y8, sum_z8 = _D3_Peskin_function_WS_box_(_gridVelX_, _gridVelY_, _gridVelZ_, coord, _grid_coord_, ds, delr)
	
	sum_x = sum_x0 + sum_x1 + sum_x2 + sum_x3 + sum_x4 + sum_x5 + sum_x6 + sum_x7 + sum_x8  
	sum_y = sum_y0 + sum_y1 + sum_y2 + sum_y3 + sum_y4 + sum_y5 + sum_y6 + sum_y7 + sum_y8
	sum_z = sum_z0 + sum_z1 + sum_z2 + sum_z3 + sum_z4 + sum_z5 + sum_z6 + sum_z7 + sum_z8
	
	return sum_x, sum_y, sum_z
	
	
