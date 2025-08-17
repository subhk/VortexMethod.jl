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
from utility_3d_paral import _chk_GridPoint_close_boundary, _chk_GridPoint_on_boundary, _chk_GridPoint_cl_corner_line_

#from utility_3d_paral import _chk_GridPoint_on_corner_line_, 
#from utility_3d_paral import _find_elements_nearby_, _chk_GridPoint_cl_wall_corner_ #, _chk_GridPoint_on_corner_pt_,

from utility_3d_paral import _find_elements_nearby_
	
	
def _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0):
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	_length_ = _all_centroid_coord_.shape[1]
	
#	for _tri in _list_:
#		
#		dx = np.subtract( coord[0], _all_centroid_coord_[_tri,0:_length_,0] )
#		dy = np.subtract( coord[1], _all_centroid_coord_[_tri,0:_length_,1] )
#		dz = np.subtract( coord[2], _all_centroid_coord_[_tri,0:_length_,2] )
#		
#		_tmp_ = np.multiply( np.multiply( np.add(1. , np.cos(np.pi*dx/eps0)) , np.add(1., np.cos(np.pi*dy/eps0)) ), np.add(1., np.cos(np.pi*dz/eps0)) )
#		_tmp_ = np.multiply( _tmp_, delr/(8.*delr**3.) )
#		
#		_sum_ = sum(_tmp_)
#		
#		sum_x += _Area_[_tri]*_eleGma0_[_tri,0]*_sum_/_length_
#		sum_y += _Area_[_tri]*_eleGma0_[_tri,1]*_sum_/_length_
#		sum_z += _Area_[_tri]*_eleGma0_[_tri,2]*_sum_/_length_	
	
	for _len in range( len(_list_) ):
		
		_tri = _list_[_len]
		_length_ = _all_centroid_coord_.shape[1]
		_sum_ = 0.
		
		for _sub_tri in range(_length_):
			
			dx = coord[0] - _all_centroid_coord_[_tri,_sub_tri,0]
			dy = coord[1] - _all_centroid_coord_[_tri,_sub_tri,1]
			dz = coord[2] - _all_centroid_coord_[_tri,_sub_tri,2]
			_tmp_ = delr*( 1. + np.cos(np.pi*dx/eps0) )*( 1. + np.cos(np.pi*dy/eps0) )*( 1. + np.cos(np.pi*dz/eps0) )/(8.*delr**3.)
			
			_sum_ += _tmp_
		
		sum_x += _Area_[_tri]*_eleGma0_[_tri,0]*_sum_/_length_
		sum_y += _Area_[_tri]*_eleGma0_[_tri,1]*_sum_/_length_
		sum_z += _Area_[_tri]*_eleGma0_[_tri,2]*_sum_/_length_
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	 
	# middle box
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid_ ) 
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_W_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	# western box = middle box - Lx
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_leng_ = len(_tri_centroid_[:,0])
	_tri_centroid0_[0:_leng_, 0] = _tri_centroid_[0:_leng_, 0] - Lx
	
	_len0_ = _all_centroid_coord_.shape[0]
	_len1_ = _all_centroid_coord_.shape[1]
	_all_centroid_coord_[0:_len0_, 0:_len1_, 0] = _all_centroid_coord_[0:_len0_, 0:_len1_, 0] - Lx
	
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_E_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	# eastern box = middle box + Lx
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_leng_ = len(_tri_centroid_[:,0])
	_tri_centroid0_[0:_leng_,0] = _tri_centroid_[0:_leng_,0] + Lx
	
	_len0_ = _all_centroid_coord_.shape[0]
	_len1_ = _all_centroid_coord_.shape[1]
	_all_centroid_coord_[0:_len0_, 0:_len1_, 0] = _all_centroid_coord_[0:_len0_, 0:_len1_, 0] + Lx
	
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_S_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	# southern box = middle box - Ly
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_leng_ = len(_tri_centroid_[:,0])
	_tri_centroid0_[0:_leng_,1] = _tri_centroid_[0:_leng_,1] - Ly
	
	_len0_ = _all_centroid_coord_.shape[0]
	_len1_ = _all_centroid_coord_.shape[1]
	_all_centroid_coord_[0:_len0_, 0:_len1_, 1] = _all_centroid_coord_[0:_len0_, 0:_len1_, 1] - Ly
	
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_N_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	# northern box = middle box + Ly
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_leng_ = len(_tri_centroid_[:,0])
	_tri_centroid0_[0:_leng_,1] = _tri_centroid_[0:_leng_,1] + Ly
	
	_len0_ = _all_centroid_coord_.shape[0]
	_len1_ = _all_centroid_coord_.shape[1]
	_all_centroid_coord_[0:_len0_, 0:_len1_, 1] = _all_centroid_coord_[0:_len0_, 0:_len1_, 1] + Ly
	
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	# west-southern box = middle box - Lx, - Ly
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_leng_ = len(_tri_centroid_[:,0]) 
	_tri_centroid0_[0:_leng_,0] = _tri_centroid_[0:_leng_,0] - Lx
	_tri_centroid0_[0:_leng_,1] = _tri_centroid_[0:_leng_,1] - Ly
	
	_len0_ = _all_centroid_coord_.shape[0]
	_len1_ = _all_centroid_coord_.shape[1]
	_all_centroid_coord_[0:_len0_, 0:_len1_, 0] = _all_centroid_coord_[0:_len0_, 0:_len1_, 0] - Lx
	_all_centroid_coord_[0:_len0_, 0:_len1_, 1] = _all_centroid_coord_[0:_len0_, 0:_len1_, 1] - Ly
		
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_ES_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	# east-southern box = middle box + Lx, - Ly
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_leng_ = len(_tri_centroid_[:,0]) 
	_tri_centroid0_[0:_leng_,0] = _tri_centroid_[0:_leng_,0] + Lx
	_tri_centroid0_[0:_leng_,1] = _tri_centroid_[0:_leng_,1] - Ly
	
	_len0_ = _all_centroid_coord_.shape[0]
	_len1_ = _all_centroid_coord_.shape[1]
	_all_centroid_coord_[0:_len0_, 0:_len1_, 0] = _all_centroid_coord_[0:_len0_, 0:_len1_, 0] + Lx
	_all_centroid_coord_[0:_len0_, 0:_len1_, 1] = _all_centroid_coord_[0:_len0_, 0:_len1_, 1] - Ly
	
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_WN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	# west-northern box = middle box - Lx, + Ly
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_leng_ = len(_tri_centroid_[:,0])
	_tri_centroid0_[0:_leng_,0] = _tri_centroid_[0:_leng_,0] - Lx
	_tri_centroid0_[0:_leng_,1] = _tri_centroid_[0:_leng_,1] + Ly
	
	_len0_ = _all_centroid_coord_.shape[0]
	_len1_ = _all_centroid_coord_.shape[1]
	_all_centroid_coord_[0:_len0_, 0:_len1_, 0] = _all_centroid_coord_[0:_len0_, 0:_len1_, 0] - Lx
	_all_centroid_coord_[0:_len0_, 0:_len1_, 1] = _all_centroid_coord_[0:_len0_, 0:_len1_, 1] + Ly
	
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_EN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	eps0 = delr*ds[0]
	
	# east-northern box = middle box + Lx, + Ly
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_leng_ = len(_tri_centroid_[:,0])
	_tri_centroid0_[0:_leng_,0] = _tri_centroid_[0:_leng_,0] + Lx
	_tri_centroid0_[0:_leng_,1] = _tri_centroid_[0:_leng_,1] + Ly
	
	_len0_ = _all_centroid_coord_.shape[0]
	_len1_ = _all_centroid_coord_.shape[1]
	_all_centroid_coord_[0:_len0_, 0:_len1_, 0] = _all_centroid_coord_[0:_len0_, 0:_len1_, 0] + Lx
	_all_centroid_coord_[0:_len0_, 0:_len1_, 1] = _all_centroid_coord_[0:_len0_, 0:_len1_, 1] + Ly
	
	_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps0, eps0, eps0, _tri_centroid0_ )
	if len(_list_) > 0: 
		sum_x, sum_y, sum_z = _add_(_eleGma0_, _all_centroid_coord_, coord, _Area_, _list_, delr, eps0)
	
	return sum_x, sum_y, sum_z
	
	
	
def _D3b_Peskin_function_toget_gridVor_( _eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_ ):
	
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny ) = ( len(x), len(y) )
	
	delr = 2.
	eps = delr*ds[0]
	
	sum_x0, sum_y0, sum_z0 = _D3_Peskin_function_       (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	sum_x4, sum_y4, sum_z4 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	
	sum_x5, sum_y5, sum_z5 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	sum_x6, sum_y6, sum_z6 = _D3_Peskin_function_ES_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	sum_x7, sum_y7, sum_z7 = _D3_Peskin_function_EN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	sum_x8, sum_y8, sum_z8 = _D3_Peskin_function_WN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, delr)
	
	sum_x = sum_x0 + sum_x1 + sum_x2 + sum_x3 + sum_x4 + sum_x5 + sum_x6 + sum_x7 + sum_x8
	sum_y = sum_y0 + sum_y1 + sum_y2 + sum_y3 + sum_y4 + sum_y5 + sum_y6 + sum_y7 + sum_y8
	sum_z = sum_z0 + sum_z1 + sum_z2 + sum_z3 + sum_z4 + sum_z5 + sum_z6 + sum_z7 + sum_z8
	
	return sum_x, sum_y, sum_z
	
	
def _remeshing_all_triangles_( _no_tri_, _triXC, _triYC, _triZC ):
	
	_no_sub_elems_ = 4 # use 4 or 16 
	_all_centroid_coord_ = np.zeros( (_no_tri_, _no_sub_elems_, 3) )
	
	for _tri in range(_no_tri_):
	
		p1_C = np.array( [_triXC[_tri,0], _triYC[_tri,0], _triZC[_tri,0]] )
		p2_C = np.array( [_triXC[_tri,1], _triYC[_tri,1], _triZC[_tri,1]] )		
		p3_C = np.array( [_triXC[_tri,2], _triYC[_tri,2], _triZC[_tri,2]] )  
		
		_centroid_sub_tri = _centroid_4subTriangle_(p1_C, p2_C, p3_C)
		
		_all_centroid_coord_[_tri,0:_no_sub_elems_,0] = _centroid_sub_tri[0:_no_sub_elems_,0]
		_all_centroid_coord_[_tri,0:_no_sub_elems_,1] = _centroid_sub_tri[0:_no_sub_elems_,1]
		_all_centroid_coord_[_tri,0:_no_sub_elems_,2] = _centroid_sub_tri[0:_no_sub_elems_,2]
		
	return _all_centroid_coord_
	
	
# ########################################################################################################
	
	
#	flag_on_wall    = _chk_GridPoint_on_boundary    ( coord[0], coord[1], coord[2] )
#	flag_cl_wall    = _chk_GridPoint_close_boundary ( coord[0], coord[1], coord[2], eps )
#	#flag_cr_wall    = _chk_GridPoint_on_corner_line_( coord[0], coord[1], coord[2] )
#	flag_cl_cr_line = _chk_GridPoint_cl_corner_line_( coord[0], coord[1], coord[2] )
#	#flag_cl_cr_wall = _chk_GridPoint_cl_wall_corner_( coord[0], coord[1], coord[2] )
#	
#	flag = True
#	_len_list_ = 0
#	
#	if coord[2] == abs(Lz):
#		flag = False
#		( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
#		_len_list_ = 0
#		
#	if flag_on_wall:
#		flag = False 
#		sum_x, sum_y, sum_z = _D3b_Peskin_function_on_wall(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#		
#	if flag_cl_wall:
#		flag= False
#		sum_x, sum_y, sum_z = _D3b_Peskin_function_cl_wall(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#	
#	if flag_cl_cr_line:
#		flag= False
#		sum_x, sum_y, sum_z = _D3b_Peskin_function_at_cl_cr_line(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#	
##	if flag_cl_cr_wall:
##		flag= False
##		sum_x, sum_y, sum_z = _D3b_Peskin_function_at_cl_cr_wall(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#		
##	if flag_cr_wall:
##		flag = False
##		if coord[0] == x[0] and coord[1] == y[0]:
#			sum_x, sum_y, sum_z = _D3b_Peskin_function_at_cr_line(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#		else:
#			( coord[0], coord[1] ) = ( x[0], y[0] ) 
#			sum_x, sum_y, sum_z = _D3b_Peskin_function_at_cr_line(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#		
#	if flag: 
#		_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps, eps, eps, _tri_centroid_ )
#		_len_list_ = len(_list_) 
#		
#		if _len_list_ > 0:
#			for _len in range( len(_list_) ):
#				
#				_tri = _list_[_len]
#				_length_ = _all_centroid_coord_.shape[1]
#				_sum_ = 0.
#				
#				for _sub_tri in range(_length_):
#					( xp, yp, zp ) = ( _all_centroid_coord_[_tri,_sub_tri,0], _all_centroid_coord_[_tri,_sub_tri,1], _all_centroid_coord_[_tri,_sub_tri,2] )
#					( dx, dy, dz ) = ( coord[0]-xp, coord[1]-yp, coord[2]-zp )
#					_tmp_ = delr*( 1. + np.cos(np.pi*dx/eps) )*( 1. + np.cos(np.pi*dy/eps) )*( 1. + np.cos(np.pi*dz/eps) )/(8.*delr**3.)
#					
#					_sum_ += _tmp_
#				
#				sum_x += _Area_[_tri]*_eleGma0_[_tri,0]*_sum_/_length_
#				sum_y += _Area_[_tri]*_eleGma0_[_tri,1]*_sum_/_length_
#				sum_z += _Area_[_tri]*_eleGma0_[_tri,2]*_sum_/_length_
#	
#
# ######################################################################################################################
	
#	def _D3b_Peskin_function_on_wall(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_):
#	
#	Lx, Ly, Lz = Domain()
#	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
#	x, y, z = _SmoothGrid_Generation_()
#	Lx, Ly, Lz = Domain()
#	
#	if coord[2] == abs(Lz):
#		( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
#	else:
#		sum_x0, sum_y0, sum_z0 = _D3_Peskin_function_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#		
#		# next domain 
#		if coord[0] == 0.: 
#			sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			sum_x4, sum_y4, sum_z4 = _D3_Peskin_function_WN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			sum_x5, sum_y5, sum_z5 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			
#			( sum_x, sum_y, sum_z ) = \
#			( sum_x0+sum_x1+sum_x2+sum_x3+sum_x4+sum_x5, sum_y0+sum_y1+sum_y2+sum_y3+sum_y4+sum_y5, sum_z0+sum_z1+sum_z2+sum_z3+sum_z4+sum_z5 )
#			
#		if coord[1] == 0.: 
#			sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			sum_x4, sum_y4, sum_z4 = _D3_Peskin_function_ES_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			sum_x5, sum_y5, sum_z5 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
#			
#			( sum_x, sum_y, sum_z ) = \
#			( sum_x0+sum_x1+sum_x2+sum_x3+sum_x4+sum_x5, sum_y0+sum_y1+sum_y2+sum_y3+sum_y4+sum_y5, sum_z0+sum_z1+sum_z2+sum_z3+sum_z4+sum_z5 )
#			
#	return sum_x, sum_y, sum_z
	
	
def _D3b_Peskin_function_cl_wall(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_):
	
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny, nz ) = ( len(x), len(y), len(z) )
	delr = 2.0
	
	sum_x0, sum_y0, sum_z0 = _D3_Peskin_function_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
	
	if coord[0] == x[nx-2] or coord[0] == x[1]:
		if coord[0] == x[1]:
			sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x4, sum_y4, sum_z4 = _D3_Peskin_function_WN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x5, sum_y5, sum_z5 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			
			( sum_x, sum_y, sum_z ) = \
			( sum_x0+sum_x1+sum_x2+sum_x3+sum_x4+sum_x5, sum_y0+sum_y1+sum_y2+sum_y3+sum_y4+sum_y5, sum_z0+sum_z1+sum_z2+sum_z3+sum_z4+sum_z5 )

		if coord[0] == x[nx-2]: 
			sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_ES_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x4, sum_y4, sum_z4 = _D3_Peskin_function_EN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x5, sum_y5, sum_z5 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			
			( sum_x, sum_y, sum_z ) = \
			( sum_x0+sum_x1+sum_x2+sum_x3+sum_x4+sum_x5, sum_y0+sum_y1+sum_y2+sum_y3+sum_y4+sum_y5, sum_z0+sum_z1+sum_z2+sum_z3+sum_z4+sum_z5 )
		
	elif coord[1] == y[ny-2] or coord[1] == y[1]:
		if coord[1] == y[1]: 
			sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x4, sum_y4, sum_z4 = _D3_Peskin_function_ES_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x5, sum_y5, sum_z5 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			
			( sum_x, sum_y, sum_z ) = \
			( sum_x0+sum_x1+sum_x2+sum_x3+sum_x4+sum_x5, sum_y0+sum_y1+sum_y2+sum_y3+sum_y4+sum_y5, sum_z0+sum_z1+sum_z2+sum_z3+sum_z4+sum_z5 )
		
		if coord[1] == y[ny-2]: 
			sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x4, sum_y4, sum_z4 = _D3_Peskin_function_EN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			sum_x5, sum_y5, sum_z5 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
			
			( sum_x, sum_y, sum_z ) = \
			( sum_x0+sum_x1+sum_x2+sum_x3+sum_x4+sum_x5, sum_y0+sum_y1+sum_y2+sum_y3+sum_y4+sum_y5, sum_z0+sum_z1+sum_z2+sum_z3+sum_z4+sum_z5 )
		
	elif coord[2] <= abs(Lz)-ds[2]:
		( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	###
	#else:
	#	if coord[0] == eps/delr: 
	#		coord0 = np.array( [ Lx, coord[1], coord[2] ] )
	#		eps0 = np.array( [ 0.5*eps, eps, eps ] )
	#		sum_xr, sum_yr, sum_zr = \
	#		_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_, eps0)
	#		
	#	if coord[1] == eps/delr: 
	#		coord0 = np.array( [ coord[0], Ly, coord[2] ] )
	#		eps0 = np.array( [ eps, 0.5*eps, eps ] )
	#		sum_xr, sum_yr, sum_zr = \
	#		_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_, eps0)
		
		#( sum_x, sum_y, sum_z ) = ( 1.42*sum_xl + 1.42*sum_xr, 1.42*sum_yl + 1.42*sum_yr, 1.42*sum_zl + 1.42*sum_zr  )
		
	return sum_x, sum_y, sum_z
	
	
def _D3b_Peskin_function_at_cr_line(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_):
	
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny, nz ) = ( len(x), len(y), len(z) )
	
	coord0 = np.array( [ 0., 0., coord[2] ] )
	sum_x0, sum_y0, sum_z0 = _D3_Peskin_function_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_)
	
	if coord[0] == 0. and coord[1] == 0.:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		
		#if coord[0] == 0. and coord[1] == 0. and coord[2] == 0.: 
		#	print( 'sum_y = ', sum_y, ' sum_y0 = ', sum_y0, ' sum_y1 = ', sum_y1, ' sum_y2 = ', sum_y2, ' sum_y3 = ', sum_y3,  )
	
	return sum_x, sum_y, sum_z
	
	
def _D3b_Peskin_function_at_cl_cr_line(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_):
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny, nz ) = ( len(x), len(y), len(z) )	
	
	sum_x0, sum_y0, sum_z0 = _D3_Peskin_function_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
	
	if coord[0] == x[1] and coord[1] == y[1]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
	
	if coord[0] == x[nx-2] and coord[1] == y[1]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_ES_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		
	if coord[0] == x[1] and coord[1] == y[ny-2]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		
	if coord[0] == x[nx-2] and coord[1] == y[ny-2]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_EN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
	
	return sum_x, sum_y, sum_z	
	
	
def _D3b_Peskin_function_at_cl_cr_wall(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_):
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny, nz ) = ( len(x), len(y), len(z) )	
	
	sum_x0, sum_y0, sum_z0 = _D3_Peskin_function_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
	
	if coord[0] == x[1] and coord[1] == y[0]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
	
	if coord[0] == x[0] and coord[1] == y[1]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WS_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
	
	if coord[0] == x[0] and coord[1] == y[ny-2]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
	
	if coord[0] == x[1] and coord[1] == y[ny-1]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_W_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_WN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
	
	if coord[0] == x[nx-2] and coord[1] == y[ny-1]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_EN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		
	if coord[0] == x[nx-1] and coord[1] == y[ny-2]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_N_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_EN_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		
	if coord[0] == x[nx-1] and coord[1] == y[1]:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_ES_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		
	if coord[0] == x[nx-2] and coord[1] == 0.:
		sum_x1, sum_y1, sum_z1 = _D3_Peskin_function_E_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x2, sum_y2, sum_z2 = _D3_Peskin_function_ES_box_(_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		sum_x3, sum_y3, sum_z3 = _D3_Peskin_function_S_box_ (_eleGma0_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_)
		
		( sum_x, sum_y, sum_z ) = \
		( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		
	return sum_x, sum_y, sum_z
	
	
