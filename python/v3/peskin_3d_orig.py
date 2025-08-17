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
from utility_3d_paral import _chk_GridPoint_close_boundary, _chk_GridPoint_on_boundary
from utility_3d_paral import _chk_GridPoint_on_corner_pt_, _chk_GridPoint_on_corner_line_, _chk_GridPoint_cl_corner_line_, _find_elements_nearby_
from utility_3d_paral import _chk_GridPoint_cl_wall_corner_


def _D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps):
	Lx, Ly, Lz = Domain()
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	#( delr0, delr1, delr2 ) = ( eps[0]/ds[0], eps[1]/ds[0], eps[2]/ds[0] ) 
	( delr0, delr1, delr2 ) = ( 2.0, 2.0, 2.0 )
	eps_t = delr0*eps[2]
	 
	 # middle box
	_list0_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	# western box = middle box - Lx
	_tri_centroid_[:,0] = _tri_centroid_[:,0] - Lx
	_list1_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	# eastern box = middle box + Lx
	_tri_centroid_[:,0] = _tri_centroid_[:,0] + Lx
	_list2_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	# southern box = middle box - Ly
	_tri_centroid_[:,1] = _tri_centroid_[:,0] - Ly
	_list3_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	# northern box = middle box + Ly
	_tri_centroid_[:,1] = _tri_centroid_[:,0] + Ly
	_list4_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	# west-south box
	_tri_centroid_[:,0] = _tri_centroid_[:,0] - Lx
	_tri_centroid_[:,1] = _tri_centroid_[:,1] - Ly
	_list5_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	# west-north box
	_tri_centroid_[:,0] = _tri_centroid_[:,0] - Lx
	_tri_centroid_[:,1] = _tri_centroid_[:,1] + Ly
	_list5_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	# east-south box
	_tri_centroid_[:,0] = _tri_centroid_[:,0] + Lx
	_tri_centroid_[:,1] = _tri_centroid_[:,1] - Ly
	_list5_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	# east-north box
	_tri_centroid_[:,0] = _tri_centroid_[:,0] + Lx
	_tri_centroid_[:,1] = _tri_centroid_[:,1] + Ly
	_list5_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )	
	
	_list_ = _list0_ + _list1_ + _list2_ + _list3_ + _list4_ + _list5_ 
	
	if len(_list_) > 0:
		for _len in range( len(_list_) ):
			
			_tri = _list_[_len]
			_length_ = _all_centroid_coord_.shape[1]
			_sum_ = 0.
			
			for _sub_tri in range(_length_):
				( xp, yp, zp ) = \
				( _all_centroid_coord_[_tri,_sub_tri,0], _all_centroid_coord_[_tri,_sub_tri,1], _all_centroid_coord_[_tri,_sub_tri,2] )
				( dx, dy, dz ) = ( coord[0]-xp, coord[1]-yp, coord[2]-zp )
				
				_tmp_ = \
				( 1. + np.cos(np.pi*dx/eps_t) )*( 1. + np.cos(np.pi*dy/eps_t) )*( 1. + np.cos(np.pi*dz/eps_t) )/( 8.*delr0*delr1*delr2 )
				#( 1. + np.cos(np.pi*dx/eps[0]) )*( 1. + np.cos(np.pi*dy/eps[1]) )*( 1. + np.cos(np.pi*dz/eps[2]) )/( 8.*delr0*delr1*delr2 )
				
				_sum_ += _tmp_
			
			sum_x += _Area_[_tri]*_eleGma0_[_tri,0]*_sum_/_length_
			sum_y += _Area_[_tri]*_eleGma0_[_tri,1]*_sum_/_length_
			sum_z += _Area_[_tri]*_eleGma0_[_tri,2]*_sum_/_length_
	
	
	return sum_x, sum_y, sum_z
	
	
def _D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, Actual_coord, Pseudo_coord, lx, ly, ds, _Area_, eps):
	
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	( delr0, delr1, delr2 ) = ( 2.0, 2.0, 2.0 ) #( eps[0]/ds[0], eps[1]/ds[0], eps[2]/ds[0] ) 
	eps_t = delr0*eps[2]
	
	_list_ = _find_elements_nearby_( Pseudo_coord[0], Pseudo_coord[1], Pseudo_coord[2], eps[0], eps[1], eps[2], _tri_centroid_ )
	
	if len(_list_) > 0:
		for _len in range( len(_list_) ):
			
			_tri = _list_[_len]
			_length_ = _all_centroid_coord_.shape[1]
			_sum_ = 0.
			
			for _sub_tri in range(_length_):
				( xp, yp, zp ) = \
				( _all_centroid_coord_[_tri,_sub_tri,0]-lx, _all_centroid_coord_[_tri,_sub_tri,1]-ly, _all_centroid_coord_[_tri,_sub_tri,2] )
				( dx, dy, dz ) = ( Actual_coord[0]-xp, Actual_coord[1]-yp, Actual_coord[2]-zp )
				
				_tmp_ = \
				( 1. + np.cos(np.pi*dx/eps_t) )*( 1. + np.cos(np.pi*dy/eps_t) )*( 1. + np.cos(np.pi*dz/eps_t) )/( 8.*delr0*delr1*delr2 )
				
				_sum_ += _tmp_
			
			sum_x += _Area_[_tri]*_eleGma0_[_tri,0]*_sum_/_length_
			sum_y += _Area_[_tri]*_eleGma0_[_tri,1]*_sum_/_length_
			sum_z += _Area_[_tri]*_eleGma0_[_tri,2]*_sum_/_length_
			
	return sum_x, sum_y, sum_z
	
	
def _D3b_Peskin_function_on_wall(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps):
	
	( sum_xl, sum_yl, sum_zl ) = ( 0., 0., 0. )
	( sum_xr, sum_yr, sum_zr ) = ( 0., 0., 0. )
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	
	
	if coord[2] == abs(Lz):
		( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	
	else:
		eps0 = np.array( [ eps, eps, eps ] )
		sum_xl, sum_yl, sum_zl = \
		_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps0)
		
		# next domain 
	#	if coord[0] == 0. and coord[1] != 0.: 
	#		coord0 = np.array( [ Lx, coord[1], coord[2] ] )
	#		eps0 = np.array( [ eps, eps, eps ] )
	#		sum_xr, sum_yr, sum_zr = \
	#		_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_, eps0)
	#		
	#		( sum_x, sum_y, sum_z ) = ( sum_xl + sum_xr, sum_yl + sum_yr, sum_zl + sum_zr )
	#		
	#	if coord[1] == 0. and coord[0] != 0.: 
	#		coord0 = np.array( [ 0., Ly, coord[2] ] )
	#		eps0 = np.array( [ eps, eps, eps ] )
	#		sum_xr, sum_yr, sum_zr = \
	#		_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_, eps0)
	#		
	#		( sum_x, sum_y, sum_z ) = ( sum_xl + sum_xr, sum_yl + sum_yr, sum_zl + sum_zr )
			
	return sum_xl, sum_yl, sum_zl
	
	
def _D3b_Peskin_function_cl_wall(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps):
	
	( sum_xl, sum_yl, sum_zl ) = ( 0., 0., 0. )
	( sum_xr, sum_yr, sum_zr ) = ( 0., 0., 0. )
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny, nz ) = ( len(x), len(y), len(z) )
	
	delr = eps/ds[0]
	eps0 = np.array( [ eps, eps, eps ] )
	
	sum_xl, sum_yl, sum_zl = \
	_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps0)	
	
	if coord[0] == x[nx-2] or coord[0] == x[1]:
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		if coord[0] == x[1]:
			Pseudo_coord0 = np.array( [ Lx, coord[1], coord[2] ] )
			sum_xr, sum_yr, sum_zr = \
			_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, +Lx, 0., ds, _Area_, eps0)			

		if coord[0] == x[nx-2]: 
			Pseudo_coord0 = np.array( [ 0., coord[1], coord[2] ] )
			sum_xr, sum_yr, sum_zr = \
			_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_xl + 0.*sum_xr, sum_yl + 0.*sum_yr, sum_zl + 0.*sum_zr  )
		
	elif coord[1] == y[ny-2] or coord[1] == y[1]:
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		if coord[1] == y[1]: 
			Pseudo_coord0 = np.array( [ coord[0], Ly, coord[2] ] )
			sum_xr, sum_yr, sum_zr = \
			_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., +Ly, ds, _Area_, eps0)
		
		if coord[1] == y[ny-2]: 
			Pseudo_coord0 = np.array( [ coord[0], 0., coord[2] ] )
			sum_xr, sum_yr, sum_zr = \
			_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., -Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_xl + 0.*sum_xr, sum_yl + 0.*sum_yr, sum_zl + 0.*sum_zr  )
		
	elif coord[2] <= abs(Lz)-eps/delr:
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
	
	
def _D3b_Peskin_function_at_cr_line(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps):
	
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny, nz ) = ( len(x), len(y), len(z) )
	
	eps0 = np.array( [ eps, eps, eps ] )
	coord0 = np.array( [ 0., 0., coord[2] ] )
	sum_x0, sum_y0, sum_z0 = \
	_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord0, ds, _Area_, eps0)
			
	if coord[0] == 0. and coord[1] == 0.:
		# southern region
		Pseudo_coord0 = np.array( [ 0., Ly, coord[2] ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord0, Pseudo_coord0, 0., Ly, ds, _Area_, eps0)
		
		# western region
		Pseudo_coord0 = np.array( [ Lx, 0., coord[2] ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord0, Pseudo_coord0, Lx, 0., ds, _Area_, eps0)
		
		# south-west region
		Pseudo_coord0 = np.array( [ Lx, Ly, coord[2] ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord0, Pseudo_coord0, Lx, Ly, ds, _Area_, eps0)
		
		sum_x = sum_x0 + sum_x1 + sum_x2 + sum_x3  
		sum_y = sum_y0 + sum_y1 + sum_y2 + sum_y3
		sum_z = sum_z0 + sum_z1 + sum_z2 + sum_z3
		
		#if coord[0] == 0. and coord[1] == 0. and coord[2] == 0.: 
		#	print( 'sum_y = ', sum_y, ' sum_y0 = ', sum_y0, ' sum_y1 = ', sum_y1, ' sum_y2 = ', sum_y2, ' sum_y3 = ', sum_y3,  )
	
	return sum_x, sum_y, sum_z
	
	
def _D3b_Peskin_function_at_cl_cr_line(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps):
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny, nz ) = ( len(x), len(y), len(z) )	
	
	delr = eps/ds[0]
	eps0 = np.array( [ eps, eps, eps ] )
	sum_x0, sum_y0, sum_z0 = \
	_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps0)
	
	if coord[0] == x[1] and coord[1] == y[1]:
		Pseudo_coord0 = np.array( [ Lx, y[1], coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ x[1], Ly, coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, Ly, coord[2] ] )
		eps0 = np.array( [ 0.5*eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		( sum_x, sum_y, sum_z ) = ( 1.1*sum_x, 1.1*sum_y, 1.1*sum_z )
	
	if coord[0] == x[nx-2] and coord[1] == y[1]:
		Pseudo_coord0 = np.array( [ 0., y[1], coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ x[nx-2], 0., coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., -Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., 0., coord[2] ] )
		eps0 = np.array( [ 0.5*eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		#( sum_x, sum_y, sum_z ) = ( 1.31*sum_x, 1.31*sum_y, 1.31*sum_z )
		
	if coord[0] == x[1] and coord[1] == y[ny-2]:
		Pseudo_coord0 = np.array( [ Lx, y[ny-2], coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ x[1], 0., coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., -Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, 0., coord[2] ] )
		eps0 = np.array( [ 0.5*eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, -Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		( sum_x, sum_y, sum_z ) = ( 1.1*sum_x, 1.1*sum_y, 1.1*sum_z )
		
	if coord[0] == x[nx-2] and coord[1] == y[ny-2]:
		Pseudo_coord0 = np.array( [ 0., y[ny-2], coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )		
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ x[nx-2], 0., coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )	
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., -Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., 0., coord[2] ] )
		eps0 = np.array( [ 0.5*eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, -Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		( sum_x, sum_y, sum_z ) = ( 1.1*sum_x, 1.1*sum_y, 1.1*sum_z )
	
	return sum_x, sum_y, sum_z	
	
	
def _D3b_Peskin_function_at_cl_cr_wall(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps):
	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny, nz ) = ( len(x), len(y), len(z) )	
	
	delr = eps/ds[0]
	eps0 = np.array( [ eps, eps, eps ] )
	sum_x0, sum_y0, sum_z0 = \
	_D3_Peskin_function_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps0)
	
	
	if coord[0] == x[1] and coord[1] == y[0]:
		Pseudo_coord0 = np.array( [ Lx, 0., coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ x[1], Ly, coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, Ly, coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		#( sum_x, sum_y, sum_z ) = ( 1.45*sum_x, 1.45*sum_y, 1.45*sum_z )
	
	if coord[0] == x[0] and coord[1] == y[1]:
		Pseudo_coord0 = np.array( [ Lx, y[1], coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ x[0], Ly, coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, Ly, coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		#( sum_x, sum_y, sum_z ) = ( 1.51*sum_x, 1.51*sum_y, 1.51*sum_z )
	
	if coord[0] == x[0] and coord[1] == y[ny-2]:
		Pseudo_coord0 = np.array( [ Lx, y[ny-2], coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., 0., coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, 0., coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, 0., ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		#( sum_x, sum_y, sum_z ) = ( 1.*sum_x, 1.*sum_y, 1.*sum_z )
	
	if coord[0] == x[1] and coord[1] == y[ny-1]:
		Pseudo_coord0 = np.array( [ x[1], 0., coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., -Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, 0., coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, -Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, Ly, coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, Lx, 0., ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		#( sum_x, sum_y, sum_z ) = ( 1.5*sum_x, 1.5*sum_y, 1.5*sum_z )
	
	if coord[0] == x[nx-2] and coord[1] == y[ny-1]:
		Pseudo_coord0 = np.array( [ x[nx-2], 0., coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., -Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., 0., coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, -Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., Ly, coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		#( sum_x, sum_y, sum_z ) = ( 1.5*sum_x, 1.5*sum_y, 1.5*sum_z )
		
	if coord[0] == x[nx-1] and coord[1] == y[ny-2]:
		Pseudo_coord0 = np.array( [ 0., y[ny-2], coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., 0., coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, -Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, 0., coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., -Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		( sum_x, sum_y, sum_z ) = ( 1.46*sum_x, 1.46*sum_y, 1.46*sum_z )
		
	if coord[0] == x[nx-1] and coord[1] == y[1]:
		Pseudo_coord0 = np.array( [ 0., y[1], coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., Ly, coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, +Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, Ly, coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., +Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		( sum_x, sum_y, sum_z ) = ( 1.5*sum_x, 1.5*sum_y, 1.5*sum_z )
		
	if coord[0] == x[nx-1] and coord[1] == y[1]:
		Pseudo_coord0 = np.array( [ 0., y[1], coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., Ly, coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, +Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ Lx, Ly, coord[2] ] )
		eps0 = np.array( [ eps, 0.5*eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., +Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		( sum_x, sum_y, sum_z ) = ( 1.5*sum_x, 1.5*sum_y, 1.5*sum_z )
		
	if coord[0] == x[nx-2] and coord[1] == 0.:
		Pseudo_coord0 = np.array( [ 0., 0., coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x1, sum_y1, sum_z1 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, 0., ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ 0., Ly, coord[2] ] )
		eps0 = np.array( [ 0.5*eps, eps, eps ] )
		sum_x2, sum_y2, sum_z2 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, -Lx, +Ly, ds, _Area_, eps0)
		
		Pseudo_coord0 = np.array( [ x[nx-2], y[ny-1], coord[2] ] )
		eps0 = np.array( [ eps, eps, eps ] )
		sum_x3, sum_y3, sum_z3 = \
		_D3_Peskin_function_with_periodicity_(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, Pseudo_coord0, 0., +Ly, ds, _Area_, eps0)
		
		( sum_x, sum_y, sum_z ) = ( sum_x0+sum_x1+sum_x2+sum_x3, sum_y0+sum_y1+sum_y2+sum_y3, sum_z0+sum_z1+sum_z2+sum_z3 )
		#( sum_x, sum_y, sum_z ) = ( 1.46*sum_x, 1.46*sum_y, 1.46*sum_z )
		
		
	return sum_x, sum_y, sum_z
	
	
def _D3b_Peskin_function_toget_gridVor_( _eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_ ):

	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
	Lx, Ly, Lz = Domain()
	x, y, z = _SmoothGrid_Generation_()
	( nx, ny ) = ( len(x), len(y) )
	
	delr = 2.
	eps = delr*ds[0]
	
	flag_on_wall = _chk_GridPoint_on_boundary( coord[0], coord[1], coord[2] )
	flag_cl_wall = _chk_GridPoint_close_boundary( coord[0], coord[1], coord[2], eps )
	flag_cr_wall = _chk_GridPoint_on_corner_line_( coord[0], coord[1], coord[2] )
	flag_cl_cr_line = _chk_GridPoint_cl_corner_line_( coord[0], coord[1], coord[2] )
	flag_cl_cr_wall = _chk_GridPoint_cl_wall_corner_( coord[0], coord[1], coord[2] )
	
	flag = True
	_len_list_ = 0
	
	if coord[2] == abs(Lz):
		flag = False
		( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
		_len_list_ = 0
		
	if flag_on_wall:
		flag = False 
		sum_x, sum_y, sum_z = \
		_D3b_Peskin_function_on_wall(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps)
		
	if flag_cl_wall:
		flag= False
		sum_x, sum_y, sum_z = \
		_D3b_Peskin_function_cl_wall(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps)
	
	if flag_cl_cr_line:
		flag= False
		sum_x, sum_y, sum_z = \
		_D3b_Peskin_function_at_cl_cr_line(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps)
	
	if flag_cl_cr_wall:
		flag= False
		sum_x, sum_y, sum_z = \
		_D3b_Peskin_function_at_cl_cr_wall(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps)	
		
	if flag_cr_wall:
		flag = False
		if coord[0] == x[0] and coord[1] == y[0]:
			sum_x, sum_y, sum_z = \
			_D3b_Peskin_function_at_cr_line(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps)
		else:
			( coord[0], coord[1] ) = ( x[0], y[0] ) 
			sum_x, sum_y, sum_z = \
			_D3b_Peskin_function_at_cr_line(_eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_, eps)
		
	if flag: 
		_list_ = _find_elements_nearby_( coord[0], coord[1], coord[2], eps, eps, eps, _tri_centroid_ )
		_len_list_ = len(_list_) 
		
		if _len_list_ > 0:
			for _len in range( len(_list_) ):
				
				_tri = _list_[_len]
				_length_ = _all_centroid_coord_.shape[1]
				_sum_ = 0.
				
				for _sub_tri in range(_length_):
					( xp, yp, zp ) = \
					( _all_centroid_coord_[_tri,_sub_tri,0], _all_centroid_coord_[_tri,_sub_tri,1], _all_centroid_coord_[_tri,_sub_tri,2] )
					( dx, dy, dz ) = ( coord[0]-xp, coord[1]-yp, coord[2]-zp )
					
					_tmp_ = \
					delr*( 1. + np.cos(np.pi*dx/eps) )*( 1. + np.cos(np.pi*dy/eps) )*( 1. + np.cos(np.pi*dz/eps) )/(8.*delr**3.)
					
					_sum_ += _tmp_
				
				sum_x += _Area_[_tri]*_eleGma0_[_tri,0]*_sum_/_length_
				sum_y += _Area_[_tri]*_eleGma0_[_tri,1]*_sum_/_length_
				sum_z += _Area_[_tri]*_eleGma0_[_tri,2]*_sum_/_length_
				
	#if coord[0] == x[nx-2] and coord[1] == y[ny-1] and coord[2] == 0.: print( sum_x, sum_y, sum_z )
	#if coord[0] == x[1] and coord[1] == y[0] and coord[2] == 0.: print( sum_x, sum_y, sum_z )
	
	return sum_x, sum_y, sum_z
	
	
def _remeshing_all_triangles_( _no_tri_, _triXC, _triYC, _triZC ):
	
	_no_sub_elems_ = 16 # use 4 or 16 
	_all_centroid_coord_ = np.zeros( (_no_tri_, _no_sub_elems_, 3) )
	
	for _tri in range(_no_tri_):
	
		p1_C = np.array( [_triXC[_tri,0], _triYC[_tri,0], _triZC[_tri,0]] )
		p2_C = np.array( [_triXC[_tri,1], _triYC[_tri,1], _triZC[_tri,1]] )		
		p3_C = np.array( [_triXC[_tri,2], _triYC[_tri,2], _triZC[_tri,2]] )  
		
		_centroid_sub_tri = _centroid_16subTriangle_(p1_C, p2_C, p3_C)
		
		_all_centroid_coord_[_tri,0:_no_sub_elems_,0] = _centroid_sub_tri[0:_no_sub_elems_,0]
		_all_centroid_coord_[_tri,0:_no_sub_elems_,1] = _centroid_sub_tri[0:_no_sub_elems_,1]
		_all_centroid_coord_[_tri,0:_no_sub_elems_,2] = _centroid_sub_tri[0:_no_sub_elems_,2]
		
	return _all_centroid_coord_
	
	
	
	
#def _D3a_Peskin_function_toget_gridVor_( _eleGma0_, _no_tri_, _triXC, _triYC, _triZC, _all_centroid_coord_, coord, ds, _Area_ ):
#	
#	( sum_x, sum_y, sum_z ) = ( 0., 0., 0. )
#	Lx, Ly, Lz = Domain()
#
#	( x0, y0, z0 ) = ( coord[0], coord[1], coord[2] )
#	
#	delr = 2.
#	eps = delr*ds[0] 
#	
#	flag_on_wall = _chk_GridPoint_on_boundary( x0, y0, z0 )
#	flag_cl_wall = _chk_GridPoint_close_boundary( x0, y0, z0, eps )
#	flag_cr_wall = _chk_GridPoint_on_corner_( x0, y0, z0 )
#	
#	for _tri in range(_no_tri_):
#	
#		_length_ = _all_centroid_coord_.shape[1]
#		
#		_sum_ = 0.
#		for _len in range(_length_):
#		
#			( xp, yp, zp ) = \
#			( _all_centroid_coord_[_tri,_len,0], _all_centroid_coord_[_tri,_len,1], _all_centroid_coord_[_tri,_len,2] )
#			( dx, dy, dz ) = ( abs(x0-xp), abs(y0-yp), abs(z0-zp) ) 
#			
#			if dx <= eps and dy <= eps and dz <= eps:
#				if flag_on_wall:
#					_tmp_ = \
#					( 1. + np.cos(np.pi*dx/eps) )*( 1. + np.cos(np.pi*dy/eps) )*( 1. + np.cos(np.pi*dz/eps) )/(4.*delr**3.)
#				elif flag_cr_wall:
#					_tmp_ = \
#					( 1. + np.cos(np.pi*dx/eps) )*( 1. + np.cos(np.pi*dy/eps) )*( 1. + np.cos(np.pi*dz/eps) )/(1.*delr**3.)
#				elif flag_cl_wall:
#					_tmp_ = \
#					( 1. + np.cos(np.pi*dx/eps) )*( 1. + np.cos(np.pi*dy/eps) )*( 1. + np.cos(np.pi*dz/eps) )/(8.*delr**3.)
#				else:
#					_tmp_ = \
#					( 1. + np.cos(np.pi*dx/eps) )*( 1. + np.cos(np.pi*dy/eps) )*( 1. + np.cos(np.pi*dz/eps) )/(8.*delr**3.)
#			else:
#				_tmp_ = 0.
#			
#			_sum_ += _tmp_
#		
#		sum_x += _Area_[_tri]*_eleGma0_[_tri,0]*_sum_/_length_
#		sum_y += _Area_[_tri]*_eleGma0_[_tri,1]*_sum_/_length_
#		sum_z += _Area_[_tri]*_eleGma0_[_tri,2]*_sum_/_length_
#		
#	if x0==0.25 and y0==0.25 and z0==0.: 
#		print(sum_y)
#	
#	return sum_x, sum_y, sum_z
