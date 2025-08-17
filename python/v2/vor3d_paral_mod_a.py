#import sys
#import os
import time
import numpy as np
from scipy import linalg
import scipy.io as sio
import multiprocessing as mp
from numpy.fft import fft, ifft, fftfreq
import unittest
import ctypes
from scipy.linalg import lstsq

#import multiprocessing as mp
from remeshing_3d import _elements_splitting_
from utility_3d_paral import _allTriangle_Area_, _D1_Peskin_function_, _SmoothGrid_Generation_, _allTriangle_Centroid_
from utility_3d_paral import _centroid_16subTriangle_, _centroid_4subTriangle_, Domain, _allTriangle_Area_ 

#from utility_3d_paral import _chk_GridPoint_close_boundary, _chk_GridPoint_on_boundary
#from utility_3d_paral import _chk_GridPoint_on_corner_pt_, _chk_GridPoint_on_corner_line_, _chk_GridPoint_cl_corner_line_, _find_elements_nearby_
from init_3d import _tri_coord_
#_find_alltriangels_around_meshgridPoint_, 


from peskin_3d_grid import _remeshing_all_triangles_, _D3b_Peskin_function_toget_gridVor_
from peskin_3d_node import _D3b_Peskin_function_toget_nodeVel_


shared_array = None

def _init_(shared_array_base, nx, ny):
	global shared_array
	shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
	shared_array = shared_array.reshape(nx, ny)

# ------------------------------------------------------
# element Vorticity = Area x element Vortex Strength 
# ------------------------------------------------------

def _MatrixInversion_2_cal_nodeCircu_frm_eleVor_( _nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma_ ):
	
	#_nodeXC = _nodeCoord[:,0]
	#_nodeYC = _nodeCoord[:,1]
	#_nodeZC = _nodeCoord[:,2]
		
	#Obj_ = _elements_splitting_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, del_split )	
	#_TriVert_p1C, _TriVert_p2C, _TriVert_p3C = Obj_._TriVert_Cord() 	
		
	_Triangle_Area_ = _allTriangle_Area_( _triXC, _triYC, _triZC, _no_tri_, tri_verti_ )
#	print('_len_area_ = ', len(_Triangle_Area_))
#	print('_len_eleGma_ = ', _eleGma_.shape[0] )
	
	_nodeCirc = np.zeros( (_no_tri_,3) )
	
	for _tri in range(_no_tri_):
	
		#( Xvec_p1p2, Xvec_p2p3, Xvec_p3p1 ) = \
		#( _TriVert_p2C[_tri,0] - _TriVert_p1C[_tri,0], _TriVert_p3C[_tri,0] - _TriVert_p2C[_tri,0], _TriVert_p1C[_tri,0] - _TriVert_p3C[_tri,0] )
		
		#( Yvec_p1p2, Yvec_p2p3, Yvec_p3p1 ) = \
		#( _TriVert_p2C[_tri,1] - _TriVert_p1C[_tri,1], _TriVert_p3C[_tri,1] - _TriVert_p2C[_tri,1], _TriVert_p1C[_tri,1] - _TriVert_p3C[_tri,1] )
		
		#( Zvec_p1p2, Zvec_p2p3, Zvec_p3p1 ) = \
		#( _TriVert_p2C[_tri,2] - _TriVert_p1C[_tri,2], _TriVert_p3C[_tri,2] - _TriVert_p2C[_tri,2], _TriVert_p1C[_tri,2] - _TriVert_p3C[_tri,2] )
		
		( Xvec_p1p2, Xvec_p2p3, Xvec_p3p1 ) = \
		( _triXC[_tri,1] - _triXC[_tri,0], _triXC[_tri,2] - _triXC[_tri,1], _triXC[_tri,0] - _triXC[_tri,2] )
		
		( Yvec_p1p2, Yvec_p2p3, Yvec_p3p1 ) = \
		( _triYC[_tri,1] - _triYC[_tri,0], _triYC[_tri,2] - _triYC[_tri,1], _triYC[_tri,0] - _triYC[_tri,2] )
		
		( Zvec_p1p2, Zvec_p2p3, Zvec_p3p1 ) = \
		( _triZC[_tri,1] - _triZC[_tri,0], _triZC[_tri,2] - _triZC[_tri,1], _triZC[_tri,0] - _triZC[_tri,2] )
		
		_Matrix_ = \
		np.array([[Xvec_p1p2,Xvec_p2p3,Xvec_p3p1], [Yvec_p1p2,Yvec_p2p3,Yvec_p3p1], [Zvec_p1p2,Zvec_p2p3,Zvec_p3p1], [1.,1.,1.]])
		
		_Xvor_ = _Triangle_Area_[_tri]*_eleGma_[_tri,0]
		_Yvor_ = _Triangle_Area_[_tri]*_eleGma_[_tri,1]
		_Zvor_ = _Triangle_Area_[_tri]*_eleGma_[_tri,2] 
		
		_Vec_ = np.array([[_Xvor_], [_Yvor_], [_Zvor_], [0.]])
		
		#tmp = np.matmul(np.linalg.pinv(_Matrix_), _Vec_) 
		tmp, _, _, _ = lstsq(_Matrix_, _Vec_)
		
		#if _tri == 0: print(tmp)
		
		for i in range(3): _nodeCirc[_tri, i] = tmp[i]
		
	return _nodeCirc
	
	
class _node_Circulation(object):

	def __init__(self, _nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma_):
		self._nodeCoord = _nodeCoord
		
		self._triXC = _triXC
		self._triYC = _triYC
		self._triZC = _triZC
		
		self._no_tri_ = _no_tri_
		self.tri_verti_ = tri_verti_
		self._eleGma_ = _eleGma_
	
	def _change_in_Circu(self):
	
		_nodeCirc = \
		_MatrixInversion_2_cal_nodeCircu_frm_eleVor_\
		( self._nodeCoord, self._triXC, self._triYC, self._triZC, self._no_tri_, self.tri_verti_, self._eleGma_ )
	
		return _nodeCirc


def _cal_gridVoricity_frm_eleVorStrgth_( _eleGma0_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC ):
	
	_Area_ = _allTriangle_Area_( _triXC, _triYC, _triZC, _no_tri_, tri_verti_ )
	_tri_centroid_ = _allTriangle_Centroid_( _triXC, _triYC, _triZC, _no_tri_, tri_verti_ )
	_all_centroid_coord_ = _remeshing_all_triangles_( _no_tri_, _triXC, _triYC, _triZC )
	
	print( 'max vor strength = ', np.max(_eleGma0_[:,1]), 'min vor strength = ', np.min(_eleGma0_[:,1]) )
	
	start = time.time()
	x, y, z = _SmoothGrid_Generation_()
	
	ds = np.zeros(3)
	( ds[0], ds[1], ds[2] ) = ( abs(x[1]-x[0]), abs(y[1]-y[0]), abs(z[1]-z[0]) )
	
	x3d, y3d, z3d = np.meshgrid(x, y, z)
	( x3d, y3d, z3d ) = ( x3d.flatten(), y3d.flatten(), z3d.flatten() )
	
	shared_array_base = mp.Array( ctypes.c_double, len(x3d)*3 )
	
	nprocs = 25 #mp.cpu_count()
	if len(x3d)%nprocs != 0:
		raise ValueError ( 'Change number of processors to make len(x3d)%nprocs = 0', len(x3d) )
	
	inputs = \
	[(rank, nprocs, _eleGma0_, x3d, y3d, z3d, _tri_centroid_, _all_centroid_coord_, ds, _Area_ ) \
	for rank in range(nprocs)]
	
	pool = mp.Pool( processes=nprocs, initializer=_init_, initargs=(shared_array_base, len(x3d), 3,) )
	pool.starmap( _sum_in_parallel_A_, inputs )
	pool.close()
	
	shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())	
	shared_array = shared_array.reshape( len(x3d), 3 )
	#print( 'shape = ',  np.shape(shared_array) )
	
	#for i in range(len(x3d)):
	#	if x3d[i] == 0. and y3d[i] == 0. and z3d[i] == 0.: print(shared_array[i,1])
	
	( nx, ny, nz ) = ( len(x), len(y), len(z) )
	
	_gridVorX_ = np.reshape( shared_array[:,0], ( ny, nx, nz ) )
	_gridVorY_ = np.reshape( shared_array[:,1], ( ny, nx, nz ) )
	_gridVorZ_ = np.reshape( shared_array[:,2], ( ny, nx, nz ) )
	
	#( _gridVorX_[ny-1,:,:], _gridVorY_[ny-1,:,:], _gridVorZ_[ny-1,:,:] ) = ( _gridVorX_[0,:,:], _gridVorY_[0,:,:], _gridVorZ_[0,:,:] ) 
	#( _gridVorX_[:,nx-1,:], _gridVorY_[:,nx-1,:], _gridVorZ_[:,nx-1,:] ) = ( _gridVorX_[:,0,:], _gridVorY_[:,0,:], _gridVorZ_[:,0,:] ) 
	#( _gridVorX_[:,:,nz-1], _gridVorY_[:,:,nz-1], _gridVorZ_[:,:,nz-1] ) = ( _gridVorX_[:,:,0], _gridVorY_[:,:,0], _gridVorZ_[:,:,0] )
	
	sio.savemat('VorY.mat', {'VorY':_gridVorY_})
	
	end = time.time()
	print('calculation of grid-vorticity is done - ', end - start)
	
	return _gridVorX_, _gridVorY_, _gridVorZ_
	
	
def _sum_in_parallel_A_( rank, nprocs, _eleGma_, x3d, y3d, z3d, _tri_centroid_, _all_centroid_coord_, ds, _Area_ ):
	
	#_gridVor_ = np.zeros( (len(x3d),3) ) 
	
	for itr in range( rank, len(x3d), nprocs ):
		#if rank == 0:  print('itr= ', itr, 'rank = ', rank)
		#if rank == 24: print( 'itr= ', itr, 'rank = ', rank )
		if itr==len(x3d)-1: print( 'itr= ', itr, 'rank = ', rank )
		
		coord = np.array( [ x3d[itr], y3d[itr], z3d[itr]  ] )
		
		sum_x, sum_y, sum_z = \
		_D3b_Peskin_function_toget_gridVor_( _eleGma_, _tri_centroid_, _all_centroid_coord_, coord, ds, _Area_ )
		
		shared_array[itr,0] = sum_x/( ds[0]*ds[1]*ds[2] )   # ds[0]**3. 
		shared_array[itr,1] = sum_y/( ds[0]*ds[1]*ds[2] )   # ds[0]**3.
		shared_array[itr,2] = sum_z/( ds[0]*ds[1]*ds[2] )   # ds[0]**3. 
		
		#if coord[0] == 0. and coord[1] == 0. and coord[2] == 0.: print( rank, itr, shared_array[itr,1] )
		
	#return shared_array #_gridVor_ 
	
	
def _cal_rhs_( _gridVorX_, _gridVorY_, _gridVorZ_, dx, dy, dz ):

	( nx, ny, nz ) = ( _gridVorX_.shape[1], _gridVorX_.shape[0], _gridVorX_.shape[2] )
	
	_gridVorX_dY_ = np.zeros( (ny, nx, nz) ) 
	_gridVorX_dZ_ = np.zeros( (ny, nx, nz) ) 
		
	_gridVorY_dX_ = np.zeros( (ny, nx, nz) )
	_gridVorY_dZ_ = np.zeros( (ny, nx, nz) )
		
	_gridVorZ_dX_ = np.zeros( (ny, nx, nz) )
	_gridVorZ_dY_ = np.zeros( (ny, nx, nz) )	
	
# ------------------------------------
	for i in range(2,nx-2):
		_gridVorY_dX_[:,i,:] = (_gridVorY_[:,i-2,:]/12. - 2.*_gridVorY_[:,i-1,:]/3. + 2.*_gridVorY_[:,i+1,:]/3. - _gridVorY_[:,i+2,:]/12.)/dx
		_gridVorZ_dX_[:,i,:] = (_gridVorZ_[:,i-2,:]/12. - 2.*_gridVorZ_[:,i-1,:]/3. + 2.*_gridVorZ_[:,i+1,:]/3. - _gridVorZ_[:,i+2,:]/12.)/dx		
	
	for i in range(0,2):
		_gridVorY_dX_[:,i,:] = (-3.*_gridVorY_[:,i,:]/2. + 2.*_gridVorY_[:,i+1,:] - _gridVorY_[:,i+2,:]/2.)/dx
		_gridVorZ_dX_[:,i,:] = (-3.*_gridVorZ_[:,i,:]/2. + 2.*_gridVorZ_[:,i+1,:] - _gridVorZ_[:,i+2,:]/2.)/dx
	
	_gridVorY_dX_[:,nx-2,:] = (3.*_gridVorY_[:,nx-2,:]/2. - 2.*_gridVorY_[:,nx-3,:] + _gridVorY_[:,nx-4,:]/2.)/dx
	_gridVorZ_dX_[:,nx-2,:] = (3.*_gridVorZ_[:,nx-2,:]/2. - 2.*_gridVorZ_[:,nx-3,:] + _gridVorZ_[:,nx-4,:]/2.)/dx
	
	_gridVorY_dX_[:,nx-1,:] = _gridVorY_dX_[:,0,:]  
	_gridVorZ_dX_[:,nx-1,:] = _gridVorZ_dX_[:,0,:] 
	
# ------------------------------------
	for j in range(2,ny-2):
		_gridVorX_dY_[j,:,:] = (_gridVorX_[j-2,:,:]/12. - 2.*_gridVorX_[j-1,:,:]/3. + 2.*_gridVorX_[j+1,:,:]/3. - _gridVorX_[j+2,:,:]/12.)/dy
		_gridVorZ_dY_[j,:,:] = (_gridVorZ_[j-2,:,:]/12. - 2.*_gridVorZ_[j-1,:,:]/3. + 2.*_gridVorZ_[j+1,:,:]/3. - _gridVorZ_[j+2,:,:]/12.)/dy
	
	for j in range(0,2):
		_gridVorX_dY_[j,:,:] = (-3.*_gridVorX_[j,:,:]/2. + 2.*_gridVorX_[j+1,:,:] - _gridVorX_[j+2,:,:]/2. )/dy
		_gridVorZ_dY_[j,:,:] = (-3.*_gridVorZ_[j,:,:]/2. + 2.*_gridVorZ_[j+1,:,:] - _gridVorZ_[j+2,:,:]/2. )/dy
	
	_gridVorX_dY_[ny-2,:,:] = (3.*_gridVorX_[ny-2,:,:]/2. - 2.*_gridVorX_[ny-3,:,:] + _gridVorX_[ny-4,:,:]/2.)/dy
	_gridVorZ_dY_[ny-2,:,:] = (3.*_gridVorZ_[ny-2,:,:]/2. - 2.*_gridVorZ_[ny-3,:,:] + _gridVorZ_[ny-4,:,:]/2.)/dy
	
	_gridVorX_dY_[ny-1,:,:] = _gridVorX_dY_[0,:,:]  
	_gridVorZ_dY_[ny-1,:,:] = _gridVorZ_dY_[0,:,:] 
	
# ------------------------------------		
	for k in range(2,nz-2):
		_gridVorX_dZ_[:,:,k] = (_gridVorX_[:,:,k-2]/12. - 2.*_gridVorX_[:,:,k-1]/3. + 2.*_gridVorX_[:,:,k+1]/3. - _gridVorX_[:,:,k+2]/12.)/dz
		_gridVorY_dZ_[:,:,k] = (_gridVorY_[:,:,k-2]/12. - 2.*_gridVorY_[:,:,k-1]/3. + 2.*_gridVorY_[:,:,k+1]/3. - _gridVorY_[:,:,k+2]/12.)/dz
	
	for k in range(0,2):
		_gridVorX_dZ_[:,:,k] = (-3.*_gridVorX_[:,:,k]/2. + 2.*_gridVorX_[:,:,k+1] - _gridVorX_[:,:,k+2]/2. )/dz
		_gridVorY_dZ_[:,:,k] = (-3.*_gridVorY_[:,:,k]/2. + 2.*_gridVorY_[:,:,k+1] - _gridVorY_[:,:,k+2]/2. )/dz
	
	_gridVorX_dZ_[:,:,nz-2] = (3.*_gridVorX_[:,:,nz-2]/2. - 2.*_gridVorX_[:,:,nz-3] + _gridVorX_[:,:,nz-4]/2.)/dz
	_gridVorY_dZ_[:,:,nz-2] = (3.*_gridVorY_[:,:,nz-2]/2. - 2.*_gridVorY_[:,:,nz-3] + _gridVorY_[:,:,nz-4]/2.)/dz
	
	_gridVorX_dZ_[:,:,nz-1] = _gridVorX_dZ_[:,:,0]  
	_gridVorY_dZ_[:,:,nz-1] = _gridVorY_dZ_[:,:,0]
	
# ------ enforce periodic boundary condition 
	( _gridVorX_dY_[0,:,:], _gridVorX_dZ_[0,:,:], _gridVorY_dX_[0,:,:], _gridVorY_dZ_[0,:,:], _gridVorZ_dX_[0,:,:], _gridVorZ_dY_[0,:,:] ) = \
	( _gridVorX_dY_[ny-1,:,:], _gridVorX_dZ_[ny-1,:,:], _gridVorY_dX_[ny-1,:,:], _gridVorY_dZ_[ny-1,:,:], _gridVorZ_dX_[ny-1,:,:], _gridVorZ_dY_[ny-1,:,:] )
	
	( _gridVorX_dY_[:,0,:], _gridVorX_dZ_[:,0,:], _gridVorY_dX_[:,0,:], _gridVorY_dZ_[:,0,:], _gridVorZ_dX_[:,0,:], _gridVorZ_dY_[:,0,:] ) = \
	( _gridVorX_dY_[:,nx-1,:], _gridVorX_dZ_[:,nx-1,:], _gridVorY_dX_[:,nx-1,:], _gridVorY_dZ_[:,nx-1,:], _gridVorZ_dX_[:,nx-1,:], _gridVorZ_dY_[:,nx-1,:] )
	
	( _gridVorX_dY_[:,:,0], _gridVorX_dZ_[:,:,0], _gridVorY_dX_[:,:,0], _gridVorY_dZ_[:,:,0], _gridVorZ_dX_[:,:,0], _gridVorZ_dY_[:,:,0] ) = \
	( _gridVorX_dY_[:,:,nz-1], _gridVorX_dZ_[:,:,nz-1], _gridVorY_dX_[:,:,nz-1], _gridVorY_dZ_[:,:,nz-1], _gridVorZ_dX_[:,:,nz-1], _gridVorZ_dY_[:,:,nz-1] ) 
	
	u_frhs = -1.*( _gridVorZ_dY_ - _gridVorY_dZ_ )
	v_frhs = -1.*( _gridVorX_dZ_ - _gridVorZ_dX_ )
	w_frhs = -1.*( _gridVorY_dX_ - _gridVorX_dY_ )
	
	return u_frhs, v_frhs, w_frhs
	
	
def _D3_PoissonSolver_fft_(_eleGma_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC):

	x,y,z = _SmoothGrid_Generation_()
	( dx, dy, dz ) = ( abs(x[1]-x[0]), abs(y[1]-y[0]), abs(z[1]-z[0]) )
	#x3d, y3d, z3d = np.meshgrid( x, y, z )
	
	( nx, ny, nz ) = ( len(x), len(y), len(z) )
	
	kx = np.fft.fftfreq( nx )/dx
	ky = np.fft.fftfreq( ny )/dy
	kz = np.fft.fftfreq( nz )/dz
	
	kx3d, ky3d, kz3d = np.meshgrid( kx, ky, kz )
	
	_gridVorX_, _gridVorY_, _gridVorZ_ = \
	_cal_gridVoricity_frm_eleVorStrgth_(_eleGma_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
		
	u_frhs, v_frhs, w_frhs = _cal_rhs_(_gridVorX_, _gridVorY_, _gridVorZ_, dx, dy, dz)
	
# --------------------------------------
	Lx, Ly, Lz = Domain()
	Lz = 2.*Lz 
	
	# FFT of of u-vel of RHS
	F = np.fft.fftn(u_frhs)
	oldDC = F[0,0,0]
	np.seterr(divide='ignore', invalid='ignore')
	F = 0.5/(Lx*Ly*Lz)*F/( (np.cos(2.*np.pi*kx3d/nx) - 1.)/dx**2. + (np.cos(2.*np.pi*ky3d/ny) - 1.)/dy**2. \
	+ (np.cos(2.0*np.pi*kz3d/nz) - 1.)/dz**2. )
	F[0,0,0] = oldDC
	 
	# transform back to real space
	_gridVelX_ = np.real( np.fft.ifftn(F) )
	
# --------------------------------------
	# FFT of of v-vel of RHS
	F = np.fft.fftn(v_frhs)
	oldDC = F[0,0,0]
	np.seterr(divide='ignore', invalid='ignore')
	F = 0.5/(Lx*Ly*Lz)*F/( (np.cos(2.*np.pi*kx3d/nx) - 1.)/dx**2. + (np.cos(2.*np.pi*ky3d/ny) - 1.)/dy**2. \
	+ (np.cos(2.*np.pi*kz3d/nz) - 1.)/dz**2. )
	F[0,0,0] = oldDC
	
	# transform back to real space
	_gridVelY_ = np.real( np.fft.ifftn(F) )
	
# --------------------------------------
	# FFT of of w-vel of RHS
	F = np.fft.fftn(w_frhs)
	oldDC = F[0,0,0]
	np.seterr(divide='ignore', invalid='ignore')
	F = 0.5/(Lx*Ly*Lz)*F/( (np.cos(2.*np.pi*kx3d/nx) - 1.)/dx**2. + (np.cos(2.*np.pi*ky3d/ny) - 1.)/dy**2. \
	+ (np.cos(2.*np.pi*kz3d/nz) - 1.)/dz**2. )
	F[0,0,0] = oldDC
	 
	# transform back to real space
	_gridVelZ_ = np.real( np.fft.ifftn(F) )
	
	( _gridVelX_[0,:,:], _gridVelY_[0,:,:], _gridVelZ_[0,:,:] ) = ( _gridVelX_[ny-1,:,:], _gridVelY_[ny-1,:,:], _gridVelZ_[ny-1,:,:] ) 
	( _gridVelX_[:,0,:], _gridVelY_[:,0,:], _gridVelZ_[:,0,:] ) = ( _gridVelX_[:,nx-1,:], _gridVelY_[:,nx-1,:], _gridVelZ_[:,nx-1,:] ) 
	( _gridVelX_[:,:,0], _gridVelY_[:,:,0], _gridVelZ_[:,:,0] ) = ( _gridVelX_[:,:,nz-1], _gridVelY_[:,:,nz-1], _gridVelZ_[:,:,nz-1] )  
	
	sio.savemat('VelX.mat', {'VelX':_gridVelX_})
	sio.savemat('VelY.mat', {'VelY':_gridVelY_})
	sio.savemat('VelZ.mat', {'VelZ':_gridVelZ_})
	
	return _gridVelX_, _gridVelY_, _gridVelZ_


def _cal_nodeVelocity_frm_gridVelocity(_eleGma_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC):
	
	start = time.time()
	_gridVelX_, _gridVelY_, _gridVelZ_ = \
	_D3_PoissonSolver_fft_(_eleGma_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
	
	end = time.time()
	print('calculation of grid-velocity is done - ', end - start )
	
	start = time.time()
	D3_gridKE = D3_KE_gridVel_(_gridVelX_, _gridVelY_, _gridVelZ_)
	
	_gridVelX_ = _gridVelX_.flatten()
	_gridVelY_ = _gridVelY_.flatten()
	_gridVelZ_ = _gridVelZ_.flatten()
	
	x, y, z = _SmoothGrid_Generation_()
	ds = np.zeros(3)
	( ds[0], ds[1], ds[2] ) = ( abs(x[1]-x[0]), abs(y[1]-y[0]), abs(z[1]-z[0]) )
	
	x3d, y3d, z3d = np.meshgrid(x, y, z)
	x3d = x3d.flatten()
	y3d = y3d.flatten()
	z3d = z3d.flatten()
	
	
	nprocs = 25 #mp.cpu_count()
	
	if len(_nodeXC)%nprocs == 0:
		
		shared_array_base = mp.Array( ctypes.c_double, len(_nodeXC)*3 )
		
		inputs = \
		[(rank, nprocs, _gridVelX_, _gridVelY_, _gridVelZ_, _nodeXC, _nodeYC, _nodeZC, x3d, y3d, z3d, ds ) for rank in range(nprocs)]
			
		pool = mp.Pool(processes=nprocs, initializer=_init_, initargs=(shared_array_base, len(_nodeXC), 3,))
		#_nodeVel_ = pool.starmap( _sum_in_parallel_B_, inputs )
		pool.starmap( _sum_in_parallel_B_, inputs )
		pool.close()
		
		shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
		shared_array = shared_array.reshape( len(_nodeXC), 3 )
		print( np.shape(shared_array) )
		
		_uXg0_ = shared_array[:,0]
		_vYg0_ = shared_array[:,1]
		_wZg0_ = shared_array[:,2]
		
	else:
		
		remainder_ = len(_nodeXC)%nprocs
		_len_ = len(_nodeXC) - remainder_
		print('len(_nodeXC)%nprocs is not 0: remainder list of', remainder_, 'will run on single processor')
		
		shared_array_base = mp.Array( ctypes.c_double, _len_*3 )
		
		inputs = \
		[(rank, nprocs, _gridVelX_, _gridVelY_, _gridVelZ_, _nodeXC[0:_len_], _nodeYC[0:_len_], _nodeZC[0:_len_], x3d, y3d, z3d, ds ) for rank in range(nprocs)]
		
		pool = mp.Pool(processes=nprocs, initializer=_init_, initargs=(shared_array_base, _len_, 3,))
		#_nodeVel_ = pool.starmap( _sum_in_parallel_B_, inputs )
		pool.starmap( _sum_in_parallel_B_, inputs )
		pool.close()
		
		shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
		shared_array = shared_array.reshape( _len_, 3 )
		print( 'len_shared_array = ', len(shared_array[:,0]) )
		
		_array = _sum_in_single_B_\
		( _gridVelX_, _gridVelY_, _gridVelZ_, _nodeXC[_len_:len(_nodeXC)], _nodeYC[_len_:len(_nodeXC)], _nodeZC[_len_:len(_nodeXC)], x3d, y3d, z3d, ds )
		
		_array = np.concatenate( (shared_array, _array) )
		
		_uXg0_ = _array[:,0]
		_vYg0_ = _array[:,1]
		_wZg0_ = _array[:,2]
		
		print( '_len_uXg0_ (single+parallel) = ', len(_uXg0_) )
	
	end = time.time()
	print('calculation of node-velocity is done - ', end - start )
	
	return _uXg0_, _vYg0_, _wZg0_, D3_gridKE


def _sum_in_parallel_B_( rank, nprocs, _gridVelX_, _gridVelY_, _gridVelZ_, _nodeXC0, _nodeYC0, _nodeZC0, x3d, y3d, z3d, ds ):
	
	for itr in range( rank, len(_nodeXC0), nprocs ):
		_nodeC = np.array( [ _nodeXC0[itr], _nodeYC0[itr], _nodeZC0[itr] ] )
		sum_x, sum_y, sum_z = \
		_D3b_Peskin_function_toget_nodeVel_( _gridVelX_, _gridVelY_, _gridVelZ_, _nodeC, x3d, y3d, z3d, ds )
			
		shared_array[itr,0] = sum_x
		shared_array[itr,1] = sum_y
		shared_array[itr,2] = sum_z
	
	#return shared_array # _nodeVel_
	
	
def _sum_in_single_B_( _gridVelX_, _gridVelY_, _gridVelZ_, _nodeXC0, _nodeYC0, _nodeZC0, x3d, y3d, z3d, ds ):
	
	_array = np.zeros( (len(_nodeXC0),3) )
	for itr in range( len(_nodeXC0) ):
		_nodeC = np.array( [ _nodeXC0[itr], _nodeYC0[itr], _nodeZC0[itr] ] )
		sum_x, sum_y, sum_z = \
		_D3b_Peskin_function_toget_nodeVel_( _gridVelX_, _gridVelY_, _gridVelZ_, _nodeC, x3d, y3d, z3d, ds )
			
		_array[itr,0] = sum_x
		_array[itr,1] = sum_y
		_array[itr,2] = sum_z
	
	return _array 
	
	
	
def _cal_eleVorStrgth_frm_nodeCirculation( _nodeTau_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC ):

	_eleGmaX0_ = np.zeros( _no_tri_ )
	_eleGmaY0_ = np.zeros( _no_tri_ )
	_eleGmaZ0_ = np.zeros( _no_tri_ )
				
	_Area_ = _allTriangle_Area_( _triXC, _triYC, _triZC, _no_tri_, tri_verti_ )
	
	#Obj_ = _elements_splitting_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, del_split )
	#_TriVert_p1C, _TriVert_p2C, _TriVert_p3C = Obj_._TriVert_Cord() 
	
	for _tri in range(_no_tri_):
		
		#( Xvec_p1p2, Xvec_p2p3, Xvec_p3p1 ) = \
		#(_TriVert_p2C[_tri,0]-_TriVert_p1C[_tri,0], _TriVert_p3C[_tri,0]-_TriVert_p2C[_tri,0], _TriVert_p1C[_tri,0]-_TriVert_p3C[_tri,0]) 
		
		#( Yvec_p1p2, Yvec_p2p3, Yvec_p3p1 ) = \
		#(_TriVert_p2C[_tri,1]-_TriVert_p1C[_tri,1], _TriVert_p3C[_tri,1]-_TriVert_p2C[_tri,1], _TriVert_p1C[_tri,1]-_TriVert_p3C[_tri,1]) 
		
		#( Zvec_p1p2, Zvec_p2p3, Zvec_p3p1 ) = \
		#(_TriVert_p2C[_tri,2]-_TriVert_p1C[_tri,2], _TriVert_p3C[_tri,2]-_TriVert_p2C[_tri,2], _TriVert_p1C[_tri,2]-_TriVert_p3C[_tri,2]) 
		
		( Xvec_p1p2, Xvec_p2p3, Xvec_p1p3 ) = \
		( _triXC[_tri,1] - _triXC[_tri,0], _triXC[_tri,2] - _triXC[_tri,1], _triXC[_tri,0] - _triXC[_tri,2] )
		
		( Yvec_p1p2, Yvec_p2p3, Yvec_p1p3 ) = \
		( _triYC[_tri,1] - _triYC[_tri,0], _triYC[_tri,2] - _triYC[_tri,1], _triYC[_tri,0] - _triYC[_tri,2] )
		
		( Zvec_p1p2, Zvec_p2p3, Zvec_p1p3 ) = \
		( _triZC[_tri,1] - _triZC[_tri,0], _triZC[_tri,2] - _triZC[_tri,1], _triZC[_tri,0] - _triZC[_tri,2] )
		
		_eleGmaX0_[_tri]  = ( _nodeTau_[_tri,0]*Xvec_p1p2 + _nodeTau_[_tri,1]*Xvec_p2p3 + _nodeTau_[_tri,2]*Xvec_p1p3 )/_Area_[_tri]
		_eleGmaY0_[_tri]  = ( _nodeTau_[_tri,0]*Yvec_p1p2 + _nodeTau_[_tri,1]*Yvec_p2p3 + _nodeTau_[_tri,2]*Yvec_p1p3 )/_Area_[_tri]
		_eleGmaZ0_[_tri]  = ( _nodeTau_[_tri,0]*Zvec_p1p2 + _nodeTau_[_tri,1]*Zvec_p2p3 + _nodeTau_[_tri,2]*Zvec_p1p3 )/_Area_[_tri]
		
	return _eleGmaX0_, _eleGmaY0_, _eleGmaZ0_
	
	
def D3_KE_gridVel_(_gridVelX_, _gridVelY_, _gridVelZ_):
	
	#_gridVelX_, _gridVelY_, _gridVelZ_ = \
	#_D3_PoissonSolver_fft_(_eleGma_, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
	
	_gridVelX_ = _gridVelX_.flatten()
	_gridVelY_ = _gridVelY_.flatten()
	_gridVelZ_ = _gridVelZ_.flatten()
	
	x, y, z = _SmoothGrid_Generation_()
	ds = np.zeros(3)
	( ds[0], ds[1], ds[2] ) = ( abs(x[1]-x[0]), abs(y[1]-y[0]), abs(z[1]-z[0]) )
	
	D3_KE = 0.
	
	for itr in range( len(_gridVelX_) ):
		D3_KE = D3_KE + ( _gridVelX_[itr]**2. + _gridVelY_[itr]**2. + _gridVelZ_[itr]**2. )*ds[0]*ds[1]*ds[2]
		
	D3_KE = 0.5*D3_KE
	
	return D3_KE
	
