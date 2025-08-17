import numpy as np
import multiprocessing as mp
import scipy.io as sio
import unittest

def _sum_in_parellel_BioSavart_Uvel(rank, nprocs, Arg0_, Arg1_, Arg2_, _Triangle_Area_):

	delta_ = 0.1	
	_no_node = len(Arg0_[:,0])
	_no_triangle_ = len(Arg1_[:,0])
	_tmp_ = np.zeros(_no_node)
	
	for itr in range(rank, _no_node, nprocs):
		_sum_ = 0.
		#if itr == 2497: print('itr = ', itr)
		for jtr in range(_no_triangle_):	
		
			_rij_2 = (Arg0_[itr,0]-Arg2_[jtr,0])**2. + (Arg0_[itr,1]-Arg2_[jtr,1])**2. + (Arg0_[itr,2]-Arg2_[jtr,2])**2. + delta_**2.
						
			_sum_ += ( (Arg0_[itr,1]-Arg2_[jtr,1])*Arg1_[jtr,2] - (Arg0_[itr,2]-Arg2_[jtr,2])*Arg1_[jtr,1] )*_Triangle_Area_[jtr]/_rij_2**1.5

		_tmp_[itr] = -_sum_/(4.*np.pi)
										
	return _tmp_	


def _sum_in_parellel_BioSavart_Vvel(rank, nprocs, Arg0_, Arg1_, Arg2_, _Triangle_Area_):

	delta_ = 0.1	
	_no_node = len(Arg0_[:,0])
	_no_triangle_ = len(Arg1_[:,0])
	_tmp_ = np.zeros(_no_node)
	
	for itr in range(rank, _no_node, nprocs):
		_sum_ = 0.
		for jtr in range(_no_triangle_):

			_rij_2 = (Arg0_[itr,0]-Arg2_[jtr,0])**2. + (Arg0_[itr,1]-Arg2_[jtr,1])**2. + (Arg0_[itr,2]-Arg2_[jtr,2])**2. + delta_**2.
						
			_sum_ += ( (Arg0_[itr,2]-Arg2_[jtr,2])*Arg1_[jtr,0] - (Arg0_[itr,0]-Arg2_[jtr,0])*Arg1_[jtr,2] )*_Triangle_Area_[jtr]/_rij_2**1.5

		_tmp_[itr] = -_sum_/(4.*np.pi)
										
	return _tmp_	
	

def _sum_in_parellel_BioSavart_Wvel(rank, nprocs, Arg0_, Arg1_, Arg2_, _Triangle_Area_):

	delta_ = 0.1
	_no_node = len(Arg0_[:,0])
	_no_triangle_ = len(Arg1_[:,0])
	_tmp_ = np.zeros(_no_node)
	
	for itr in range(rank, _no_node, nprocs):
		_sum_ = 0.
		for jtr in range(_no_triangle_):

			_rij_2 = (Arg0_[itr,0]-Arg2_[jtr,0])**2. + (Arg0_[itr,1]-Arg2_[jtr,1])**2. + (Arg0_[itr,2]-Arg2_[jtr,2])**2. + delta_**2.
						
			_sum_ += ( (Arg0_[itr,0]-Arg2_[jtr,0])*Arg1_[jtr,1] - (Arg0_[itr,1]-Arg2_[jtr,1])*Arg1_[jtr,0] )*_Triangle_Area_[jtr]/_rij_2**1.5
		
		_tmp_[itr] = -_sum_/(4.*np.pi)
		#print(_sum_)								
	return _tmp_	
			

def _cal_Upv_gRidp_( _nodeCoord, _eleGma_, delta_, _Centroid_, _Triangle_Area_ ):
		
	Arg0_ = _nodeCoord 
	Arg1_ = _eleGma_ 
	Arg2_ = _Centroid_
	
	nprocs = mp.cpu_count()
	inputs = [(rank, nprocs, Arg0_, Arg1_, Arg2_, _Triangle_Area_) for rank in range(nprocs)]
	
	#assertEqual( len(Arg0_[:,0])%nprocs, 0 )
	
	pool = mp.Pool(processes=nprocs)					
	_uX0_ = pool.starmap( _sum_in_parellel_BioSavart_Uvel, inputs )
	_vY0_ = pool.starmap( _sum_in_parellel_BioSavart_Vvel, inputs )
	_wZ0_ = pool.starmap( _sum_in_parellel_BioSavart_Wvel, inputs )
	pool.close()
	

#	_uX_ = _sum_in_BioSavart_Uvel( Arg0_, Arg1_, Arg2_, _Triangle_Area_ )
#	_vY_ = _sum_in_BioSavart_Vvel( Arg0_, Arg1_, Arg2_, _Triangle_Area_ )
#	_wZ_ = _sum_in_BioSavart_Wvel( Arg0_, Arg1_, Arg2_, _Triangle_Area_ )
		
	_uX_ = _uX0_[0]  
	_vY_ = _vY0_[0]  
	_wZ_ = _wZ0_[0]   
	
	for _id in range(1, nprocs):
		_uX_ += _uX0_[_id]
		_vY_ += _vY0_[_id]
		_wZ_ += _wZ0_[_id]

										
	return _uX_, _vY_, _wZ_		
	

class Vel3d_(object):

	def __init__(self, _nodeCoord, _eleGma, delta_, _Centroid_, _Triangle_Area_):
		self._nodeCoord = _nodeCoord
		self._eleGma = _eleGma
		self.delta_ = delta_
		self._Centroid_ = _Centroid_
		self._Triangle_Area_ = _Triangle_Area_
		

	def _update_Upv_gRrip_(self):
	
		_uXg_, _vYg_, _wZg_ = \
		_cal_Upv_gRidp_(self._nodeCoord, self._eleGma, self.delta_, self._Centroid_, self._Triangle_Area_)
			
		return _uXg_, _vYg_, _wZg_
		


def _sum_in_BioSavart_Uvel(Arg0_, Arg1_, Arg2_, _Triangle_Area_):

	delta_ = 0.
	_no_node = len(Arg0_[:,0])
	_no_triangle_ = len(Arg1_[:,0])
	_tmp_ = np.zeros(_no_node)
	
	for itr in range(_no_node):
		_sum_ = 0.
		for jtr in range(_no_triangle_):	
		
			_rij_2 = (Arg0_[itr,0]-Arg2_[jtr,0])**2. + (Arg0_[itr,1]-Arg2_[jtr,1])**2. + (Arg0_[itr,2]-Arg2_[jtr,2])**2. + delta_**2.
						
			_sum_ += ( (Arg0_[itr,1]-Arg2_[jtr,1])*Arg1_[jtr,2] - (Arg0_[itr,2]-Arg2_[jtr,2])*Arg1_[jtr,1] )*_Triangle_Area_[jtr]/_rij_2**1.5

		_tmp_[itr] = -_sum_/(4.*np.pi)
										
	return _tmp_	


def _sum_in_BioSavart_Vvel(Arg0_, Arg1_, Arg2_, _Triangle_Area_):

	delta_ = 0.1	
	_no_node = len(Arg0_[:,0])
	_no_triangle_ = len(Arg1_[:,0])
	_tmp_ = np.zeros(_no_node)
	
	for itr in range(_no_node):
		_sum_ = 0.
		for jtr in range(_no_triangle_):

			_rij_2 = (Arg0_[itr,0]-Arg2_[jtr,0])**2. + (Arg0_[itr,1]-Arg2_[jtr,1])**2. + (Arg0_[itr,2]-Arg2_[jtr,2])**2. + delta_**2.
						
			_sum_ += ( (Arg0_[itr,2]-Arg2_[jtr,2])*Arg1_[jtr,0] - (Arg0_[itr,0]-Arg2_[jtr,0])*Arg1_[jtr,2] )*_Triangle_Area_[jtr]/_rij_2**1.5

		_tmp_[itr] = -_sum_/(4.*np.pi)
										
	return _tmp_	
	

def _sum_in_BioSavart_Wvel(Arg0_, Arg1_, Arg2_, _Triangle_Area_):

	delta_ = 0.1	
	_no_node = len(Arg0_[:,0])
	_no_triangle_ = len(Arg1_[:,0])
	_tmp_ = np.zeros(_no_node)
	
	for itr in range(_no_node):
		_sum_ = 0.
		for jtr in range(_no_triangle_):

			_rij_2 = (Arg0_[itr,0]-Arg2_[jtr,0])**2. + (Arg0_[itr,1]-Arg2_[jtr,1])**2. + (Arg0_[itr,2]-Arg2_[jtr,2])**2. + delta_**2.
						
			_sum_ += ( (Arg0_[itr,0]-Arg2_[jtr,0])*Arg1_[jtr,1] - (Arg0_[itr,1]-Arg2_[jtr,1])*Arg1_[jtr,0] )*_Triangle_Area_[jtr]/_rij_2**1.5

		_tmp_[itr] = -_sum_/(4.*np.pi)
								
	return _tmp_				
	
