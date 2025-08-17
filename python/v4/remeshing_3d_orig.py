import numpy as np
import scipy.io as sio
import matplotlib.tri as mtri
from scipy.interpolate import Rbf
from scipy.interpolate import RegularGridInterpolator

from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

from utility_3d_paral import _cal_Normal_Vector
from utility_3d_paral import _allTriangle_Area_, _Enforce_Predic_Cond_on_nodeCoord_

from init_3d import _finding_boundary_nodes_, _boundary_nodes_

def _euclid_3d_dist(x1, y1, z1, x2, y2, z2):
	
	dist = np.sqrt( (x1-x2)**2. + (y1-y2)**2. + (z1-z2)**2. )
	
	return dist
	
	
def _euclid_p2p_3d_dist(p1_C, p2_C):
	
	_tmp_ = np.sqrt( (p1_C[0]-p2_C[0])**2. + (p1_C[1]-p2_C[1])**2. + (p1_C[2]-p2_C[2])**2. )
	
	return _tmp_
	
	
def _Centroid_triangle_(p1_C, p2_C, p3_C):
	
	_centroid_ = np.zeros(3)
	_centroid_[0] = (p1_C[0]+p2_C[0]+p3_C[0])/3.
	_centroid_[1] = (p1_C[1]+p2_C[1]+p3_C[1])/3.
	_centroid_[2] = (p1_C[2]+p2_C[2]+p3_C[2])/3.
	
	return _centroid_
	
	
def _tri_coord_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_):
	
	_triXC = np.zeros( (_no_tri_,3) )
	_triYC = np.zeros( (_no_tri_,3) )
	_triZC = np.zeros( (_no_tri_,3) )
	
	_triXC[0:_no_tri_,0:3] = _nodeXC[tri_verti_[0:_no_tri_,0:3]]
	_triYC[0:_no_tri_,0:3] = _nodeYC[tri_verti_[0:_no_tri_,0:3]]
	_triZC[0:_no_tri_,0:3] = _nodeZC[tri_verti_[0:_no_tri_,0:3]]
	
	return _triXC, _triYC, _triZC
	
	
# store all the nodes around a node around a critical distance
def _allNodes_around_aNode_( _nodeXC, _nodeYC, _nodeZC, _nodeC0, eps ):

	gr_pts = len(_nodeXC)
	_node_list= []
	for _len in range(gr_pts):
	
		_nodeC = np.array( [_nodeXC[_len], _nodeYC[_len], _nodeZC[_len]] )
		_dist = _euclid_p2p_3d_dist(_nodeC0, _nodeC)
		
		if _dist <= eps and _dist != 0.:
			_node_list.append(_len)
	
	_node_list = np.unique(_node_list).tolist()
	length = len(_node_list)
	
	return _node_list, length
	
	
def _sorting_triangles_max_( _edge_len_, ds ):
	
	ind0 = np.argsort(_edge_len_[:,0])
	mxm0 = max(ind0)
	
	ind1 = np.argsort(_edge_len_[:,1])
	mxm1 = max(ind1)
	
	ind2 = np.argsort(_edge_len_[:,2])
	mxm2 = max(ind2)
	
	VAL = np.zeros(3)
	VAL[0] = _edge_len_[ ind0[mxm0], 0 ]
	VAL[1] = _edge_len_[ ind1[mxm1], 1 ]
	VAL[2] = _edge_len_[ ind2[mxm2], 2 ]
	
	MAX = max(VAL)
	POS = [i for i, j in enumerate(VAL) if j == MAX]
	if VAL[ POS[0] ] > ds:
		_edge_ = POS[0] 
		if _edge_ == 0: _tri = ind0[mxm0]
		if _edge_ == 1: _tri = ind1[mxm1]
		if _edge_ == 2: _tri = ind2[mxm2]
		print( '-> element = ', _tri, ' max edge length : ', _edge_len_[_tri,_edge_], '(', ds , ')'  )
	else:
		_tri = -1
		_edge_ = -1
	
	return _tri, _edge_
	
	
def _sorting_triangles_min_( _edge_len_, ds ):
	
	ind0 = np.argsort(_edge_len_[:,0])
	mnm0 = min(ind0)
	
	ind1 = np.argsort(_edge_len_[:,1])
	mnm1 = min(ind1)
	
	ind2 = np.argsort(_edge_len_[:,2])
	mnm2 = min(ind2)
	
	VAL = np.zeros(3)
	VAL[0] = _edge_len_[ ind0[mnm0], 0 ]
	VAL[1] = _edge_len_[ ind1[mnm1], 1 ]
	VAL[2] = _edge_len_[ ind2[mnm2], 2 ]
	
	MIN = min(VAL)
	POS = [i for i, j in enumerate(VAL) if j == MIN]
	if VAL[ POS[0] ] < ds:
		_edge_ = POS[0] 
		if _edge_ == 0: _tri = ind0[mnm0]
		if _edge_ == 1: _tri = ind1[mnm1]
		if _edge_ == 2: _tri = ind2[mnm2]
		print( '-> element = ', _tri, ' min edge length : ', _edge_len_[_tri,_edge_], '(', ds , ')'  )
		if _edge_len_[_tri,_edge_] == 0.: print('edge len = ', VAL )
	else:
		_tri = -1
		_edge_ = -1
	
	return _tri, _edge_
	
	
def _tri_edge_len_( _triXC, _triYC, _triZC ):
	'''	
	this function store the edge-length of all the elements
	'''
	
	_no_tri_ = len(_triXC[:,0])
	_edge_len_ = np.zeros( (_no_tri_,3) )
	
	for _tri in range(_no_tri_):
		
		_edge_len_[_tri,0] = \
		_euclid_3d_dist( _triXC[_tri,0], _triYC[_tri,0], _triZC[_tri,0], _triXC[_tri,1], _triYC[_tri,1], _triZC[_tri,1] )
		
		_edge_len_[_tri,1] = \
		_euclid_3d_dist( _triXC[_tri,1], _triYC[_tri,1], _triZC[_tri,1], _triXC[_tri,2], _triYC[_tri,2], _triZC[_tri,2] )
		
		_edge_len_[_tri,2] = \
		_euclid_3d_dist( _triXC[_tri,2], _triYC[_tri,2], _triZC[_tri,2], _triXC[_tri,0], _triYC[_tri,0], _triZC[_tri,0] )
	
#	sio.savemat( 'edge_len.mat',  {'len':_edge_len_} )
	
	return _edge_len_
	
	
def _max_edge_length_dection_(_triXC, _triYC, _triZC, ds):
	
	_edge_len_ = _tri_edge_len_( _triXC, _triYC, _triZC )
	
	_tri_sorted_, _edge_sorted_ = _sorting_triangles_max_( _edge_len_, ds )
	
	return _tri_sorted_, _edge_sorted_
	
	
def _min_edge_length_dection_(_triXC, _triYC, _triZC, ds):
	
	_edge_len_ = _tri_edge_len_( _triXC, _triYC, _triZC )
	
	_tri_sorted_, _edge_sorted_ = _sorting_triangles_min_( _edge_len_, ds )
	
	if _edge_sorted_ > 0:
		
		_coord0_, _coord1_ = _tri_coord_(_triXC, _triYC, _triZC, _tri_sorted_, _edge_sorted_)
		flag0 = _inquire_min_edge_at_boundary_( _coord0_[0], _coord0_[1] )
		flag1 = _inquire_min_edge_at_boundary_( _coord1_[0], _coord1_[1] )
		
		if flag0 and flag1:
			print('I am repeating here')
			ind0 = np.argsort(_edge_len_[:,0])
			ind1 = np.argsort(_edge_len_[:,1])
			ind2 = np.argsort(_edge_len_[:,2])
			mnm0 = mnm1 = mnm2 = 1
			
			VAL = np.zeros(3)
			VAL[0] = _edge_len_[ ind0[mnm0], 0 ]
			VAL[1] = _edge_len_[ ind1[mnm1], 1 ]
			VAL[2] = _edge_len_[ ind2[mnm2], 2 ]
			
			MIN = min(VAL)
			POS = [i for i, j in enumerate(VAL) if j == MIN]
			if VAL[ POS[0] ] < ds:
				_edge_sorted_ = POS[0] 
				if _edge_sorted_ == 0: _tri_sorted_ = ind0[mnm0]
				if _edge_sorted_ == 1: _tri_sorted_ = ind1[mnm1]
				if _edge_sorted_ == 2: _tri_sorted_ = ind2[mnm2]
				print( '-> element = ', _tri_sorted_, ' min edge length : ', _edge_len_[_tri_sorted_,_edge_sorted_], '(', ds , ')'  )
				if _edge_len_[_tri_sorted_,_edge_sorted_] == 0.: print('edge len = ', VAL )
			else:
				_tri_sorted_ = -1
				_edge_sorted_ = -1
	
	return _tri_sorted_, _edge_sorted_
	
	
def _tri_coord_(_triXC, _triYC, _triZC, _tri_sorted_, _edge_sorted_):
	
	_coord0_ = np.zeros( 3 )
	_coord1_ = np.zeros( 3 )
	
	if _edge_sorted_ == 0: 
		( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_sorted_, 0], _triYC[_tri_sorted_, 0], _triZC[_tri_sorted_, 0] )
		( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_sorted_, 1], _triYC[_tri_sorted_, 1], _triZC[_tri_sorted_, 1] )
		
	if _edge_sorted_ == 1: 
		( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_sorted_, 1], _triYC[_tri_sorted_, 1], _triZC[_tri_sorted_, 1] )
		( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_sorted_, 2], _triYC[_tri_sorted_, 2], _triZC[_tri_sorted_, 2] )
		
	if _edge_sorted_ == 2: 
		( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_sorted_, 2], _triYC[_tri_sorted_, 2], _triZC[_tri_sorted_, 2] )
		( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_sorted_, 0], _triYC[_tri_sorted_, 0], _triZC[_tri_sorted_, 0] )
		
	return _coord0_, _coord1_
	
	
def _inquire_min_edge_at_boundary_(x1, y1):
	
	flag = False
	
	if x1 == 0.: 
		print('****node found to be at the eastern boundary****')
		flag = True
	
	if x1 == 1.: 
		print('****node found to be at the western boundary****')
		flag = True
	
	if y1 == 0.: 
		print('****node found to be at the northern boundary****')
		flag = True
	
	if y1 == 1.: 
		print('****node found to be at the southern boundary****')
		flag = True
	
	return flag
	
	
def _element_merging_below_critical_length_\
(_triXC, _triYC, _triZC, _delete_tri_ele_, _edge_sorted_, _nodeXC, _nodeYC, _nodeZC, _Triangle_Area_, _eleGma, ds_min, _node_lt, _node_rt, _node_dn, _node_up):
	
	while _delete_tri_ele_ >= 0:
		
		_coord0_ = np.zeros( 3 )
		_coord1_ = np.zeros( 3 )
		
		_tri_ele_ = _delete_tri_ele_
		
		# remove Lagrangian node contains in '_coord0_'
		if _edge_sorted_ == 0: 
			( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
			( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			
		if _edge_sorted_ == 1: 
			( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			
		if _edge_sorted_ == 2: 
			( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
		
		if _coord0_[0] == _coord1_[0] and _coord0_[1] == _coord1_[1] and _coord0_[2] == _coord1_[2]:
			_delete_ = []
			indx1 = np.where( _coord0_[0] == _nodeXC )[0]
			indy1 = np.where( _coord0_[1] == _nodeYC )[0]
			p0 = np.intersect1d( indx1, indy1 )
			_delete_.append(p0[0])
			_delete_ = np.unique(_delete_).tolist()
			_nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, _delete_)
			_triXC, _triYC, _triZC, _eleGma = eles_to_del4(_triXC, _triYC, _triZC, _eleGma, _tri_ele_)
		else:
			flag0 = _inquire_min_edge_at_boundary_( _coord0_[0], _coord0_[1] )
			flag1 = _inquire_min_edge_at_boundary_( _coord1_[0], _coord1_[1] )
			
			if flag0:
				tmp = np.copy( _coord0_ )
				_coord0_ = _coord1_
				_coord1_ = np.copy( tmp )
			
			_child_nodeXC = []
			_child_nodeYC = []
			_child_nodeZC = []
			
			_child_nodeXC.append( _coord1_[0] )
			_child_nodeYC.append( _coord1_[1] )
			_child_nodeZC.append( _coord1_[2] )
			
			indx1 = np.where( _coord0_[0] == _triXC )[0]
			indy1 = np.where( _coord0_[1] == _triYC )[0]
			indz1 = np.where( _coord0_[2] == _triZC )[0]
			
			indx2 = np.where( _coord1_[0] == _triXC )[0]
			indy2 = np.where( _coord1_[1] == _triYC )[0]
			indz2 = np.where( _coord1_[2] == _triZC )[0]
			
			p0 = np.intersect1d( np.intersect1d( indx1, indy1), indz1 )
			p1 = np.intersect1d( np.intersect1d( indx2, indy2), indz2 )
			
			_comm_ele_ = np.intersect1d( p0, p1 )
			
			if len(_comm_ele_) == 1:
				print( '<<<< two common triangles could not be found : not a good condition >>>>' )
				
				_delete_ = []
				for i in range( len(p0) ):
					if p0[i] == _comm_ele_[0]: _delete_.append(i)
				p0 = np.delete(p0, _delete_)
				
				_delete_ = []
				for i in range( len(p1) ):
					if p1[i] == _comm_ele_[0]: _delete_.append(i)
				p1 = np.delete(p1, _delete_)
				
				tot_vor_ = 0.
				for jtr in range( len(_comm_ele_) ): tot_vor_ += _Triangle_Area_[ _comm_ele_[jtr] ]*_eleGma[ _comm_ele_[jtr] ]
				
				_Area_ = 0.
				for jtr in range( len(p0) ):
					_tri_ele_ = p0[jtr]
					if _triXC[_tri_ele_,0] == _coord0_[0] and _triYC[_tri_ele_,0] == _coord0_[1]:
						( _triXC[_tri_ele_,0], _triYC[_tri_ele_,0], _triZC[_tri_ele_,0] ) = ( _coord1_[0], _coord1_[1], _coord1_[2] )
					
					if _triXC[_tri_ele_,1] == _coord0_[0] and _triYC[_tri_ele_,1] == _coord0_[1]:
						( _triXC[_tri_ele_,1], _triYC[_tri_ele_,1], _triZC[_tri_ele_,1] ) = ( _coord1_[0], _coord1_[1], _coord1_[2] )
					
					if _triXC[_tri_ele_,2] == _coord0_[0] and _triYC[_tri_ele_,2] == _coord0_[1]:
						( _triXC[_tri_ele_,2], _triYC[_tri_ele_,2], _triZC[_tri_ele_,2] ) = ( _coord1_[0], _coord1_[1], _coord1_[2] )
					
					p1C = np.array( [ _triXC[_tri_ele_,0], _triYC[_tri_ele_,0], _triZC[_tri_ele_,0] ] )
					p2C = np.array( [ _triXC[_tri_ele_,1], _triYC[_tri_ele_,1], _triZC[_tri_ele_,1] ] )
					p3C = np.array( [ _triXC[_tri_ele_,2], _triYC[_tri_ele_,2], _triZC[_tri_ele_,2] ] )
					
					_Area_ += _oneTriangle_Area_( p1C, p2C, p3C )
					
				for jtr in range( len(p0) ):
					_tri_ele_ = p0[jtr] 
					_eleGma[_tri_ele_] += tot_vor_/( len(p0)*_Area_ )
				
				# deleting the Lagrangian nodes
				_delete_ = []
				indx1 = np.where( _coord0_[0] == _nodeXC )[0]
				indy1 = np.where( _coord0_[1] == _nodeYC )[0]
				p0 = np.intersect1d( indx1, indy1 )
				_delete_.append(p0[0])
				
				_delete_ = np.unique(_delete_).tolist()
				_nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, _delete_)
				
				
#				_node_lt, _node_rt, _node_dn, _node_up = \
#				_rearranging_boundary_node_list_(_node_lt, _node_rt, _node_dn, _node_up, _delete_)
				
				_nodeXC = np.concatenate( (_nodeXC, _child_nodeXC) )
				_nodeYC = np.concatenate( (_nodeYC, _child_nodeYC) )
				_nodeZC = np.concatenate( (_nodeZC, _child_nodeZC) )
				
				_triXC, _triYC, _triZC, _eleGma = eles_to_del4(_triXC, _triYC, _triZC, _eleGma, _comm_ele_) 
				
			if len(_comm_ele_) == 2:
				
				_delete_ = []
				for i in range( len(p0) ):
					if p0[i] == _comm_ele_[0]: _delete_.append(i)
					if p0[i] == _comm_ele_[1]: _delete_.append(i)
				p0 = np.delete(p0, _delete_) 
				
				_delete_ = []
				for i in range( len(p1) ):
					if p1[i] == _comm_ele_[0]: _delete_.append(i)
					if p1[i] == _comm_ele_[1]: _delete_.append(i)
				p1 = np.delete(p1, _delete_)
				
				tot_vor_ = 0.
				for jtr in range( len(_comm_ele_) ): tot_vor_ += _Triangle_Area_[ _comm_ele_[jtr] ]*_eleGma[ _comm_ele_[jtr] ]
				
				_Area_ = 0.
				for jtr in range( len(p0) ):
					_tri_ele_ = p0[jtr]
					if _triXC[_tri_ele_,0] == _coord0_[0] and _triYC[_tri_ele_,0] == _coord0_[1]:
						( _triXC[_tri_ele_,0], _triYC[_tri_ele_,0], _triZC[_tri_ele_,0] ) = ( _coord1_[0], _coord1_[1], _coord1_[2] )
					
					if _triXC[_tri_ele_,1] == _coord0_[0] and _triYC[_tri_ele_,1] == _coord0_[1]:
						( _triXC[_tri_ele_,1], _triYC[_tri_ele_,1], _triZC[_tri_ele_,1] ) = ( _coord1_[0], _coord1_[1], _coord1_[2] )
					
					if _triXC[_tri_ele_,2] == _coord0_[0] and _triYC[_tri_ele_,2] == _coord0_[1]:
						( _triXC[_tri_ele_,2], _triYC[_tri_ele_,2], _triZC[_tri_ele_,2] ) = ( _coord1_[0], _coord1_[1], _coord1_[2] )
					
					p1C = np.array( [ _triXC[_tri_ele_,0], _triYC[_tri_ele_,0], _triZC[_tri_ele_,0] ] )
					p2C = np.array( [ _triXC[_tri_ele_,1], _triYC[_tri_ele_,1], _triZC[_tri_ele_,1] ] )
					p3C = np.array( [ _triXC[_tri_ele_,2], _triYC[_tri_ele_,2], _triZC[_tri_ele_,2] ] )
					
					_Area_ += _oneTriangle_Area_( p1C, p2C, p3C )
					
				for jtr in range( len(p0) ):
					_tri_ele_ = p0[jtr] 
					_eleGma[_tri_ele_] += tot_vor_/( len(p0)*_Area_ )
				
				# deleting the Lagrangian nodes
				_delete_ = []
				indx1 = np.where( _coord0_[0] == _nodeXC )[0]
				indy1 = np.where( _coord0_[1] == _nodeYC )[0]
				p0 = np.intersect1d( indx1, indy1 )
				_delete_.append(p0[0])
				_delete_ = np.unique(_delete_).tolist()
				
				_nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, _delete_)
				
#				_node_lt, _node_rt, _node_dn, _node_up = \
#				_rearranging_boundary_node_list_(_node_lt, _node_rt, _node_dn, _node_up, _delete_)
				
				_nodeXC = np.concatenate( (_nodeXC, _child_nodeXC) )
				_nodeYC = np.concatenate( (_nodeYC, _child_nodeYC) )
				_nodeZC = np.concatenate( (_nodeZC, _child_nodeZC) )
				
				_triXC, _triYC, _triZC, _eleGma = eles_to_del4(_triXC, _triYC, _triZC, _eleGma, _comm_ele_) 
			
#			_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC = _repositioning_boundary_eles_\
#			(_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, _node_lt, _node_rt, _node_dn, _node_up, 1., 1.)
			
#			_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC = _fixing_node_to_put_inside_domain \
#			(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _node_lt, _node_rt, _node_dn, _node_up, ds_min)
			
			_Triangle_Area_ = _allTriangle_Area_( _triXC, _triYC, _triZC )
			
			_delete_tri_ele_, _edge_sorted_ = _min_edge_length_dection_( _triXC, _triYC, _triZC, ds_min )
			
			if _tri_ele_ == _delete_tri_ele_: _delete_tri_ele_ = -1
	
	_triVt0 = []
	_triVt1 = []
	_triVt2 = []
	for ktr in range( len(_triXC[:,0]) ):
		p0 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,0] == _nodeXC)[0], np.where(_triYC[ktr,0] == _nodeYC)[0]), np.where(_triZC[ktr,0] == _nodeZC)[0])
		p1 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,1] == _nodeXC)[0], np.where(_triYC[ktr,1] == _nodeYC)[0]), np.where(_triZC[ktr,1] == _nodeZC)[0])
		p2 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,2] == _nodeXC)[0], np.where(_triYC[ktr,2] == _nodeYC)[0]), np.where(_triZC[ktr,2] == _nodeZC)[0])
		
		_triVt0.append(p0[0])
		_triVt1.append(p1[0])
		_triVt2.append(p2[0])
	
	_triVT0 = np.zeros( (len(_triVt0), 3) )
	_triVT0[0:len(_triVt0),0] = _triVt0[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),1] = _triVt1[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),2] = _triVt2[0:len(_triVt0)]
	
	return _triXC, _triYC, _triZC, _triVT0, _nodeXC, _nodeYC, _nodeZC, _eleGma, _node_lt, _node_rt, _node_dn, _node_up
	
	
	
def _element_splitting_above_critical_length_ \
(_triXC, _triYC, _triZC, _add_tri_ele_, _edge_sorted_, _nodeXC, _nodeYC, _nodeZC, _Triangle_Area_, _eleGma, ds_max, _node_lt, _node_rt, _node_dn, _node_up):
	
	_coord0_ = np.zeros( 3 )
	_coord1_ = np.zeros( 3 )
	_mid_pt  = np.zeros( 3 )
	while _add_tri_ele_ >= 0:
		
		_tri_ele_ = _add_tri_ele_
		
		# two Lagrangian nodes
		if _edge_sorted_ == 0: 
			( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
			( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			
		if _edge_sorted_ == 1: 
			( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			
		if _edge_sorted_ == 2: 
			( _coord0_[0], _coord0_[1], _coord0_[2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			( _coord1_[0], _coord1_[1], _coord1_[2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
		
		( _mid_pt[0], _mid_pt[1], _mid_pt[2] ) = \
		( 0.5*(_coord0_[0]+_coord1_[0]), 0.5*(_coord0_[1]+_coord1_[1]), 0.5*(_coord0_[2]+_coord1_[2]) )
		
		if _mid_pt[0] == 0. or _mid_pt[0] == 1.  or _mid_pt[1] == 0. or _mid_pt[1] == 1.:
			print( 'additional node has been added at the boundary' )
			print( _coord0_, _coord1_ )
			
			if _mid_pt[0] == 0.: _node_lt.append( len(_nodeXC) )
			if _mid_pt[0] == 1.: _node_rt.append( len(_nodeXC) )
			if _mid_pt[1] == 0.: _node_dn.append( len(_nodeXC) )
			if _mid_pt[1] == 1.: _node_up.append( len(_nodeXC) )
		
		indx1 = np.where( _coord0_[0] == _triXC )[0]
		indy1 = np.where( _coord0_[1] == _triYC )[0]
		indz1 = np.where( _coord0_[2] == _triZC )[0]
		
		indx2 = np.where( _coord1_[0] == _triXC )[0]
		indy2 = np.where( _coord1_[1] == _triYC )[0]
		indz2 = np.where( _coord1_[2] == _triZC )[0]
		
		p0 = np.intersect1d( np.intersect1d( indx1, indy1), indz1 )
		p1 = np.intersect1d( np.intersect1d( indx2, indy2), indz2 )
		_comm_ele_ = np.intersect1d( p0, p1 )
		
#		print( '_comm_ele_ = ', _comm_ele_ )
		
		if len(_comm_ele_) == 1:
			print( len(p0), len(p1) )
			print( '<<<< two common triangles could not be found : not a good condition >>>>' )
			
			_ele_ = _comm_ele_[0]
			
			_ptrixc0, _ptriyc0, _ptrizc0, _ctrixc0, _ctriyc0, _ctrizc0 = \
			_two_triele_frm_one_triele_( _triXC[_ele_,0:3], _triYC[_ele_,0:3], _triZC[_ele_,0:3], _coord0_, _coord1_ )
			
			_child_eleGma = np.zeros( (2,3) )
			_child_eleGma[0,0:3 ] = _eleGma[ _ele_,0:3 ]
			_child_eleGma[1,0:3 ] = _eleGma[ _ele_,0:3 ]
			
			_child_triXC = np.zeros( (2,3) )
			_child_triYC = np.zeros( (2,3) )
			_child_triZC = np.zeros( (2,3) )
			
			for jtr in range(3):
				( _child_triXC[0,jtr], _child_triYC[0,jtr], _child_triZC[0,jtr] ) = ( _ptrixc0[jtr], _ptriyc0[jtr], _ptrizc0[jtr] )
				( _child_triXC[1,jtr], _child_triYC[1,jtr], _child_triZC[1,jtr] ) = ( _ctrixc0[jtr], _ctriyc0[jtr], _ctrizc0[jtr] )
			
			_nodeXC = np.append( _nodeXC, _mid_pt[0] )
			_nodeYC = np.append( _nodeYC, _mid_pt[1] )
			_nodeZC = np.append( _nodeZC, _mid_pt[2] )
			
			_triXC, _triYC, _triZC, _eleGma = eles_to_del4(_triXC, _triYC, _triZC, _eleGma, _ele_)
			
			_triXC = np.concatenate( (_triXC, _child_triXC) )
			_triYC = np.concatenate( (_triYC, _child_triYC) )
			_triZC = np.concatenate( (_triZC, _child_triZC) )
			_eleGma = np.concatenate( (_eleGma, _child_eleGma) )
			
		if len(_comm_ele_) == 2:
			_ptrixc0, _ptriyc0, _ptrizc0, _ctrixc0, _ctriyc0, _ctrizc0 = \
			_two_triele_frm_one_triele_( _triXC[_comm_ele_[0],0:3], _triYC[_comm_ele_[0],0:3], _triZC[_comm_ele_[0],0:3], _coord0_, _coord1_ )
			
			_ptrixc1, _ptriyc1, _ptrizc1, _ctrixc1, _ctriyc1, _ctrizc1 = \
			_two_triele_frm_one_triele_( _triXC[_comm_ele_[1],0:3], _triYC[_comm_ele_[1],0:3], _triZC[_comm_ele_[1],0:3], _coord0_, _coord1_ )
			
			_child_eleGma = np.zeros( (4,3) )
			_child_eleGma[0, 0:3 ] = _eleGma[ _comm_ele_[0], 0:3 ]
			_child_eleGma[1, 0:3 ] = _eleGma[ _comm_ele_[0], 0:3 ]
			_child_eleGma[2, 0:3 ] = _eleGma[ _comm_ele_[1], 0:3 ]
			_child_eleGma[3, 0:3 ] = _eleGma[ _comm_ele_[1], 0:3 ]
			
			_child_triXC = np.zeros( (4,3) )
			_child_triYC = np.zeros( (4,3) )
			_child_triZC = np.zeros( (4,3) )
			
			for jtr in range(3):
				( _child_triXC[0,jtr], _child_triYC[0,jtr], _child_triZC[0,jtr] ) = ( _ptrixc0[jtr], _ptriyc0[jtr], _ptrizc0[jtr] )
				( _child_triXC[1,jtr], _child_triYC[1,jtr], _child_triZC[1,jtr] ) = ( _ctrixc0[jtr], _ctriyc0[jtr], _ctrizc0[jtr] )
				( _child_triXC[2,jtr], _child_triYC[2,jtr], _child_triZC[2,jtr] ) = ( _ptrixc1[jtr], _ptriyc1[jtr], _ptrizc1[jtr] )
				( _child_triXC[3,jtr], _child_triYC[3,jtr], _child_triZC[3,jtr] ) = ( _ctrixc1[jtr], _ctriyc1[jtr], _ctrizc1[jtr] )
			
			_nodeXC = np.append( _nodeXC, _mid_pt[0] )
			_nodeYC = np.append( _nodeYC, _mid_pt[1] )
			_nodeZC = np.append( _nodeZC, _mid_pt[2] )
			
			_triXC, _triYC, _triZC, _eleGma = eles_to_del4(_triXC, _triYC, _triZC, _eleGma, _comm_ele_)
			
			_triXC = np.concatenate( (_triXC, _child_triXC) )
			_triYC = np.concatenate( (_triYC, _child_triYC) )
			_triZC = np.concatenate( (_triZC, _child_triZC) )
			_eleGma = np.concatenate( (_eleGma, _child_eleGma) )
		
		if len(_comm_ele_) > 2 :
			_ele_ = _comm_ele_[2:]
			_triXC, _triYC, _triZC, _eleGma = eles_to_del4(_triXC, _triYC, _triZC, _eleGma, _ele_)
			
		
#		_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC = _repositioning_boundary_eles_\
#		(_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, _node_lt, _node_rt, _node_dn, _node_up, 1., 1.)
		
#		_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC = _fixing_node_to_put_inside_domain \
#		(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _node_lt, _node_rt, _node_dn, _node_up, 0.002 )
		
		_add_tri_ele_, _edge_sorted_ = _max_edge_length_dection_( _triXC, _triYC, _triZC, ds_max )
	
	_triVt0 = []
	_triVt1 = []
	_triVt2 = []
	for ktr in range( len(_triXC[:,0]) ):
		p0 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,0] == _nodeXC)[0], np.where(_triYC[ktr,0] == _nodeYC)[0]), np.where(_triZC[ktr,0] == _nodeZC)[0])
		p1 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,1] == _nodeXC)[0], np.where(_triYC[ktr,1] == _nodeYC)[0]), np.where(_triZC[ktr,1] == _nodeZC)[0])
		p2 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,2] == _nodeXC)[0], np.where(_triYC[ktr,2] == _nodeYC)[0]), np.where(_triZC[ktr,2] == _nodeZC)[0])
		
		_triVt0.append(p0[0])
		_triVt1.append(p1[0])
		_triVt2.append(p2[0])
	
	_triVT0 = np.zeros( (len(_triVt0), 3) )
	_triVT0[0:len(_triVt0),0] = _triVt0[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),1] = _triVt1[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),2] = _triVt2[0:len(_triVt0)]
	
	return _triXC, _triYC, _triZC, _triVT0, _nodeXC, _nodeYC, _nodeZC, _eleGma, _node_lt, _node_rt, _node_dn, _node_up
	
	
	
def _finding_vertex_within_tri_ele_( _trixc, _triyc, _coord0_, _coord1_):
	
	indx = np.where( _coord0_[0] == _trixc )[0]
	indy = np.where( _coord0_[1] == _triyc )[0]
	v0 = np.intersect1d( indx, indy )[0]
	
	indx = np.where( _coord1_[0] == _trixc )[0]
	indy = np.where( _coord1_[1] == _triyc )[0]
	v1 = np.intersect1d( indx, indy )[0]
	
	return v0, v1
	
	
def _two_triele_frm_one_triele_( _trixc, _triyc, _trizc, _coord0_, _coord1_ ):
	
	v0, v1 = _finding_vertex_within_tri_ele_( _trixc, _triyc, _coord0_, _coord1_)
	
	_ptrixc = np.zeros( 3 )
	_ptriyc = np.zeros( 3 )
	_ptrizc = np.zeros( 3 )
	
	_ctrixc = np.zeros( 3 )
	_ctriyc = np.zeros( 3 )
	_ctrizc = np.zeros( 3 )
	
	( mcx, mcy, mcz ) = ( 0.5*(_coord0_[0]+_coord1_[0]), 0.5*(_coord0_[1]+_coord1_[1]), 0.5*(_coord0_[2]+_coord1_[2]) )
	
	if v0 + v1 == 1: #( v0=0 & v1=1, or, vice-versa )
		( _ptrixc[0], _ptriyc[0], _ptrizc[0] ) = ( _trixc[0], _triyc[0], _trizc[0] )
		( _ptrixc[1], _ptriyc[1], _ptrizc[1] ) = ( mcx, mcy, mcz )
		( _ptrixc[2], _ptriyc[2], _ptrizc[2] ) = ( _trixc[2], _triyc[2], _trizc[2] )

		( _ctrixc[0], _ctriyc[0], _ctrizc[0] ) = ( mcx, mcy, mcz )
		( _ctrixc[1], _ctriyc[1], _ctrizc[1] ) = ( _trixc[1], _triyc[1], _trizc[1] )
		( _ctrixc[2], _ctriyc[2], _ctrizc[2] ) = ( _trixc[2], _triyc[2], _trizc[2] )
		
	if v0 + v1 == 2: #( v0=0 & v1=2, or, vice-versa )
		( _ptrixc[0], _ptriyc[0], _ptrizc[0] ) = ( _trixc[0], _triyc[0], _trizc[0] )
		( _ptrixc[1], _ptriyc[1], _ptrizc[1] ) = ( _trixc[1], _triyc[1], _trizc[1] )
		( _ptrixc[2], _ptriyc[2], _ptrizc[2] ) = ( mcx, mcy, mcz )
		
		( _ctrixc[0], _ctriyc[0], _ctrizc[0] ) = ( mcx, mcy, mcz )
		( _ctrixc[1], _ctriyc[1], _ctrizc[1] ) = ( _trixc[1], _triyc[1], _trizc[1] )
		( _ctrixc[2], _ctriyc[2], _ctrizc[2] ) = ( _trixc[2], _triyc[2], _trizc[2] )
	
	if v0 + v1 == 3: #( v0=1 & v1=2, or, vice-versa )
		( _ptrixc[0], _ptriyc[0], _ptrizc[0] ) = ( _trixc[0], _triyc[0], _trizc[0] )
		( _ptrixc[1], _ptriyc[1], _ptrizc[1] ) = ( _trixc[1], _triyc[1], _trizc[1] )
		( _ptrixc[2], _ptriyc[2], _ptrizc[2] ) = ( mcx, mcy, mcz )
		
		( _ctrixc[0], _ctriyc[0], _ctrizc[0] ) = ( _trixc[0], _triyc[0], _trizc[0] )
		( _ctrixc[1], _ctriyc[1], _ctrizc[1] ) = ( mcx, mcy, mcz )
		( _ctrixc[2], _ctriyc[2], _ctrizc[2] ) = ( _trixc[2], _triyc[2], _trizc[2] )
	
	return _ptrixc, _ptriyc, _ptrizc, _ctrixc, _ctriyc, _ctrizc
	
	
def _oneTriangle_Area_(p1_C, p2_C, p3_C):
	
	a = np.sqrt( (p1_C[0]-p2_C[0])**2. + (p1_C[1]-p2_C[1])**2. + (p1_C[2]-p2_C[2])**2. )
	b = np.sqrt( (p2_C[0]-p3_C[0])**2. + (p2_C[1]-p3_C[1])**2. + (p2_C[2]-p3_C[2])**2. )
	c = np.sqrt( (p3_C[0]-p1_C[0])**2. + (p3_C[1]-p1_C[1])**2. + (p3_C[2]-p1_C[2])**2. )
	
	s = (a+b+c)/2.
	
	_Area_ = np.sqrt( s*(s-a)*(s-b)*(s-c) )
	
	return _Area_
	
	
def _repositioning_boundary_eles_\
(_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up, Lx, Ly):
	
#	for itr in range( len(_bndry_node_lt) ):
#		_node_ = _bndry_node_lt[itr]
#		ind0 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,0] )[0], np.where( _nodeYC[_node_] == _triYC[:,0] )[0] )
#		ind1 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,1] )[0], np.where( _nodeYC[_node_] == _triYC[:,1] )[0] )
#		ind2 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,2] )[0], np.where( _nodeYC[_node_] == _triYC[:,2] )[0] )
#		
#		if len(ind0) > 0:
#			for jtr in range( len(ind0) ): _triXC[ ind0[jtr], 0 ] = 0.
#		if len(ind1) > 0:
#			for jtr in range( len(ind1) ): _triXC[ ind1[jtr], 1 ] = 0.
#		if len(ind2) > 0:
#			for jtr in range( len(ind2) ): _triXC[ ind2[jtr], 2 ] = 0.
#		
#		_nodeXC[ _node_ ] = 0.
#	
#	for itr in range( len(_bndry_node_rt) ):
#		_node_ = _bndry_node_rt[itr]
#		ind0 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,0] )[0], np.where( _nodeYC[_node_] == _triYC[:,0] )[0] )
#		ind1 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,1] )[0], np.where( _nodeYC[_node_] == _triYC[:,1] )[0] )
#		ind2 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,2] )[0], np.where( _nodeYC[_node_] == _triYC[:,2] )[0] )
#		
#		if len(ind0) > 0:
#			for jtr in range( len(ind0) ): _triXC[ ind0[jtr], 0 ] = Lx
#		if len(ind1) > 0:
#			for jtr in range( len(ind1) ): _triXC[ ind1[jtr], 1 ] = Lx
#		if len(ind2) > 0:
#			for jtr in range( len(ind2) ): _triXC[ ind2[jtr], 2 ] = Lx
#		
#		_nodeXC[ _node_ ] = Lx
	
	
	for itr in range( len(_bndry_node_dn) ):
		_node_ = _bndry_node_dn[itr]
		ind0 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,0] )[0], np.where( _nodeYC[_node_] == _triYC[:,0] )[0] )
		ind1 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,1] )[0], np.where( _nodeYC[_node_] == _triYC[:,1] )[0] )
		ind2 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,2] )[0], np.where( _nodeYC[_node_] == _triYC[:,2] )[0] )
		
		if len(ind0) > 0:
			for jtr in range( len(ind0) ): _triYC[ ind0[jtr], 0 ] = 0.
		if len(ind1) > 0:
			for jtr in range( len(ind1) ): _triYC[ ind1[jtr], 1 ] = 0.
		if len(ind2) > 0:
			for jtr in range( len(ind2) ): _triYC[ ind2[jtr], 2 ] = 0.
		
		_nodeYC[ _node_ ] = 0.
		
	for itr in range( len(_bndry_node_up) ):
		_node_ = _bndry_node_up[itr]
		ind0 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,0] )[0], np.where( _nodeYC[_node_] == _triYC[:,0] )[0] )
		ind1 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,1] )[0], np.where( _nodeYC[_node_] == _triYC[:,1] )[0] )
		ind2 = np.intersect1d( np.where( _nodeXC[_node_] == _triXC[:,2] )[0], np.where( _nodeYC[_node_] == _triYC[:,2] )[0] )
		
		if len(ind0) > 0:
			for jtr in range( len(ind0) ): _triYC[ ind0[jtr], 0 ] = Ly
		if len(ind1) > 0:
			for jtr in range( len(ind1) ): _triYC[ ind1[jtr], 1 ] = Ly
		if len(ind2) > 0:
			for jtr in range( len(ind2) ): _triYC[ ind2[jtr], 2 ] = Ly
		
		_nodeYC[ _node_ ] = Ly
	
	return _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC
	
	
def _node_outside_domain(_nodeXC, _nodeYC, node_lt, node_rt, node_dn, node_up):
	
#	nodex_l = _nodeXC[ node_lt[0:len(node_lt)] ]
#	nodex_r = _nodeXC[ node_rt[0:len(node_rt)] ]
#	
#	local_min_pos = min(nodex_l)
#	local_max_pos = max(nodex_r)
#	
#	global_min_pos = min(_nodeXC)
#	global_max_pos = max(_nodeXC)
	
	indw = inde = inds = indn = -1
#	
#	if global_min_pos < local_min_pos: indw = np.where( global_min_pos == _nodeXC )[0][0]
#	if global_max_pos > local_max_pos: inde = np.where( global_max_pos == _nodeXC )[0][0]
	
	nodey_l = _nodeYC[ node_dn[0:len(node_dn)] ]
	nodey_r = _nodeYC[ node_up[0:len(node_up)] ]
	
	local_min_pos = min(nodey_l)
	local_max_pos = max(nodey_r)
	
	global_min_pos = min(_nodeYC)
	global_max_pos = max(_nodeYC)
	
	if global_min_pos < local_min_pos: inds = np.where( global_min_pos == _nodeYC )[0][0]
	if global_max_pos > local_max_pos: indn = np.where( global_max_pos == _nodeYC )[0][0]
	
#	return indw, inde, inds, indn
	
	return inds, indn
	
	
def _fixing_node_to_put_inside_domain \
(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, node_lt, node_rt, node_dn, node_up, ds):
	
#	indw, inde, inds, indn = \
#	_node_outside_domain(_nodeXC, _nodeYC, node_lt, node_rt, node_dn, node_up)
	
	inds, indn = \
	_node_outside_domain(_nodeXC, _nodeYC, node_lt, node_rt, node_dn, node_up)
		
#	while indw >= 0 or inde >= 0 or inds >= 0 or indn >= 0: 
	
	while inds >= 0 or indn >= 0: 	 
		
#		if indw >= 0:
#			ind0 = np.intersect1d( np.where( _nodeXC[indw] == _triXC[:,0] )[0], np.where( _nodeYC[indw] == _triYC[:,0] )[0]  ) 
#			ind1 = np.intersect1d( np.where( _nodeXC[indw] == _triXC[:,1] )[0], np.where( _nodeYC[indw] == _triYC[:,1] )[0]  )
#			ind2 = np.intersect1d( np.where( _nodeXC[indw] == _triXC[:,2] )[0], np.where( _nodeYC[indw] == _triYC[:,2] )[0]  )
#			
#			_nodeXC[indw] = 0. + 2.*ds
#			if len(ind0) > 0: 
#				for itr in range( len(ind0) ): _triXC[ ind0[itr],0 ] = _nodeXC[indw]
#			if len(ind1) > 0: 
#				for itr in range( len(ind1) ): _triXC[ ind1[itr],1 ] = _nodeXC[indw]
#			if len(ind2) > 0: 
#				for itr in range( len(ind2) ): _triXC[ ind2[itr],2 ] = _nodeXC[indw]
#		
#		if inde >= 0: 
#			ind0 = np.intersect1d( np.where( _nodeXC[inde] == _triXC[:,0] )[0], np.where( _nodeYC[inde] == _triYC[:,0] )[0]  ) 
#			ind1 = np.intersect1d( np.where( _nodeXC[inde] == _triXC[:,1] )[0], np.where( _nodeYC[inde] == _triYC[:,1] )[0]  )
#			ind2 = np.intersect1d( np.where( _nodeXC[inde] == _triXC[:,2] )[0], np.where( _nodeYC[inde] == _triYC[:,2] )[0]  )
#			
#			_nodeXC[inde] = 1. - 2.*ds
#			if len(ind0) > 0: 
#				for itr in range( len(ind0) ): _triXC[ ind0[itr],0 ] = _nodeXC[inde]
#			if len(ind1) > 0: 
#				for itr in range( len(ind1) ): _triXC[ ind1[itr],1 ] = _nodeXC[inde]
#			if len(ind2) > 0: 
#				for itr in range( len(ind2) ): _triXC[ ind2[itr],2 ] = _nodeXC[inde]
#		
		if inds >= 0: 
			ind0 = np.intersect1d( np.where( _nodeXC[inds] == _triXC[:,0] )[0], np.where( _nodeYC[inds] == _triYC[:,0] )[0]  ) 
			ind1 = np.intersect1d( np.where( _nodeXC[inds] == _triXC[:,1] )[0], np.where( _nodeYC[inds] == _triYC[:,1] )[0]  )
			ind2 = np.intersect1d( np.where( _nodeXC[inds] == _triXC[:,2] )[0], np.where( _nodeYC[inds] == _triYC[:,2] )[0]  )
			
			_nodeYC[inds] = 0. + 2.*ds
			if len(ind0) > 0: 
				for itr in range( len(ind0) ): _triYC[ ind0[itr],0 ] = _nodeYC[inds]
			if len(ind1) > 0: 
				for itr in range( len(ind1) ): _triYC[ ind1[itr],1 ] = _nodeYC[inds]
			if len(ind2) > 0: 
				for itr in range( len(ind2) ): _triYC[ ind2[itr],2 ] = _nodeYC[inds]
		
		if indn >= 0: 
			ind0 = np.intersect1d( np.where( _nodeXC[indn] == _triXC[:,0] )[0], np.where( _nodeYC[indn] == _triYC[:,0] )[0]  ) 
			ind1 = np.intersect1d( np.where( _nodeXC[indn] == _triXC[:,1] )[0], np.where( _nodeYC[indn] == _triYC[:,1] )[0]  )
			ind2 = np.intersect1d( np.where( _nodeXC[indn] == _triXC[:,2] )[0], np.where( _nodeYC[indn] == _triYC[:,2] )[0]  )
			
			_nodeYC[indn] = 1. - 2.*ds
			if len(ind0) > 0: 
				for itr in range( len(ind0) ): _triYC[ ind0[itr],0 ] = _nodeYC[indn]
			if len(ind1) > 0: 
				for itr in range( len(ind1) ): _triYC[ ind1[itr],1 ] = _nodeYC[indn]
			if len(ind2) > 0: 
				for itr in range( len(ind2) ): _triYC[ ind2[itr],2 ] = _nodeYC[indn]
		
		#indw, inde, 
		inds, indn = \
		_node_outside_domain(_nodeXC, _nodeYC, node_lt, node_rt, node_dn, node_up)
	
	
	return _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC
	
	
#	return _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, tri_verti_, _eleGma, node_lt, node_rt, node_dn, node_up
	
	
def _del_straddle_ele(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGma, tri_verti_, node_lt, node_rt, node_dn, node_up):
	
	node_del = []
	eles_del = []
	for itr in range( len(_nodeXC) ):
		indx0 = np.where( _nodeXC[itr] == _triXC )[0]
		indy0 = np.where( _nodeYC[itr] == _triYC )[0]
		p0 = np.intersect1d( indx0, indy0 )
		
		if len(p0) == 1:
			print('one lonely element found')
			node_del.append(itr)
			eles_del.append(p0[0])
		
	if len(node_del) > 0:
		node_lt, node_rt, node_dn, node_up = \
		_rearranging_boundary_node_list_(node_lt, node_rt, node_dn, node_up, node_del)
	
		_nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, node_del)
	
	if len(eles_del) > 0:
		_triXC, _triYC, _triZC, _eleGma = eles_to_del(_triXC, _triYC, _triZC, tri_verti_, _eleGma, eles_del)
	
	return _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGma, tri_verti_, node_lt, node_rt, node_dn, node_up
	
	
def _del_ele_contains_straddle_node_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, tri_verti_, coord):
	
	indx = np.where( coord[0] == _triXC )[0]
	indy = np.where( coord[1] == _triYC )[0]
	
	p = np.intersect1d( indx, indy )[0]
	
	return p
	
	
def node_to_del(_nodeXC, _nodeYC, _nodeZC, node_del):
	
	_nodeXC = np.delete( _nodeXC, node_del )
	_nodeYC = np.delete( _nodeYC, node_del )
	_nodeZC = np.delete( _nodeZC, node_del )
	
	return _nodeXC, _nodeYC, _nodeZC
	
	
def eles_to_del(_triXC, _triYC, _triZC, tri_verti_, eleGma, eles_del):
	
	_triXC = np.delete( _triXC, eles_del, axis=0 )
	_triYC = np.delete( _triYC, eles_del, axis=0 )
	_triZC = np.delete( _triZC, eles_del, axis=0 )
	eleGma = np.delete( eleGma, eles_del, axis=0 )
	tri_verti_ = np.delete( tri_verti_, eles_del, axis=0 )
	
	return _triXC, _triYC, _triZC, tri_verti_, eleGma
	
	
def eles_to_del4(_triXC, _triYC, _triZC, eleGma, eles_del):
	
	_triXC = np.delete( _triXC, eles_del, axis=0 )
	_triYC = np.delete( _triYC, eles_del, axis=0 )
	_triZC = np.delete( _triZC, eles_del, axis=0 )
	eleGma = np.delete( eleGma, eles_del, axis=0 )
	
	return _triXC, _triYC, _triZC, eleGma
	
	
def eles_to_del5(_triXC, _triYC, _triZC, tri_verti_, eleGma, eles_del):
	
	_triXC = np.delete( _triXC, eles_del, axis=0 )
	_triYC = np.delete( _triYC, eles_del, axis=0 )
	_triZC = np.delete( _triZC, eles_del, axis=0 )
	eleGma = np.delete( eleGma, eles_del, axis=0 )
	
	tri_verti_ = np.delete( tri_verti_, eles_del, axis=0 )
	
	return _triXC, _triYC, _triZC, tri_verti_, eleGma
	
	
def _rearranging_boundary_node_list_(node_lt, node_rt, node_dn, node_up, delete_node):
	
	for jtr in range( len(delete_node) ):
	
		for itr in range( len(node_lt) ):
			if node_lt[itr] > delete_node[jtr]: node_lt[itr] -= 1
			
		for itr in range( len(node_rt) ):
			if node_rt[itr] > delete_node[jtr]: node_rt[itr] -= 1
		
		for itr in range( len(node_dn) ):
			if node_dn[itr] > delete_node[jtr]: node_dn[itr] -= 1
		
		for itr in range( len(node_up) ):
			if node_up[itr] > delete_node[jtr]: node_up[itr] -= 1
	
	return node_lt, node_rt, node_dn, node_up
	
	
def _construct_elements_in_X_(_triXC, _triYC, _triZC, _eleGma, _nodeXC, _nodeYC, _nodeZC, Lx, ds_max):
	
	tmpxA = []
	tmpyA = []
	tmpzA = []
	Iter = []
	# check whether nodes are crossing western border after a critical distance
	for itr in range( len(_nodeXC) ):
		if _nodeXC[itr] < -ds_max: 
			tmpxA.append( _nodeXC[itr] )
			tmpyA.append( _nodeYC[itr] )
			tmpzA.append( _nodeZC[itr] )
			Iter.append(itr)
	
	if len(tmpxA) > 0:
		ind = np.argsort(tmpxA)
		mnm = min(ind)
		itr0 = ind[mnm]
		
		p0 = np.intersect1d( np.where(_triXC == tmpxA[itr0])[0], np.where(_triYC == tmpyA[itr0])[0] )
		p0 = np.unique(p0).tolist()
	else:
		p0 = []
	
	while len(p0) > 0:
		print('----> tri eles at the western boundary ', p0) 
		itrx = Iter[itr0]
		
		for ktr in range( len(p0) ):
			ele_to_move = p0[ktr]
			_triXC, _triYC, _triZC = \
			_moving_tri_ele_in_X_(_nodeXC[itrx], _nodeYC[itrx], _nodeZC[itrx], ele_to_move, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, Lx)
		
		_nodeXC, _nodeYC, _nodeZC = _moving_nodes_( p0, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC )
		_nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, itrx)
		
		tmpxA = []
		tmpyA = []
		tmpzA = []
		Iter = []
		for itr in range( len(_nodeXC) ):
			if _nodeXC[itr] < -ds_max: 
				tmpxA.append( _nodeXC[itr] )
				tmpyA.append( _nodeYC[itr] )
				tmpzA.append( _nodeZC[itr] )
				Iter.append(itr)
		
		if len(tmpxA) > 0:
			ind = np.argsort(tmpxA)
			mnm = min(ind)
			itr0 = ind[mnm]
			
			p0 = np.intersect1d( np.where(_triXC == tmpxA[itr0])[0], np.where(_triYC == tmpyA[itr0])[0] )
			p0 = np.unique(p0).tolist()
		else:
			p0 = []
################################################
	tmpxB = []
	tmpyB = []
	tmpzB = []
	Iter = []
	# check whether nodes are crossing eastern border after a critical distance
	for itr in range( len(_nodeXC) ):
		if _nodeXC[itr] > Lx+ds_max: 
			tmpxB.append( _nodeXC[itr] )
			tmpyB.append( _nodeYC[itr] )
			tmpzB.append( _nodeZC[itr] )
			Iter.append(itr)
		
	if len(tmpxB) > 0:
		ind = np.argsort(tmpxB)
		mxm = max(ind)
		itr0 = ind[mxm]
		
		p1 = np.intersect1d( np.where(_triXC == tmpxB[itr0])[0], np.where(_triYC == tmpyB[itr0])[0] )
		p1 = np.unique(p1).tolist()
	else:
		p1 = [] 
	
	while len(p1) > 0:
		print('----> tri eles at the eastern boundary ', p1 ) 
		itrx = Iter[itr0]
		
		for ktr in range( len(p1) ):
			ele_to_move = p1[ktr]
			_triXC, _triYC, _triZC = \
			_moving_tri_ele_in_X_(_nodeXC[itrx], _nodeYC[itrx], _nodeZC[itrx], ele_to_move, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, -Lx)
			
#			print(' coord of tri element -> ')
#			print(_triXC[ele_to_move,:])
#			print(_triYC[ele_to_move,:])
#			print(_triZC[ele_to_move,:])
			
		_nodeXC, _nodeYC, _nodeZC = _moving_nodes_( p1, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC )
		_nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, itrx)
		
		tmpxB = []
		tmpyB = []
		tmpzB = []
		Iter = []
		for itr in range( len(_nodeXC) ):
			if _nodeXC[itr] > Lx+ds_max: 
				tmpxB.append( _nodeXC[itr] )
				tmpyB.append( _nodeYC[itr] )
				tmpzB.append( _nodeZC[itr] )
				Iter.append(itr)
		
		if len(tmpxB) > 0:
			ind = np.argsort(tmpxB)
			mxm = max(ind)
			itr0 = ind[mxm]
			
			p1 = np.intersect1d( np.where(_triXC == tmpxB[itr0])[0], np.where(_triYC == tmpyB[itr0])[0] )
			p1 = np.unique(p1).tolist()
		else:
			p1 = []
	
	_triVt0 = []
	_triVt1 = []
	_triVt2 = []
	for ktr in range( len(_triXC[:,0]) ):
		p0 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,0] == _nodeXC)[0], np.where(_triYC[ktr,0] == _nodeYC)[0]), np.where(_triZC[ktr,0] == _nodeZC)[0])
		p1 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,1] == _nodeXC)[0], np.where(_triYC[ktr,1] == _nodeYC)[0]), np.where(_triZC[ktr,1] == _nodeZC)[0])
		p2 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,2] == _nodeXC)[0], np.where(_triYC[ktr,2] == _nodeYC)[0]), np.where(_triZC[ktr,2] == _nodeZC)[0])
		
		_triVt0.append(p0[0])
		_triVt1.append(p1[0])
		_triVt2.append(p2[0])
	
	_triVT0 = np.zeros( (len(_triVt0), 3) )
	_triVT0[0:len(_triVt0),0] = _triVt0[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),1] = _triVt1[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),2] = _triVt2[0:len(_triVt0)]
	
	return _triXC, _triYC, _triZC, _triVT0, _eleGma, _nodeXC, _nodeYC, _nodeZC
	
	
	
def _construct_elements_in_Y_(_triXC, _triYC, _triZC, _eleGma, _nodeXC, _nodeYC, _nodeZC, Ly, ds_max):
	
	tmpxA = []
	tmpyA = []
	tmpzA = []
	Iter = []
	# check whether nodes are crossing western border after a critical distance
	for itr in range( len(_nodeYC) ):
		if _nodeYC[itr] < -ds_max: 
			tmpxA.append( _nodeXC[itr] )
			tmpyA.append( _nodeYC[itr] )
			tmpzA.append( _nodeZC[itr] )
			Iter.append(itr)
	
	if len(tmpyA) > 0:
		ind = np.argsort(tmpyA)
		mnm = min(ind)
		itr0 = ind[mnm]
		
		p0 = np.intersect1d( np.where(_triXC == tmpxA[itr0])[0], np.where(_triYC == tmpyA[itr0])[0] )
		p0 = np.unique(p0).tolist()
	else:
		p0 = []
	
	while len(p0) > 0:
		print('----> tri eles at the southern boundary ', p0) 
		itrx = Iter[itr0]
		
		for ktr in range( len(p0) ):
			ele_to_move = p0[ktr]
			_triXC, _triYC, _triZC = \
			_moving_tri_ele_in_Y_(_nodeXC[itrx], _nodeYC[itrx], _nodeZC[itrx], ele_to_move, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, Ly)
		
		_nodeXC, _nodeYC, _nodeZC = _moving_nodes_( p0, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC )
		_nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, itrx)
		
		tmpxA = []
		tmpyA = []
		tmpzA = []
		Iter = []
		for itr in range( len(_nodeYC) ):
			if _nodeYC[itr] < -ds_max: 
				tmpxA.append( _nodeXC[itr] )
				tmpyA.append( _nodeYC[itr] )
				tmpzA.append( _nodeZC[itr] )
				Iter.append(itr)
		
		if len(tmpyA) > 0:
			ind = np.argsort(tmpyA)
			mnm = min(ind)
			itr0 = ind[mnm]
			
			p0 = np.intersect1d( np.where(_triXC == tmpxA[itr0])[0], np.where(_triYC == tmpyA[itr0])[0] )
			p0 = np.unique(p0).tolist()
		else:
			p0 = []
################################################
	tmpxB = []
	tmpyB = []
	tmpzB = []
	Iter = []
	# check whether nodes are crossing eastern border after a critical distance
	for itr in range( len(_nodeYC) ):
		if _nodeYC[itr] > Ly+ds_max: 
			tmpxB.append( _nodeXC[itr] )
			tmpyB.append( _nodeYC[itr] )
			tmpzB.append( _nodeZC[itr] )
			Iter.append(itr)
		
	if len(tmpyB) > 0:
		ind = np.argsort(tmpyB)
		mxm = max(ind)
		itr0 = ind[mxm]
		
		p1 = np.intersect1d( np.where(_triXC == tmpxB[itr0])[0], np.where(_triYC == tmpyB[itr0])[0] )
		p1 = np.unique(p1).tolist()
	else:
		p1 = [] 
	
	while len(p1) > 0:
		print('----> tri eles at the northern boundary ', p1 ) 
		itrx = Iter[itr0]
		
		for ktr in range( len(p1) ):
			ele_to_move = p1[ktr]
			_triXC, _triYC, _triZC = \
			_moving_tri_ele_in_Y_(_nodeXC[itrx], _nodeYC[itrx], _nodeZC[itrx], ele_to_move, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, -Ly)
			
		_nodeXC, _nodeYC, _nodeZC = _moving_nodes_( p1, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC )
		_nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, itrx)
		
		tmpxB = []
		tmpyB = []
		tmpzB = []
		Iter = []
		for itr in range( len(_nodeYC) ):
			if _nodeYC[itr] > Ly+ds_max: 
				tmpxB.append( _nodeXC[itr] )
				tmpyB.append( _nodeYC[itr] )
				tmpzB.append( _nodeZC[itr] )
				Iter.append(itr)
		
		if len(tmpyB) > 0:
			ind = np.argsort(tmpyB)
			mxm = max(ind)
			itr0 = ind[mxm]
			
			p1 = np.intersect1d( np.where(_triXC == tmpxB[itr0])[0], np.where(_triYC == tmpyB[itr0])[0] )
			p1 = np.unique(p1).tolist()
		else:
			p1 = []
	
	_triVt0 = []
	_triVt1 = []
	_triVt2 = []
	for ktr in range( len(_triXC[:,0]) ):
		p0 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,0] == _nodeXC)[0], np.where(_triYC[ktr,0] == _nodeYC)[0]), np.where(_triZC[ktr,0] == _nodeZC)[0])
		p1 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,1] == _nodeXC)[0], np.where(_triYC[ktr,1] == _nodeYC)[0]), np.where(_triZC[ktr,1] == _nodeZC)[0])
		p2 =  \
		np.intersect1d( np.intersect1d(np.where(_triXC[ktr,2] == _nodeXC)[0], np.where(_triYC[ktr,2] == _nodeYC)[0]), np.where(_triZC[ktr,2] == _nodeZC)[0])
		
		_triVt0.append(p0[0])
		_triVt1.append(p1[0])
		_triVt2.append(p2[0])
	
	_triVT0 = np.zeros( (len(_triVt0), 3) )
	_triVT0[0:len(_triVt0),0] = _triVt0[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),1] = _triVt1[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),2] = _triVt2[0:len(_triVt0)]
	
	return _triXC, _triYC, _triZC, _triVT0, _eleGma, _nodeXC, _nodeYC, _nodeZC
	
	
def _find_nearest_two_nodes_( _nodeXC, _nodeYC, _nodeZC, px, py, pz ):
	
	_len_ = len(_nodeXC)
	_dist_ = np.zeros(_len_)
	
	for itr in range(_len_):
		_dist_[itr] = np.sqrt( (px - _nodeXC[itr])**2. + (py - _nodeYC[itr])**2. + (pz - _nodeZC[itr])**2. )
	
	idx = np.argsort(_dist_)
#	print( 'distance = ', _dist_[idx[0:3]] )
	_nearest_nodes_ = idx[1:3]
	
	return _nearest_nodes_
	
	
def _find_nearest_node_( _nodeXC, _nodeYC, _nodeZC, px, py, pz ):
	
	_len_ = len(_nodeXC)
	_dist_ = np.zeros(_len_)
	
	for itr in range(_len_):
		_dist_[itr] = np.sqrt( (px - _nodeXC[itr])**2. + (py - _nodeYC[itr])**2. + (pz - _nodeZC[itr])**2. )
	
	idx = np.argsort(_dist_)
#	print( 'distance = ', _dist_[idx[0:3]] )
	_nearest_node_ = idx[0]
	
	return _nearest_node_
	
	
def _moving_tri_ele_in_X_(px0, py0, pz0, ele_to_move, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, dx):
	
	ind = np.intersect1d( np.where(_triXC[ele_to_move,0] == px0)[0], np.where(_triYC[ele_to_move,0] == py0)[0] )
	if len(ind) > 0: _edge_ = 0
	
	ind = np.intersect1d( np.where(_triXC[ele_to_move,1] == px0)[0], np.where(_triYC[ele_to_move,1] == py0)[0] )
	if len(ind) > 0: _edge_ = 1
	
	ind = np.intersect1d( np.where(_triXC[ele_to_move,2] == px0)[0], np.where(_triYC[ele_to_move,2] == py0)[0] )
	if len(ind) > 0: _edge_ = 2
	
	if _edge_ == 0:
		_triXC[ele_to_move, 0] += dx
		
		_nearest_node_ = _find_nearest_node_( _nodeXC, _nodeYC, _nodeZC, _triXC[ele_to_move,0], _triYC[ele_to_move,0], _triZC[ele_to_move,0] )
		(_triXC[ele_to_move,0], _triYC[ele_to_move,0], _triZC[ele_to_move,0]) = (_nodeXC[_nearest_node_], _nodeYC[_nearest_node_], _nodeZC[_nearest_node_])
		
		_triXC[ele_to_move,1] += dx
		_triXC[ele_to_move,2] += dx
		
	if _edge_ == 1:
		_triXC[ele_to_move, 1] += dx
		
		_nearest_node_ = _find_nearest_node_( _nodeXC, _nodeYC, _nodeZC, _triXC[ele_to_move,1], _triYC[ele_to_move,1], _triZC[ele_to_move,1] )
		(_triXC[ele_to_move,1], _triYC[ele_to_move,1], _triZC[ele_to_move,1]) = (_nodeXC[_nearest_node_], _nodeYC[_nearest_node_], _nodeZC[_nearest_node_])
		
		_triXC[ele_to_move,0] += dx
		_triXC[ele_to_move,2] += dx
		
	if _edge_ == 2:
		_triXC[ele_to_move, 2] += dx
		
		_nearest_node_ = _find_nearest_node_( _nodeXC, _nodeYC, _nodeZC, _triXC[ele_to_move,2], _triYC[ele_to_move,2], _triZC[ele_to_move,2] )
		(_triXC[ele_to_move,2], _triYC[ele_to_move,2], _triZC[ele_to_move,2]) = (_nodeXC[_nearest_node_], _nodeYC[_nearest_node_], _nodeZC[_nearest_node_])
		
		_triXC[ele_to_move,0] += dx
		_triXC[ele_to_move,1] += dx
	
	return _triXC, _triYC, _triZC
	
	
def _moving_tri_ele_in_Y_(px0, py0, pz0, ele_to_move, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC, dy):
	
	ind = np.intersect1d( np.where(_triXC[ele_to_move,0] == px0)[0], np.where(_triYC[ele_to_move,0] == py0)[0] )
	if len(ind) > 0: _edge_ = 0
	
	ind = np.intersect1d( np.where(_triXC[ele_to_move,1] == px0)[0], np.where(_triYC[ele_to_move,1] == py0)[0] )
	if len(ind) > 0: _edge_ = 1
	
	ind = np.intersect1d( np.where(_triXC[ele_to_move,2] == px0)[0], np.where(_triYC[ele_to_move,2] == py0)[0] )
	if len(ind) > 0: _edge_ = 2
	
	if _edge_ == 0:
		_triYC[ele_to_move, 0] += dy
		
		_nearest_node_ = _find_nearest_node_( _nodeXC, _nodeYC, _nodeZC, _triXC[ele_to_move,0], _triYC[ele_to_move,0], _triZC[ele_to_move,0] )
		(_triXC[ele_to_move,0], _triYC[ele_to_move,0], _triZC[ele_to_move,0]) = (_nodeXC[_nearest_node_], _nodeYC[_nearest_node_], _nodeZC[_nearest_node_])
		
		_triYC[ele_to_move,1] += dy
		_triYC[ele_to_move,2] += dy
		
	if _edge_ == 1:
		_triYC[ele_to_move, 1] += dy
		
		_nearest_node_ = _find_nearest_node_( _nodeXC, _nodeYC, _nodeZC, _triXC[ele_to_move,1], _triYC[ele_to_move,1], _triZC[ele_to_move,1] )
		(_triXC[ele_to_move,1], _triYC[ele_to_move,1], _triZC[ele_to_move,1]) = (_nodeXC[_nearest_node_], _nodeYC[_nearest_node_], _nodeZC[_nearest_node_])
		
		_triYC[ele_to_move,0] += dy
		_triYC[ele_to_move,2] += dy
		
	if _edge_ == 2:
		_triYC[ele_to_move, 2] += dy
		
		_nearest_node_ = _find_nearest_node_( _nodeXC, _nodeYC, _nodeZC, _triXC[ele_to_move,2], _triYC[ele_to_move,2], _triZC[ele_to_move,2] )
		(_triXC[ele_to_move,2], _triYC[ele_to_move,2], _triZC[ele_to_move,2]) = (_nodeXC[_nearest_node_], _nodeYC[_nearest_node_], _nodeZC[_nearest_node_])
		
		_triYC[ele_to_move,0] += dy
		_triYC[ele_to_move,1] += dy
	
	return _triXC, _triYC, _triZC
	
	
def _moving_nodes_( eles_to_move, _triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC ):
	
	coordx = []
	coordy = []
	coordz = []
	for ktr in range( len(eles_to_move) ):
		ele_to_move = eles_to_move[ktr]
		for vertex in range(3):
			coordx.append( _triXC[ele_to_move,vertex] )
			coordy.append( _triYC[ele_to_move,vertex] )
			coordz.append( _triZC[ele_to_move,vertex] )
	
	coord = np.zeros( (len(coordx),3) )
	for ktr in range( len(coordx) ):
		coord[ktr,0] = coordx[ktr]
		coord[ktr,1] = coordy[ktr]
		coord[ktr,2] = coordz[ktr]
	
	dd = np.zeros( len(coordx) )
	for ktr in range( len(coordx) ):
		dd[ktr] = np.sqrt( coord[ktr,0]**2. + coord[ktr,1]**2. + coord[ktr,2]**2. )
	
	xx = np.unique(dd)
	nodex = np.zeros( len(xx) )
	nodey = np.zeros( len(xx) )
	nodez = np.zeros( len(xx) )
	for ktr in range( len(xx) ):
		idx = np.where( xx[ktr] == dd )[0]
		nodex[ktr] = coord[idx[0],0]
		nodey[ktr] = coord[idx[0],1]
		nodez[ktr] = coord[idx[0],2]
	
	_nodeXC = np.append( _nodeXC, nodex )
	_nodeYC = np.append( _nodeYC, nodey )
	_nodeZC = np.append( _nodeZC, nodez )
	
	return _nodeXC, _nodeYC, _nodeZC
	
