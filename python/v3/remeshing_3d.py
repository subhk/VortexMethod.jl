import numpy as np
import scipy.io as sio
import matplotlib.tri as mtri
from scipy.interpolate import Rbf
from scipy.interpolate import RegularGridInterpolator

from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

#from init_3d import _tri_coord_
from utility_3d_paral import _cal_Normal_Vector


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
	
	
def _eleNode_Nodes_distance(_nodeC0, _eleC1, _eleC2, eps): 
	
	ds01 = _euclid_p2p_3d_dist(_nodeC0, _eleC1)
	ds02 = _euclid_p2p_3d_dist(_nodeC0, _eleC2)
	
	if ds01 >=eps and ds02>=eps:
		flag = True
	else:
		flag = False
	
	return flag
	
	
def _Other_2nodes_( _nodeC0, _eleC0, _eleC1, _eleC2):

	_dist_nodeC0__eleC0 = _euclid_p2p_3d_dist(_nodeC0, _eleC0)
	_dist_nodeC0__eleC1 = _euclid_p2p_3d_dist(_nodeC0, _eleC1)
	_dist_nodeC0__eleC2 = _euclid_p2p_3d_dist(_nodeC0, _eleC2)		
	
	if _dist_nodeC0__eleC0 == 0. and _dist_nodeC0__eleC1 != 0. and _dist_nodeC0__eleC2 != 0.:
		_Ver1_ = _eleC1
		_Ver2_ = _eleC2	
	elif _dist_nodeC0__eleC0 != 0. and _dist_nodeC0__eleC1 == 0. and _dist_nodeC0__eleC2 != 0.:
		_Ver1_ = _eleC0
		_Ver2_ = _eleC2	
	else:
		_Ver1_ = _eleC0
		_Ver2_ = _eleC1
		
	return _Ver1_, _Ver2_
	
	
def _tri_coord_to_node( _triC, _nodeCX, _nodeCY, _nodeCZ ):
	
	indx0 = [ i for i, value in enumerate(_nodeCY) if value == _triC[1] ]
	indx1 = [ i for i, value in enumerate(_nodeCZ) if value == _triC[2] ]
	p = list( set(indx0).intersection(indx1) )
	
	if len(p) != 1:
		raise ValueError( 'Common index could not find -> check remeshing_3d.py file' )
	
	indx = p[0]
	
	return indx
	
	
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
	
	_no_tri_ = _edge_len_.shape[0]
	_tri_sorted_  = []
	_edge_sorted_ = []
	
	for itr in range(_no_tri_):
		
		VAL = _edge_len_[itr,0:3]
		POS = [i for i, j in enumerate(VAL) if j >= ds]
		
		if len(POS) == 1:
			_tri_sorted_.append(itr)
			_edge_sorted_.append(POS[0])
		
		if len(POS) == 2:
			_tri_sorted_.append(itr)
			MAX = max(VAL)
			POS_ = [i for i, j in enumerate(VAL) if j == MAX]
			_edge_sorted_.append(POS_[0])
	
	return _tri_sorted_, _edge_sorted_
	
	
def _sorting_triangles_min_( _edge_len_, ds ):
	
	_no_tri_ = _edge_len_.shape[0]
	_tri_sorted_ = []
	_edge_sorted_ = []
	
	for itr in range(_no_tri_):
		
		VAL = _edge_len_[itr,0:3]
		POS = [i for i, j in enumerate(VAL) if j <= ds]
		
		if len(POS) == 1:
			_tri_sorted_.append(itr)
			_edge_sorted_.append(POS[0])
		
		if len(POS) == 2:
			print('I am here')
			_tri_sorted_.append(itr)
			MIN = min(VAL)
			POS_ = [i for i, j in enumerate(VAL) if j == MIN]
			_edge_sorted_.append(POS_[0])
	
	return _tri_sorted_, _edge_sorted_ 
	
	
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
	
	_cntr = []
	for itr in range( len(_tri_sorted_) ):
		_tri_ele_ = _tri_sorted_[itr]
		
		for jtr in range(3):
			if _triXC[_tri_ele_,jtr] <= 0. or _triXC[_tri_ele_,jtr] <= 0.+ 2*ds: _cntr.append(itr)
			if _triXC[_tri_ele_,jtr] >= 1. or _triXC[_tri_ele_,jtr] >= 1.+ 2*ds: _cntr.append(itr)
			
			if _triYC[_tri_ele_,jtr] <= 0. or _triYC[_tri_ele_,jtr] <= 0.+ 2*ds: _cntr.append(itr)
			if _triYC[_tri_ele_,jtr] >= 1. or _triYC[_tri_ele_,jtr] >= 1.+ 2*ds: _cntr.append(itr)
		
	_tri_sorted_  = np.delete( _tri_sorted_,  _cntr )
	_edge_sorted_ = np.delete( _edge_sorted_, _cntr )
	
	return _tri_sorted_, _edge_sorted_
	
	
def _creating_child_nodes_(_triXC, _triYC, _triZC, _adding_tri_ele_, _edge_sorted_):
	
	_no_child_node = len(_adding_tri_ele_)
	
	_child_nodeXC = []
	_child_nodeYC = []
	_child_nodeZC = []
	
	for itr in range(_no_child_node):
		_tri_ele = _adding_tri_ele_[itr]
		
		if _edge_sorted_[itr] == 0:  
			_child_nodeXC.append( 0.5*(_triXC[_tri_ele,0] + _triXC[_tri_ele,1]) ) 
			_child_nodeYC.append( 0.5*(_triYC[_tri_ele,0] + _triYC[_tri_ele,1]) )
			_child_nodeZC.append( 0.5*(_triZC[_tri_ele,0] + _triZC[_tri_ele,1]) )
		
		if _edge_sorted_[itr] == 1:  
			_child_nodeXC.append( 0.5*(_triXC[_tri_ele,1] + _triXC[_tri_ele,2]) ) 
			_child_nodeYC.append( 0.5*(_triYC[_tri_ele,1] + _triYC[_tri_ele,2]) )
			_child_nodeZC.append( 0.5*(_triZC[_tri_ele,1] + _triZC[_tri_ele,2]) )
		
		if _edge_sorted_[itr] == 2:  
			_child_nodeXC.append( 0.5*(_triXC[_tri_ele,2] + _triXC[_tri_ele,0]) ) 
			_child_nodeYC.append( 0.5*(_triYC[_tri_ele,2] + _triYC[_tri_ele,0]) )
			_child_nodeZC.append( 0.5*(_triZC[_tri_ele,2] + _triZC[_tri_ele,0]) )
	
	return _child_nodeXC, _child_nodeYC, _child_nodeZC
	
	
#def _creating_child_nodes_(_triXC, _triYC, _triZC, _adding_tri_ele_):
#	
#	_no_child_node = len(_adding_tri_ele_)
#	
#	_child_nodeXC = []
#	_child_nodeYC = []
#	_child_nodeZC = []
#	
#	for itr in range(_no_child_node):
#		_tri_ele = _adding_tri_ele_[itr]
#		
#		p1_C = np.array( [_triXC[_tri_ele,0], _triYC[_tri_ele,0], _triZC[_tri_ele,0]] )
#		p2_C = np.array( [_triXC[_tri_ele,1], _triYC[_tri_ele,1], _triZC[_tri_ele,1]] )
#		p3_C = np.array( [_triXC[_tri_ele,2], _triYC[_tri_ele,2], _triZC[_tri_ele,2]] ) 
#		
#		_centroid_ = _Centroid_triangle_( p1_C, p2_C, p3_C )
#		
#		_child_nodeXC.append( _centroid_[0] ) 
#		_child_nodeYC.append( _centroid_[1] )
#		_child_nodeZC.append( _centroid_[2] )
#	
#	return _child_nodeXC, _child_nodeYC, _child_nodeZC
	
	
def _merging_parent_child_nodes_(_triXC, _triYC, _triZC, _adding_tri_ele_, _edge_sorted_, _parent_nodeXC, _parent_nodeYC, _parent_nodeZC):
	
	_child_nodeXC, _child_nodeYC, _child_nodeZC = \
	_creating_child_nodes_(_triXC, _triYC, _triZC, _adding_tri_ele_, _edge_sorted_)
	
	_parent_nodeXC = np.concatenate( (_parent_nodeXC, _child_nodeXC) )
	_parent_nodeYC = np.concatenate( (_parent_nodeYC, _child_nodeYC) )
	_parent_nodeZC = np.concatenate( (_parent_nodeZC, _child_nodeZC) )
	
	return _parent_nodeXC, _parent_nodeYC, _parent_nodeZC
	
	
def _creating_child_triangles_(_parent_triXC, _parent_triYC, _parent_triZC, _parent_tri_verti_, _adding_tri_ele_, _edge_sorted_, len_parent_nodeXC):
	
	_no_child_tri_to_add = len(_adding_tri_ele_)
	
	_child_nodeXC, _child_nodeYC, _child_nodeZC = \
	_creating_child_nodes_(_parent_triXC, _parent_triYC, _parent_triZC, _adding_tri_ele_, _edge_sorted_)
	
	( _triXC0, _triXC1, _triXC2 ) = ( [], [], [] )
	( _triYC0, _triYC1, _triYC2 ) = ( [], [], [] )
	( _triZC0, _triZC1, _triZC2 ) = ( [], [], [] )
	( _triVr0, _triVr1, _triVr2 ) = ( [], [], [] )
	
	for itr in range( len(_adding_tri_ele_) ):
		_tri_ele = _adding_tri_ele_[itr]
		
		if _edge_sorted_[itr] == 0:
			_triXC0.append( _child_nodeXC[itr] )
			_triXC1.append( _parent_triXC[_tri_ele,1] )
			_triXC2.append( _parent_triXC[_tri_ele,2] )
			
			_triYC0.append( _child_nodeYC[itr] )
			_triYC1.append( _parent_triYC[_tri_ele,1] )
			_triYC2.append( _parent_triYC[_tri_ele,2] )
			
			_triZC0.append( _child_nodeZC[itr] )
			_triZC1.append( _parent_triZC[_tri_ele,1] )
			_triZC2.append( _parent_triZC[_tri_ele,2] )
			
			_triVr0.append( len_parent_nodeXC + itr )
			_triVr1.append( _parent_tri_verti_[_tri_ele,1] )
			_triVr2.append( _parent_tri_verti_[_tri_ele,2] )
		
		if _edge_sorted_[itr] == 1:
			_triXC0.append( _parent_triXC[_tri_ele,0] )
			_triXC1.append( _parent_triXC[_tri_ele,1] )
			_triXC2.append( _child_nodeXC[itr] )
			
			_triYC0.append( _parent_triYC[_tri_ele,0] )
			_triYC1.append( _parent_triYC[_tri_ele,1] )
			_triYC2.append( _child_nodeYC[itr] )
			
			_triZC0.append( _parent_triZC[_tri_ele,0] )
			_triZC1.append( _parent_triZC[_tri_ele,1] )
			_triZC2.append( _child_nodeZC[itr] )
			
			_triVr0.append( _parent_tri_verti_[_tri_ele,0] )
			_triVr1.append( _parent_tri_verti_[_tri_ele,1] )
			_triVr2.append( len_parent_nodeXC + itr )
		
		if _edge_sorted_[itr] == 2:
			_triXC0.append( _parent_triXC[_tri_ele,0] )
			_triXC1.append( _parent_triXC[_tri_ele,1] )
			_triXC2.append( _child_nodeXC[itr] )
			
			_triYC0.append( _parent_triYC[_tri_ele,0] )
			_triYC1.append( _parent_triYC[_tri_ele,1] )
			_triYC2.append( _child_nodeYC[itr] )
			
			_triZC0.append( _parent_triZC[_tri_ele,0] )
			_triZC1.append( _parent_triZC[_tri_ele,1] )
			_triZC2.append( _child_nodeZC[itr] )
			
			_triVr0.append( _parent_tri_verti_[_tri_ele,0] )
			_triVr1.append( _parent_tri_verti_[_tri_ele,1] )
			_triVr2.append( len_parent_nodeXC + itr )
	
	_len_ = len(_triXC0)
	
	_child_triXC = np.zeros( (_len_,3) )
	_child_triYC = np.zeros( (_len_,3) )
	_child_triZC = np.zeros( (_len_,3) )
	_child_triVr = np.zeros( (_len_,3) )
	
	for itr in range(_len_):
		
		( _child_triXC[itr,0], _child_triXC[itr,1], _child_triXC[itr,2] ) = ( _triXC0[itr], _triXC1[itr], _triXC2[itr] )
		( _child_triYC[itr,0], _child_triYC[itr,1], _child_triYC[itr,2] ) = ( _triYC0[itr], _triYC1[itr], _triYC2[itr] )
		( _child_triZC[itr,0], _child_triZC[itr,1], _child_triZC[itr,2] ) = ( _triZC0[itr], _triZC1[itr], _triZC2[itr] )
		( _child_triVr[itr,0], _child_triVr[itr,1], _child_triVr[itr,2] ) = ( _triVr0[itr], _triVr1[itr], _triVr2[itr] )
		
		
	for itr in range( len(_adding_tri_ele_) ):
		_tri_ele = _adding_tri_ele_[itr]
		
		if _edge_sorted_[itr] == 0:
			( _parent_triXC[_tri_ele,1], _parent_triYC[_tri_ele,1], _parent_triZC[_tri_ele,1] ) = \
			( _child_nodeXC[itr], _child_nodeYC[itr], _child_nodeZC[itr] )
		
			_parent_tri_verti_[_tri_ele,1] = len_parent_nodeXC + itr 
	
		if _edge_sorted_[itr] == 1:
			( _parent_triXC[_tri_ele,1], _parent_triYC[_tri_ele,1], _parent_triZC[_tri_ele,1] ) = \
			( _child_nodeXC[itr], _child_nodeYC[itr], _child_nodeZC[itr] )
		
			_parent_tri_verti_[_tri_ele,1] = len_parent_nodeXC + itr
		
		if _edge_sorted_[itr] == 2:
			( _parent_triXC[_tri_ele,0], _parent_triYC[_tri_ele,0], _parent_triZC[_tri_ele,0] ) = \
			( _child_nodeXC[itr], _child_nodeYC[itr], _child_nodeZC[itr] )
		
			_parent_tri_verti_[_tri_ele,0] = len_parent_nodeXC + itr
		
	_parent_triXC = np.concatenate( (_parent_triXC, _child_triXC) )
	_parent_triYC = np.concatenate( (_parent_triYC, _child_triYC) )
	_parent_triZC = np.concatenate( (_parent_triZC, _child_triZC) )
	_parent_tri_verti_ = np.concatenate( (_parent_tri_verti_, _child_triVr) )
	
	return _parent_triXC, _parent_triYC, _parent_triZC, _parent_tri_verti_
	
	
def _reassign_eleGma_on_added_tri_( _parent_eleGma_, _adding_tri_ele_ ):
	
	( _child_eleGma0_, _child_eleGma1_, _child_eleGma2_  ) = ( [], [], []  )
	
	for itr in range( len(_adding_tri_ele_) ):
		_tri_ele = _adding_tri_ele_[itr]
		
		_child_eleGma0_.append( _parent_eleGma_[_tri_ele,0] )
		_child_eleGma1_.append( _parent_eleGma_[_tri_ele,1] )
		_child_eleGma2_.append( _parent_eleGma_[_tri_ele,2] )
	
	_len_ = len(_child_eleGma0_)
	_child_eleGma_ = np.zeros( (_len_,3) )
	
	_child_eleGma_[0:_len_,0] = _child_eleGma0_[0:_len_]
	_child_eleGma_[0:_len_,1] = _child_eleGma1_[0:_len_]
	_child_eleGma_[0:_len_,2] = _child_eleGma2_[0:_len_]
	
	_parent_eleGma_ = np.concatenate( (_parent_eleGma_, _child_eleGma_) )
	
	return _parent_eleGma_
	
	
#def _reassign_nodeCirc_on_added_nodes_( _parent_nodeCirc_, _adding_tri_ele_ ):
#	
#	_len_ = len(_adding_tri_ele_)
#	_child_nodeCirc_ = np.zeros( (_length_,3,3) )
#	
#	for itr in range(_length_):
#		_tri = _parent_tri_sorted_[itr]
#		_child_nodeCirc_[itr,0,0:3] = np.array( [ _parent_nodeCirc_[_tri,0], _parent_nodeCirc_[_tri,1], _parent_nodeCirc_[_tri,2] ] )
#		_child_nodeCirc_[itr,1,0:3] = np.array( [ _parent_nodeCirc_[_tri,2], _parent_nodeCirc_[_tri,1], _parent_nodeCirc_[_tri,3] ] )
#		_child_nodeCirc_[itr,2,0:3] = np.array( [ _parent_nodeCirc_[_tri,1], _parent_nodeCirc_[_tri,2], _parent_nodeCirc_[_tri,3] ] )
#		
#		if sum(_child_nodeCirc_[itr,0,:]) != 0 and sum(_child_nodeCirc_[itr,1,:]) != 0 and sum(_child_nodeCirc_[itr,2,:]) != 0:
#			raise ValueError( 'Circulation must be conserved -> Check remeshing_3d.py file' )
#	
#	return _child_nodeCirc_
	
	
def _delete_straddle_elemenents_x_dir_(_triXC, _triYC, _triZC, _no_tri_, _triVT, Lx, ds):
	
	_no_tri0_ = np.shape(_triVT)[0]
	( _triXC0, _triXC1, _triXC2 ) = ( [], [], [] )
	( _triYC0, _triYC1, _triYC2 ) = ( [], [], [] )
	( _triZC0, _triZC1, _triZC2 ) = ( [], [], [] )
	( _triVr0, _triVr1, _triVr2 ) = ( [], [], [] )
	
	for itr in range(_no_tri_):
		flag = True
		MAX = max( _triXC[itr,0:3] )
		MIN = min( _triXC[itr,0:3] )
		
		if MIN > 0.: MAX = MAX
		if MIN < 0.: MAX = MIN 
		
		if MAX > -ds and MAX < Lx + ds:
			_triXC0.append( _triXC[itr,0] )
			_triXC1.append( _triXC[itr,1] )
			_triXC2.append( _triXC[itr,2] )
			
			_triYC0.append( _triYC[itr,0] )
			_triYC1.append( _triYC[itr,1] )
			_triYC2.append( _triYC[itr,2] )
			
			_triZC0.append( _triZC[itr,0] )
			_triZC1.append( _triZC[itr,1] )
			_triZC2.append( _triZC[itr,2] )
			
			_triVr0.append( _triVT[itr,0] )
			_triVr1.append( _triVT[itr,1] )
			_triVr2.append( _triVT[itr,2] )
			flag = False
			#print('_triXC = ', MAX)
	
	_len_ = len(_triXC0)
	print('_length = ', _len_)
	
	_nw_triXC = np.zeros( (_len_,3) )
	_nw_triYC = np.zeros( (_len_,3) )
	_nw_triZC = np.zeros( (_len_,3) )
	_nw_triVr = np.zeros( (_len_,3) )
	
	( _nw_triXC[0:_len_,0], _nw_triXC[0:_len_,1], _nw_triXC[0:_len_,2] ) = ( _triXC0[0:_len_], _triXC1[0:_len_], _triXC2[0:_len_] )
	( _nw_triYC[0:_len_,0], _nw_triYC[0:_len_,1], _nw_triYC[0:_len_,2] ) = ( _triYC0[0:_len_], _triYC1[0:_len_], _triYC2[0:_len_] )
	( _nw_triZC[0:_len_,0], _nw_triZC[0:_len_,1], _nw_triZC[0:_len_,2] ) = ( _triZC0[0:_len_], _triZC1[0:_len_], _triZC2[0:_len_] )
	( _nw_triVr[0:_len_,0], _nw_triVr[0:_len_,1], _nw_triVr[0:_len_,2] ) = ( _triVr0[0:_len_], _triVr1[0:_len_], _triVr2[0:_len_] )
	
	_no_tri_ = np.shape(_nw_triVr)[0]
	diff = _no_tri0_ - _no_tri_
	print( 'No of straddle elements remove = ', diff )
	
	return _nw_triXC, _nw_triYC, _nw_triZC, _no_tri_, _nw_triVr
	
	
def _delete_straddle_elemenents_y_dir_(_triXC, _triYC, _triZC, _no_tri_, _triVT, Ly, ds):
	
	_no_tri0_ = np.shape(_triVT)[0]
	( _triXC0, _triXC1, _triXC2 ) = ( [], [], [] )
	( _triYC0, _triYC1, _triYC2 ) = ( [], [], [] )
	( _triZC0, _triZC1, _triZC2 ) = ( [], [], [] )
	( _triVr0, _triVr1, _triVr2 ) = ( [], [], [] )
	
	for itr in range(_no_tri_):
		flag = True
		MAX = max( _triYC[itr,0:3] )
		MIN = min( _triYC[itr,0:3] )
		
		if MIN > 0.: MAX = MAX
		if MIN < 0.: MAX = MIN 
		
		if MAX > -ds and MAX < Ly + ds:
			_triXC0.append( _triXC[itr,0] )
			_triXC1.append( _triXC[itr,1] )
			_triXC2.append( _triXC[itr,2] )
			
			_triYC0.append( _triYC[itr,0] )
			_triYC1.append( _triYC[itr,1] )
			_triYC2.append( _triYC[itr,2] )
			
			_triZC0.append( _triZC[itr,0] )
			_triZC1.append( _triZC[itr,1] )
			_triZC2.append( _triZC[itr,2] )
			
			_triVr0.append( _triVT[itr,0] )
			_triVr1.append( _triVT[itr,1] )
			_triVr2.append( _triVT[itr,2] )
			flag = False
	
	_len_ = len(_triXC0)
	
	_nw_triXC = np.zeros( (_len_,3) )
	_nw_triYC = np.zeros( (_len_,3) )
	_nw_triZC = np.zeros( (_len_,3) )
	_nw_triVr = np.zeros( (_len_,3) )
	
	( _nw_triXC[0:_len_,0], _nw_triXC[0:_len_,1], _nw_triXC[0:_len_,2] ) = ( _triXC0[0:_len_], _triXC1[0:_len_], _triXC2[0:_len_] )
	( _nw_triYC[0:_len_,0], _nw_triYC[0:_len_,1], _nw_triYC[0:_len_,2] ) = ( _triYC0[0:_len_], _triYC1[0:_len_], _triYC2[0:_len_] )
	( _nw_triZC[0:_len_,0], _nw_triZC[0:_len_,1], _nw_triZC[0:_len_,2] ) = ( _triZC0[0:_len_], _triZC1[0:_len_], _triZC2[0:_len_] )
	( _nw_triVr[0:_len_,0], _nw_triVr[0:_len_,1], _nw_triVr[0:_len_,2] ) = ( _triVr0[0:_len_], _triVr1[0:_len_], _triVr2[0:_len_] )
	
	_no_tri_ = np.shape(_nw_triVr)[0]
	diff = _no_tri0_ - _no_tri_
	print( 'No of straddle elements remove = ', diff )
	
	return _nw_triXC, _nw_triYC, _nw_triZC, _no_tri_, _nw_triVr
	
	
def _delete_straddle_nodes_(_nodeXC, _nodeYC, _nodeZC, ds, Lx, Ly): # _triXC, _triYC, _triZC, _no_tri_, _tri_verti_):
	
	_no_nodes0_ = len(_nodeXC)
	
#	( _nodeXC0, _nodeYC0, _nodeZC0 ) = ( [], [], [] )
#	
#	for itr in range(_no_nodes0_):
#		if _nodeXC[itr] > -ds and _nodeXC[itr] < Lx + ds:
#			if _nodeYC[itr] > -ds and _nodeYC[itr] < Ly + ds:
#				_nodeXC0.append(_nodeXC[itr])
#				_nodeYC0.append(_nodeYC[itr])
#				_nodeZC0.append(_nodeZC[itr])
#				
#	_no_nodes_ = len(_nodeXC0)
#	diff = _no_nodes0_ - _no_nodes_
	
	_delete_nodes = []
	for itr in range(_no_nodes0_):
		flagx = True
		if _nodeXC[itr] < -ds:
			_delete_nodes.append(itr)
			flagx = False
		
		if flagx:
			if _nodeXC[itr] > Lx+ds:
				_delete_nodes.append(itr)
				flagx = False
		
		if flagx:
			if _nodeYC[itr] < -ds:
				_delete_nodes.append(itr)
				flagx = False
				
		if flagx:
			if _nodeYC[itr] > Lx+ds:
				_delete_nodes.append(itr)
	
	_delete_nodes.sort()
	
	_nodeXC = np.delete( _nodeXC, _delete_nodes )
	_nodeYC = np.delete( _nodeYC, _delete_nodes )
	_nodeZC = np.delete( _nodeZC, _delete_nodes )
	
	print( 'No of straddle nodes remove = ', len(_delete_nodes) )
	
	return _nodeXC, _nodeYC, _nodeZC, _delete_nodes
	
	
def _delete_straddle_elemenents_(_triXC, _triYC, _triZC, _no_tri_, _triVT, _nodeXC, _nodeYC, _nodeZC, _delete_nodes): # , ds, Lx, Ly):
	 
	_len_ = len(_delete_nodes)
	_delete_nodes.sort()
	
	_tri_ele_share_comm_node_ = []
	if _len_ > 0:
#		_num_ele_share_comm_node_ = np.zeros(_len_)
		
		for itr in range(_len_):
			_node_ = _delete_nodes[itr]
			#print('_node_ = ', _node_)
			for jtr in range( len(_triVT[:,0]) ):
				if _triVT[jtr,0] == _node_: _tri_ele_share_comm_node_.append( jtr )
				if _triVT[jtr,1] == _node_: _tri_ele_share_comm_node_.append( jtr )
				if _triVT[jtr,2] == _node_: _tri_ele_share_comm_node_.append( jtr ) 
		
	_tri_ele_share_comm_node_ = np.unique(_tri_ele_share_comm_node_).tolist()
	_tri_ele_share_comm_node_.sort()
	
#	print( '_tri_ele_share_comm_node_ = ', _tri_ele_share_comm_node_ )
	
	if len(_tri_ele_share_comm_node_) > 0:
		_triXC = np.delete( _triXC, _tri_ele_share_comm_node_, axis=0 )
		_triYC = np.delete( _triYC, _tri_ele_share_comm_node_, axis=0 )
		_triZC = np.delete( _triZC, _tri_ele_share_comm_node_, axis=0 )
		_triVT = np.delete( _triVT, _tri_ele_share_comm_node_, axis=0 )
	
	if _len_ > 0:
		_triVt0 = []
		_triVt1 = []
		_triVt2 = []
		for ktr in range( len(_triXC[:,0]) ):
			indx0 = np.where(_triXC[ktr,0] == _nodeXC)[0]
			indx1 = np.where(_triXC[ktr,1] == _nodeXC)[0]
			indx2 = np.where(_triXC[ktr,2] == _nodeXC)[0]
			
			indy0 = np.where(_triYC[ktr,0] == _nodeYC)[0]
			indy1 = np.where(_triYC[ktr,1] == _nodeYC)[0]
			indy2 = np.where(_triYC[ktr,2] == _nodeYC)[0]
			
			p0 =  np.intersect1d( indx0, indy0 )
			p1 =  np.intersect1d( indx1, indy1 )
			p2 =  np.intersect1d( indx2, indy2 )
			
			_triVt0.append(p0[0])
			_triVt1.append(p1[0])
			_triVt2.append(p2[0])
		
#		print( 'length  = ', len(_triVt0) )
		
		_triVT0 = np.zeros( (len(_triVt0), 3) )
		_triVT0[0:len(_triVt0),0] = _triVt0[0:len(_triVt0)]
		_triVT0[0:len(_triVt0),1] = _triVt1[0:len(_triVt0)]
		_triVT0[0:len(_triVt0),2] = _triVt2[0:len(_triVt0)]
		
	else:
		
		_triVT0 = _triVT
		
#	if _len_ > 0:
#		for ktr in range( len(_triVT[:,0]) ):
#				for itr in range(_len_):
#					_node_ = _delete_nodes[itr]
#					if any(_triVT[ktr,0:3]) == _node_ and _node_ != 0:
#						_triVT[ktr,0] = _triVT[ktr,0]
#						_triVT[ktr,1] = _triVT[ktr,1]
#						_triVT[ktr,2] = _triVT[ktr,2]
#					else:
#						if _triVT[ktr,0] > _node_ and _triVT[ktr,0] != 0: _triVT[ktr,0] += - 1
#						if _triVT[ktr,1] > _node_ and _triVT[ktr,1] != 0: _triVT[ktr,1] += - 1
#						if _triVT[ktr,2] > _node_ and _triVT[ktr,2] != 0: _triVT[ktr,2] += - 1
		
		
	_no_tri_ = len(_triVT0[:,0])
	
#	print( 'no of triangle = ', _no_tri_ )
	
	return _triXC, _triYC, _triZC, _no_tri_, _triVT0, _tri_ele_share_comm_node_
	
	
def _delete_straddle_nodeVel_(_uXg_, _vYg_, _wZg_, _delete_nodes):
	
	if len(_delete_nodes) > 0:
		_uXg_ = np.delete( _uXg_, _delete_nodes )
		_vYg_ = np.delete( _vYg_, _delete_nodes )
		_wZg_ = np.delete( _wZg_, _delete_nodes )
	
	return _uXg_, _vYg_, _wZg_
	
	
def _delete_straddle_nodeCirculation_( _nodeCirc, _tri_ele_share_comm_node_ ):
	
	if len(_tri_ele_share_comm_node_) > 0:
		_nodeCirc = np.delete( _nodeCirc, _tri_ele_share_comm_node_, axis=0 )
	
	return _nodeCirc
	
	
def _delete_straddle_eleGma_( _eleGma, _tri_ele_share_comm_node_ ):
	
	if len(_tri_ele_share_comm_node_) > 0:
		_eleGma = np.delete( _eleGma, _tri_ele_share_comm_node_, axis=0 )
	
	return _eleGma
	
	
def _element_merging_below_critical_length_\
(_triXC, _triYC, _triZC, tri_verti_, _delete_tri_ele_, _edge_sorted_, _parent_nodeXC, _parent_nodeYC, _parent_nodeZC, _Triangle_Area_, _eleGma):
	
	#print( _delete_tri_ele_ )
	_equal_len_ = np.array_equal( len(_delete_tri_ele_), len(_edge_sorted_) )
	
	_nw_triXC = np.copy( _triXC )
	_nw_triYC = np.copy( _triYC )
	_nw_triZC = np.copy( _triZC )
	
	_nw_nodeXC = np.copy( _parent_nodeXC )
	_nw_nodeYC = np.copy( _parent_nodeXC )
	_nw_nodeZC = np.copy( _parent_nodeXC )
	
	_nw_eleGma = np.copy( _eleGma )
	
	if not _equal_len_: raise ValueError  ( ' mismatch between no of triangles and no of min edges ' )
	
	_num_triVertex_ = np.zeros(2)
	
	_coord0_ = np.zeros( (len(_delete_tri_ele_), 3) )
	_coord1_ = np.zeros( (len(_delete_tri_ele_), 3) )
	
	for itr in range( len(_delete_tri_ele_) ):
		_tri_ele_ = _delete_tri_ele_[itr]
		
		if _edge_sorted_[itr] == 0: 
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			
		if _edge_sorted_[itr] == 1: 
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			
		if _edge_sorted_[itr] == 2: 
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
	
	
	for itr in range( len(_delete_tri_ele_) ):
		
		indx1 = np.where( _coord0_[itr,0] == _triXC )[0]
		indy1 = np.where( _coord0_[itr,1] == _triYC )[0]
		
		indx2 = np.where( _coord1_[itr,0] == _triXC )[0]
		indy2 = np.where( _coord1_[itr,1] == _triYC )[0]
		
		p0 = np.intersect1d( indx1, indy1 )
		p1 = np.intersect1d( indx2, indy2 )
		
		print( _coord0_[itr] )
		print( _coord1_[itr] )
		
		print(p0)
		print(p1)
		
		_comm_tri_ele_ = np.intersect1d( p0, p1 )
		if len(_comm_tri_ele_) != 2: 
			print( _comm_tri_ele_ )
			raise ValueError( 'Two common triangles could not found' )
		
		_area1_ = _Triangle_Area_[ _comm_tri_ele_[0] ]
		_area2_ = _Triangle_Area_[ _comm_tri_ele_[1] ]
		
		_delete_ = []
		for i in range( len(p0) ):
			if p0[i] == _comm_tri_ele_[0]: _delete_.append(i)
			if p0[i] == _comm_tri_ele_[1]: _delete_.append(i)
		
		p0 = np.delete(p0, _delete_) 
		
		_delete_ = []
		for i in range( len(p1) ):
			if p1[i] == _comm_tri_ele_[0]: _delete_.append(i)
			if p1[i] == _comm_tri_ele_[1]: _delete_.append(i)
		
		p1 = np.delete(p1, _delete_) 
		
		if len(p0) > 0:
			for jtr in range( len(p0) ):
				_tri_ele_ = p0[jtr]
				if _nw_triXC[_tri_ele_,0] == _coord0_[itr, 0] and _nw_triYC[_tri_ele_,0] == _coord0_[itr, 1]:
					_nw_triXC[_tri_ele_,0] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
					_nw_triXC[_tri_ele_,0] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
					_nw_triXC[_tri_ele_,0] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
				
				if _nw_triXC[_tri_ele_,1] == _coord0_[itr, 0] and _nw_triYC[_tri_ele_,1] == _coord0_[itr, 1]:
					_nw_triXC[_tri_ele_,1] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
					_nw_triXC[_tri_ele_,1] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
					_nw_triXC[_tri_ele_,1] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
				
				if _nw_triXC[_tri_ele_,2] == _coord0_[itr, 0] and _nw_triYC[_tri_ele_,2] == _coord0_[itr, 1]:
					_nw_triXC[_tri_ele_,2] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
					_nw_triXC[_tri_ele_,2] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
					_nw_triXC[_tri_ele_,2] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
				
				p1C = np.array( [ _nw_triXC[_tri_ele_,0], _nw_triYC[_tri_ele_,0], _nw_triZC[_tri_ele_,0] ] )
				p2C = np.array( [ _nw_triXC[_tri_ele_,1], _nw_triYC[_tri_ele_,1], _nw_triZC[_tri_ele_,1] ] )
				p3C = np.array( [ _nw_triXC[_tri_ele_,2], _nw_triYC[_tri_ele_,2], _nw_triZC[_tri_ele_,2] ] )
				
				_Area_ = _oneTriangle_Area_( p1C, p2C, p3C )
				_nw_eleGma[_tri_ele_] = ( _area1_*_eleGma[ _comm_tri_ele_[0] ] + _Triangle_Area_[_tri_ele_]*_eleGma[_tri_ele_] )/_Area_
		
		if len(p1) > 0:
			for jtr in range( len(p1) ):
				_tri_ele_ = p1[jtr]
				if _nw_triXC[_tri_ele_,0] == _coord1_[itr, 0] and _nw_triYC[_tri_ele_,0] == _coord1_[itr, 1]:
					_nw_triXC[_tri_ele_,0] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
					_nw_triXC[_tri_ele_,0] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
					_nw_triXC[_tri_ele_,0] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
				
				if _nw_triXC[_tri_ele_,1] == _coord1_[itr, 0] and _nw_triYC[_tri_ele_,1] == _coord1_[itr, 1]:
					_nw_triXC[_tri_ele_,1] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
					_nw_triXC[_tri_ele_,1] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
					_nw_triXC[_tri_ele_,1] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
				
				if _nw_triXC[_tri_ele_,2] == _coord1_[itr, 0] and _nw_triYC[_tri_ele_,2] == _coord1_[itr, 1]:
					_nw_triXC[_tri_ele_,2] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
					_nw_triXC[_tri_ele_,2] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
					_nw_triXC[_tri_ele_,2] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
			
				p1C = np.array( [ _nw_triXC[_tri_ele_,0], _nw_triYC[_tri_ele_,0], _nw_triZC[_tri_ele_,0] ] )
				p2C = np.array( [ _nw_triXC[_tri_ele_,1], _nw_triYC[_tri_ele_,1], _nw_triZC[_tri_ele_,1] ] )
				p3C = np.array( [ _nw_triXC[_tri_ele_,2], _nw_triYC[_tri_ele_,2], _nw_triZC[_tri_ele_,2] ] )
				
				_Area_ = _oneTriangle_Area_( p1C, p2C, p3C )
				_nw_eleGma[_tri_ele_] = ( _area2_*_eleGma[ _comm_tri_ele_[1] ] + _Triangle_Area_[_tri_ele_]*_eleGma[_tri_ele_] )/_Area_
			
	# deleting the Lagrangian nodes
	p = []
	for itr in range( len(_coord0_) ):
		indx1 = np.where( _coord0_[itr,0] == _nw_nodeXC )[0]
		indy1 = np.where( _coord0_[itr,1] == _nw_nodeYC )[0]
		p0 = np.intersect1d( indx1, indy1 )
		p.append(p0)
		
	_nw_nodeXC = np.delete( _nw_nodeXC, p )
	_nw_nodeYC = np.delete( _nw_nodeYC, p )
	_nw_nodeZC = np.delete( _nw_nodeZC, p )
	
	_nw_triXC  = np.delete( _nw_triXC,  _delete_tri_ele_, axis=0 )
	_nw_triYC  = np.delete( _nw_triYC,  _delete_tri_ele_, axis=0 )
	_nw_triZC  = np.delete( _nw_triZC,  _delete_tri_ele_, axis=0 )
	_nw_eleGma = np.delete( _nw_eleGma, _delete_tri_ele_, axis=0 )
	
	_triVt0 = []
	_triVt1 = []
	_triVt2 = []
	for ktr in range( len(_nw_triXC[:,0]) ):
		indx0 = np.where( _nw_triXC[ktr,0] == _nw_nodeXC )[0]
		indx1 = np.where( _nw_triXC[ktr,1] == _nw_nodeXC )[0]
		indx2 = np.where( _nw_triXC[ktr,2] == _nw_nodeXC )[0]
		
		indy0 = np.where( _nw_triYC[ktr,0] == _nw_nodeYC )[0]
		indy1 = np.where( _nw_triYC[ktr,1] == _nw_nodeYC )[0]
		indy2 = np.where( _nw_triYC[ktr,2] == _nw_nodeYC )[0]
		
		p0 =  np.intersect1d( indx0, indy0 )
		p1 =  np.intersect1d( indx1, indy1 )
		p2 =  np.intersect1d( indx2, indy2 )
		
		_triVt0.append(p0[0])
		_triVt1.append(p1[0])
		_triVt2.append(p2[0])
	
	_triVT0 = np.zeros( (len(_triVt0), 3) )
	_triVT0[0:len(_triVt0),0] = _triVt0[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),1] = _triVt1[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),2] = _triVt2[0:len(_triVt0)]
	
	return _nw_triXC, _nw_triYC, _nw_triZC, _tri_VT0_, _nw_nodeXC, _nw_nodeYC, _nw_nodeZC, _nw_eleGma
	
	
def _finding_comm_edge_btwn_two_elements_( _Tri_ele1, _Tri_ele2 ):
	
	_comm_edge1_ = []
	_comm_edge2_ = []
	
	for itr in range(3):
		for jtr in range(3):
			if _Tri_ele1[itr,0] == _Tri_ele2[jtr,0] and _Tri_ele1[itr,1] == _Tri_ele2[jtr,1]:
				_comm_edge1_.append(itr)
				_comm_edge2_.append(jtr)
	
	if len(_comm_edge1_) == 1: print('one common edge found')
	if len(_comm_edge1_) == 0: print('no  common edge found')
	if len(_comm_edge1_) >= 2: raise ValueError('Error - more than one edge')
	
	return _comm_edge1_, _comm_edge2_
	
	
class _elements_splitting_(object):
	
	def __init__(self, _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_, del_split):
		self._nodeXC = _nodeXC
		self._nodeYC = _nodeYC
		self._nodeZC = _nodeZC
		self._no_tri_ = _no_tri_
		self.tri_verti_ = tri_verti_
		self.del_split = del_split
	
	def _TriVert_Cord(self):  # not working: please not to use
	
		_TriVert_p1C = np.zeros( (self._no_tri_,3) )
		_TriVert_p2C = np.zeros( (self._no_tri_,3) )
		_TriVert_p3C = np.zeros( (self._no_tri_,3) )
	
		#xT, yT, zT = _tri_coord_(self._xg, self._yg, self._zg, self._no_triangle_, self.tri_vertices_)
		
		for _tri_ in range(self._no_tri_):
		
			_TriVert_p1C[_tri_,0] = self._nodeXC[self.tri_verti_[_tri_,0]]
			_TriVert_p1C[_tri_,1] = self._nodeYC[self.tri_verti_[_tri_,0]]
			_TriVert_p1C[_tri_,2] = self._nodeZC[self.tri_verti_[_tri_,0]]
			
			_TriVert_p2C[_tri_,0] = self._nodeXC[self.tri_verti_[_tri_,1]]
			_TriVert_p2C[_tri_,1] = self._nodeYC[self.tri_verti_[_tri_,1]]
			_TriVert_p2C[_tri_,2] = self._nodeZC[self.tri_verti_[_tri_,1]]
			
			_TriVert_p3C[_tri_,0] = self._nodeXC[self.tri_verti_[_tri_,2]]
			_TriVert_p3C[_tri_,1] = self._nodeYC[self.tri_verti_[_tri_,2]]
			_TriVert_p3C[_tri_,2] = self._nodeZC[self.tri_verti_[_tri_,2]]
		
		return _TriVert_p1C, _TriVert_p2C, _TriVert_p3C
	
	
#	def _remove_boundary_elements_(self):
#	
#		_loc_sorted_p1p2_,_ele_sorted_p1p2, _loc_sorted_p2p3_,_ele_sorted_p2p3, _loc_sorted_p3p1_,_ele_sorted_p3p1 =\
#		self._max_edge_length_dection_()
		
	def _finding_elements_common_vertex_(self, _triXC, _triYC, _triZC, p1_C):
			
		_common_eles_ = []
		
		for itr in range(self._no_tri_):
			flag = True 		
			if p1_C[0] == _triXC[itr,0] and p1_C[1] == _triYC[itr,0] and p1_C[2] == _triZC[itr,0]:
				_common_eles_.append(itr)
				flag = False
				
			if flag:
				if p1_C[0] == _triXC[itr,1] and p1_C[1] == _triYC[itr,1] and p1_C[2] == _triZC[itr,1]:
					_common_eles_.append(itr)
					flag = False
					
			if flag:		
				if p1_C[0] == _triXC[itr,2] and p1_C[1] == _triYC[itr,2] and p1_C[2] == _triZC[itr,2]:
					_common_eles_.append(itr)
		
		_common_eles_ = np.unique(_common_eles_).tolist()
		
		length = len(_common_eles_)
	
		return _common_eles_, length
	
	
	def _finding_elements_common_edge_(self, _triXC, _triYC, _triZC, p1_C, p2_C):
		
		_common_eles_ = []
		
		for itr in range(self._no_tri_):
			
			flag = True
			
			if p1_C[0] == _triXC[itr,0] and p1_C[1] == _triYC[itr,0] and p1_C[2] == _triZC[itr,0]:
				if p2_C[0] == _triXC[itr,1] and p2_C[1] == _triYC[itr,1] and p2_C[2] == _triZC[itr,1]:
					_common_eles_.append(itr)
					flag = False
				if flag:
					if p2_C[0] == _triXC[itr,2] and p2_C[1] == _triYC[itr,2] and p2_C[2] == _triZC[itr,2]:
						_common_eles_.append(itr)
						flag = False
		
			if flag:
				if p1_C[0] == _triXC[itr,1] and p1_C[1] == _triYC[itr,1] and p1_C[2] == _triZC[itr,1]:
					if p2_C[0] == _triXC[itr,0] and p2_C[1] == _triYC[itr,0] and p2_C[2] == _triZC[itr,0]:
						_common_eles_.append(itr)
						flag = False
				if flag:
					if p2_C[0] == _triXC[itr,2] and p2_C[1] == _triYC[itr,2] and p2_C[2] == _triZC[itr,2]:
						_common_eles_.append(itr)
						flag = False

			if flag:
				if p1_C[0] == _triXC[itr,2] and p1_C[1] == _triYC[itr,2] and p1_C[2] == _triZC[itr,2]:
					if p2_C[0] == _triXC[itr,0] and p2_C[1] == _triYC[itr,0] and p2_C[2] == _triZC[itr,0]:
						_common_eles_.append(itr)
						flag = False
				if flag:
					if p2_C[0] == _triXC[itr,1] and p2_C[1] == _triYC[itr,1] and p2_C[2] == _triZC[itr,1]:
						_common_eles_.append(itr)
						flag = False

		_common_eles_ = np.unique(_common_eles_).tolist()
	
		return _common_eles_
		
	
	def _cal_common_elements_each_node_(self, flag):	
	
		if flag:
			Obj_ = _cal_Normal_Vector( self._nodeXC, self._nodeYC, self._nodeZC, self._no_tri_, self.tri_verti_ ) 
			_ele_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
		
			_triXC, _triYC, _triZC = _tri_coord_( self._nodeXC, self._nodeYC, self._nodeZC, self._no_tri_, self.tri_verti_ )
			_tot_nodes = len(self._nodeXC)
			
			_node_share_cmn_eles_ = np.zeros( (_tot_nodes, 25), dtype='int' )
			_leng_ = np.zeros(_tot_nodes, dtype='int')
			
			for _node in range(_tot_nodes):
			
				_node_Coord = np.array([self._nodeXC[_node], self._nodeYC[_node], self._nodeZC[_node]])
					
				_tmp_, _leng_[_node] = self._finding_elements_common_vertex_(_triXC, _triYC, _triZC, _node_Coord)
								
				for i in range(_leng_[_node]):
					_node_share_cmn_eles_[_node, i] = _tmp_[i]
		
		return 	_node_share_cmn_eles_, _leng_
	
	
	def	_finding_allelements_around_node_(self, _triXC, _triYC, _triZC, _nodeC0, gr_pts, eps):
		
		# store all the nodes around a node around a critical distance	
		_node_list, length = _allNodes_around_aNode_( self._nodeXC, self._nodeYC, self._nodeZC, _nodeC0, eps )
		
		#print('_node_list = ', _node_list)
		
		_nodeElemenstList= []
		#print('length = ', length)
			
		for _len in range(length):
			
			p1_C = np.array( [ self._nodeXC[_node_list[_len]], self._nodeYC[_node_list[_len]], self._nodeZC[_node_list[_len]] ] )
			_common_eles_, _ = self._finding_elements_common_vertex_( _triXC, _triYC, _triZC, p1_C )
			_nodeElemenstList.extend(_common_eles_)
		
		_nodeElemenstList = np.unique(_nodeElemenstList).tolist()		
		_leng_ = len(_nodeElemenstList)
		#print(_nodeElemenstList)		
		
		return _nodeElemenstList, _leng_
		
		
#	def _assign_child_vorticity_( self, _parent_tri_sorted_, _parent_eleGma_ ):
#		
#		_length_ = len(_patrent_tri_sorted_)
#		_child_eleGma_ = np.zeros( (_length_,3) )
#		
#		for itr in range(_length_):	
#			_tri = _parent_tri_sorted_[itr]
#			_child_eleGma_[itr,0:3] = _parent_eleGma_[_tri,0:3]
#			
#		return _child_eleGma_
		
		
	def _assign_child_nodeCirc_( self, _parent_tri_sorted_, _parent_nodeCirc_ ):
		
		_length_ = len(_patrent_tri_sorted_)
		_child_nodeCirc_ = np.zeros( (_length_,3,3) )
		
		for itr in range(_length_):
			_tri = _parent_tri_sorted_[itr]
			_child_nodeCirc_[itr,0,0:3] = np.array( [ _parent_nodeCirc_[_tri,0], _parent_nodeCirc_[_tri,1], _parent_nodeCirc_[_tri,2] ] )
			_child_nodeCirc_[itr,1,0:3] = np.array( [ _parent_nodeCirc_[_tri,2], _parent_nodeCirc_[_tri,3], _parent_nodeCirc_[_tri,1] ] )
			_child_nodeCirc_[itr,2,0:3] = np.array( [ _parent_nodeCirc_[_tri,3], _parent_nodeCirc_[_tri,1], _parent_nodeCirc_[_tri,2] ] )
			
			if sum(_child_nodeCirc_[itr,0,:]) != 0 and sum(_child_nodeCirc_[itr,1,:]) != 0 and sum(_child_nodeCirc_[itr,2,:]) != 0:
				raise ValueError( 'Circulation must be conserved -> Check remeshing_3d.py file' )
		
		return _child_nodeCirc_
		
		
	def _assign_child_nodeVel_\
	( self, _parent_tri_sorted_, _parent_nodeVelX_, _parent_nodeVelY_, _parent_nodeVelZ_, _TriVert_p1C, _TriVert_p2C, _TriVert_p3C ):
		
		_no_child_tri, _no_child_node, _child_nodeXC, _child_nodeYC, _child_nodeZC, _, _, _ = self._remeshing_elements_()
		
		#_TriVert_p1C, _TriVert_p2C, _TriVert_p3C = self._TriVert_Cord()
		
		_length_ = len(_patrent_tri_sorted_)		
		_child_nodeVel_ = np.zeros( (_no_child_node,3) )
		
		for itr in range(_no_child_node):
			_tri = _parent_tri_sorted_[itr]
			
			_triC1 = np.array( [ _TriVert_p1C[_tri_ele_,0], _TriVert_p1C[_tri_ele_,1], _TriVert_p1C[_tri_ele_,2] ] )
			_triC2 = np.array( [ _TriVert_p2C[_tri_ele_,0], _TriVert_p2C[_tri_ele_,1], _TriVert_p2C[_tri_ele_,2] ] )
			_triC3 = np.array( [ _TriVert_p3C[_tri_ele_,0], _TriVert_p3C[_tri_ele_,1], _TriVert_p3C[_tri_ele_,2] ] )
			
			indx1 = _tri_coord_to_node( _triC1, self._nodeCX, self._nodeCY, self._nodeCZ ) 
			indx2 = _tri_coord_to_node( _triC1, self._nodeCX, self._nodeCY, self._nodeCZ )
			indx3 = _tri_coord_to_node( _triC1, self._nodeCX, self._nodeCY, self._nodeCZ )
			
			xx = np.array( [ self._nodeCX[indx1], self._nodeCX[indx2], self._nodeCX[indx3] ] )
			yy = np.array( [ self._nodeCY[indx1], self._nodeCY[indx2], self._nodeCY[indx3] ] )
			zz = np.array( [ self._nodeCZ[indx1], self._nodeCZ[indx2], self._nodeCZ[indx3] ] )
			
			ux = np.array( [ _parent_nodeVelX_[indx1], _parent_nodeVelX_[indx2], _parent_nodeVelX_[indx3] ] )
			vy = np.array( [ _parent_nodeVelY_[indx1], _parent_nodeVelY_[indx2], _parent_nodeVelY_[indx3] ] )
			wz = np.array( [ _parent_nodeVelZ_[indx1], _parent_nodeVelZ_[indx2], _parent_nodeVelZ_[indx3] ] )
			
			intrpx = RegularGridInterpolator( (x, y, z), ux )
			intrpy = RegularGridInterpolator( (x, y, z), vy )
			intrpz = RegularGridInterpolator( (x, y, z), wz )
			
			_nodeC = np.array( [ _child_nodeXC[itr], _child_nodeYC[itr], _child_nodeZC[itr] ] )
			
			_child_nodeVel_[itr,0] = intrpx( _nodeC  )
			_child_nodeVel_[itr,1] = intrpy( _nodeC  )
			_child_nodeVel_[itr,2] = intrpz( _nodeC  )
		
		return _child_nodeVel_
		
		
def _oneTriangle_Area_(p1_C, p2_C, p3_C):
	
	a = np.sqrt( (p1_C[0]-p2_C[0])**2. + (p1_C[1]-p2_C[1])**2. + (p1_C[2]-p2_C[2])**2. )
	b = np.sqrt( (p2_C[0]-p3_C[0])**2. + (p2_C[1]-p3_C[1])**2. + (p2_C[2]-p3_C[2])**2. )
	c = np.sqrt( (p3_C[0]-p1_C[0])**2. + (p3_C[1]-p1_C[1])**2. + (p3_C[2]-p1_C[2])**2. )
	
	s = (a+b+c)/2.
	
	_Area_ = np.sqrt( s*(s-a)*(s-b)*(s-c) )
	
	return _Area_
		
#	def _merging_nodes_(self, )
	
	
	
#	def _each_node_unit_normal_(self, _node_share_cmn_eles_, _leng_):
#	
#		Obj_ = _cal_Normal_Vector( self._nodeXC, self._nodeYC, self._nodeZC, self._no_tri_, self.tri_verti_ ) 
#		
#		# get the normal vectors of all the elements
#		_ele_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
#		
#		_triXC, _triYC, _triZC = _tri_coord_( self._nodeXC, self._nodeYC, self._nodeZC, self._no_tri_, self.tri_verti_ )
#		
#		_Centroid_ = np.zeros( (self._no_tri_,3) )	
#			
#		for it in range(self._no_tri_):
#
#			p1_C = np.array( [_triXC[it,0], _triYC[it,0], _triZC[it,0]] )
#			p2_C = np.array( [_triXC[it,1], _triYC[it,1], _triZC[it,1]] )		
#			p3_C = np.array( [_triXC[it,2], _triYC[it,2], _triZC[it,2]] )  
#			
#			_Centroid_[it,0:3] = _Centroid_triangle_( p1_C, p2_C, p3_C ) 
#			 
#		_tot_nodes = len(self._nodeXC)
#		_node_unit_normal_vec_ = np.zeros( (_tot_nodes,3) )
#				
#		for _node in range(_tot_nodes):
#					
#			_node_Coord = np.array( [self._triXC[_node], self._triYC[_node], self._triZC[_node]] )
#			
#			if _leng_[_node] == 1:				
#				_node_unit_normal_vec_[_node,0:3] = _ele_unit_normal_vec_[_leng_[_node],0:3]
#			else:
#				_Centroid_Coord = np.zeros( (_leng_[_node], 3) )
#				_Centroid_unit_vec_ = np.zeros( (_leng_[_node], 3) )
#				
#				for _len in range(_leng_[_node]):				
#					#print('leng = ', _leng_[_node])
#					
#					_Centroid_Coord[_len, 0:3] = _Centroid_[_node_share_cmn_eles_[_node,_len], 0:3]
#					_Centroid_unit_vec_[_len, 0:3] = _ele_unit_normal_vec_[_node_share_cmn_eles_[_node,_len], 0:3]
#					
#					#_Centroid_node_dist[_len] = _euclid_p2p_3d_dist(_Centroid_, _node_Coord)				
#								
#				rbfi = Rbf( _Centroid_Coord[:,0], _Centroid_Coord[:,1], _Centroid_Coord[:,2], _Centroid_unit_vec_[:,0])   
#				_node_unit_normal_vec_[_node,0] = rbfi(_node_Coord[0], _node_Coord[1], _node_Coord[2])
#				
#				rbfi = Rbf( _Centroid_Coord[:,0], _Centroid_Coord[:,1], _Centroid_Coord[:,2], _Centroid_unit_vec_[:,1])   
#				_node_unit_normal_vec_[_node,1] = rbfi(_node_Coord[0], _node_Coord[1], _node_Coord[2])
#				
#				rbfi = Rbf( _Centroid_Coord[:,0], _Centroid_Coord[:,1], _Centroid_Coord[:,2], _Centroid_unit_vec_[:,2])   
#				_node_unit_normal_vec_[_node,2] = rbfi(_node_Coord[0], _node_Coord[1], _node_Coord[2])
#				
#		return _node_unit_normal_vec_
	
	
	
	# calling this when triangles adding required
#def _adding_tri_elements_(_triXC, _triYC, _triZC, _no_tri_, _adding_tri_ele_):
#	
#	_no_child_node = _length_
#	
#	if _length_ > 0:
#		
#		_no_child_tri = _length_
#		
#		_child_triXC = np.zeros( (_length_,3,3) ) # 2nd gives the three trinagles and 3rd gives their X-coord
#		_child_triYC = np.zeros( (_length_,3,3) )
#		_child_triZC = np.zeros( (_length_,3,3) )
#		
#		for itr in range(_length_):
#			_tri_ele_ = _patrent_tri_sorted_[itr]
#			
#			p1_C = np.array( _TriVert_p1C[_tri_ele_,0], _TriVert_p1C[_tri_ele_,1], _TriVert_p1C[_tri_ele_,2] )
#			p2_C = np.array( _TriVert_p2C[_tri_ele_,0], _TriVert_p2C[_tri_ele_,1], _TriVert_p2C[_tri_ele_,2] )
#			p3_C = np.array( _TriVert_p3C[_tri_ele_,0], _TriVert_p3C[_tri_ele_,1], _TriVert_p3C[_tri_ele_,2] )
#			
#			_centroid_ = _Centroid_triangle_(p1_C, p2_C, p3_C)
#			
#			_child_nodeXC.append( _centroid_[0] ) 
#			_child_nodeYC.append( _centroid_[1] )
#			_child_nodeZC.append( _centroid_[2] )
#			
#			_child_triXC[itr,0,0:3] = np.array( [ _TriVert_p1C[_tri_ele_,0], _TriVert_p2C[_tri_ele_,0], _centroid_[0] ] )
#			_child_triYC[itr,0,0:3] = np.array( [ _TriVert_p1C[_tri_ele_,1], _TriVert_p2C[_tri_ele_,1], _centroid_[1] ] )
#			_child_triZC[itr,0,0:3] = np.array( [ _TriVert_p1C[_tri_ele_,2], _TriVert_p2C[_tri_ele_,2], _centroid_[2] ] )
#			
#			_child_triXC[itr,1,0:3] = np.array( [ _TriVert_p2C[_tri_ele_,0], _TriVert_p3C[_tri_ele_,0], _centroid_[0] ] )
#			_child_triYC[itr,1,0:3] = np.array( [ _TriVert_p2C[_tri_ele_,1], _TriVert_p3C[_tri_ele_,1], _centroid_[1] ] )
#			_child_triZC[itr,1,0:3] = np.array( [ _TriVert_p2C[_tri_ele_,2], _TriVert_p3C[_tri_ele_,2], _centroid_[2] ] )
#			
#			_child_triXC[itr,2,0:3] = np.array( [ _TriVert_p3C[_tri_ele_,0], _TriVert_p1C[_tri_ele_,0], _centroid_[0] ] )
#			_child_triYC[itr,2,0:3] = np.array( [ _TriVert_p3C[_tri_ele_,1], _TriVert_p1C[_tri_ele_,1], _centroid_[1] ] )
#			_child_triZC[itr,2,0:3] = np.array( [ _TriVert_p3C[_tri_ele_,2], _TriVert_p1C[_tri_ele_,2], _centroid_[2] ] )
#			
#	return _no_child_tri, _no_child_node, _child_nodeXC, _child_nodeYC, _child_nodeZC, _child_triXC, _child_triYC, _child_triZC
	
	
	
	
	
#def _creating_child_triangles0_(_parent_triXC, _parent_triYC, _parent_triZC, _parent_tri_verti_, _adding_tri_ele_):
#	
#	_nodeXC, _nodeYC, _nodeZC = \
#	_creating_child_nodes_(_parent_triXC, _parent_triYC, _parent_triZC, _adding_tri_ele_)
#	
#	points2D = np.vstack([_nodeXC, _nodeYC]).T
#	tri = Delaunay(points2D)
#	tri_vertices_ = tri.simplices  # vertices of each triangle
#	_no_triangle_ = np.shape(tri_vertices_)[0]
#	
#	return _no_triangle_, tri_vertices_
	
	
#def _creating_child_triangles_(_parent_triXC, _parent_triYC, _parent_triZC, _parent_tri_verti_, _adding_tri_ele_, len_parent_nodeXC):
#	
#	_no_child_tri_to_add = len(_adding_tri_ele_)
#	
#	_child_nodeXC, _child_nodeYC, _child_nodeZC = \
#	_creating_child_nodes_(_parent_triXC, _parent_triYC, _parent_triZC, _adding_tri_ele_)
#	
#	( _triXC0, _triXC1, _triXC2 ) = ( [], [], [] )
#	( _triYC0, _triYC1, _triYC2 ) = ( [], [], [] )
#	( _triZC0, _triZC1, _triZC2 ) = ( [], [], [] )
#	( _triVr0, _triVr1, _triVr2 ) = ( [], [], [] )
#	
#	for itr in range( len(_adding_tri_ele_) ):
#		_tri_ele = _adding_tri_ele_[itr]
#		
#		_triXC0.append( _parent_triXC[_tri_ele,0] )
#		_triXC1.append( _parent_triXC[_tri_ele,1] )
#		_triXC2.append( _child_nodeXC[itr] )
#		
#		_triYC0.append( _parent_triYC[_tri_ele,0] )
#		_triYC1.append( _parent_triYC[_tri_ele,1] )
#		_triYC2.append( _child_nodeYC[itr] )
#		
#		_triZC0.append( _parent_triZC[_tri_ele,0] )
#		_triZC1.append( _parent_triZC[_tri_ele,1] )
#		_triZC2.append( _child_nodeZC[itr] )
#		
#		_triVr0.append( _parent_tri_verti_[_tri_ele,0] )
#		_triVr1.append( _parent_tri_verti_[_tri_ele,1] )
#		_triVr2.append( len_parent_nodeXC + itr )
#####	
#		_triXC0.append( _parent_triXC[_tri_ele,1] )
#		_triXC1.append( _parent_triXC[_tri_ele,2] )
#		_triXC2.append( _child_nodeXC[itr] )
#		
#		_triYC0.append( _parent_triYC[_tri_ele,1] )
#		_triYC1.append( _parent_triYC[_tri_ele,2] )
#		_triYC2.append( _child_nodeYC[itr] )
#		
#		_triZC0.append( _parent_triZC[_tri_ele,1] )
#		_triZC1.append( _parent_triZC[_tri_ele,2] )
#		_triZC2.append( _child_nodeZC[itr] )
#		
#		_triVr0.append( _parent_tri_verti_[_tri_ele,1] )
#		_triVr1.append( _parent_tri_verti_[_tri_ele,2] )
#		_triVr2.append( len_parent_nodeXC + itr )
#	
#	_len_ = len(_triXC0)
#	print('_len_ = ', _len_, ' _len_triVr = ', len(_triVr0))
#	
#	_child_triXC = np.zeros( (_len_,3) )
#	_child_triYC = np.zeros( (_len_,3) )
#	_child_triZC = np.zeros( (_len_,3) )
#	_child_triVr = np.zeros( (_len_,3) )
#	
#	for itr in range(_len_):
#		
#		( _child_triXC[itr,0], _child_triXC[itr,1], _child_triXC[itr,2] ) = ( _triXC0[itr], _triXC1[itr], _triXC2[itr] )
#		( _child_triYC[itr,0], _child_triYC[itr,1], _child_triYC[itr,2] ) = ( _triYC0[itr], _triYC1[itr], _triYC2[itr] )
#		( _child_triZC[itr,0], _child_triZC[itr,1], _child_triZC[itr,2] ) = ( _triZC0[itr], _triZC1[itr], _triZC2[itr] )
#		( _child_triVr[itr,0], _child_triVr[itr,1], _child_triVr[itr,2] ) = ( _triVr0[itr], _triVr1[itr], _triVr2[itr] )
#		
#		
#	for itr in range( len(_adding_tri_ele_) ):
#		_tri_ele = _adding_tri_ele_[itr]
#		
#		( _parent_triXC[_tri_ele,1], _parent_triYC[_tri_ele,1], _parent_triZC[_tri_ele,1] ) = \
#		( _child_nodeXC[itr], _child_nodeYC[itr], _child_nodeZC[itr] )
#		
#		_parent_tri_verti_[_tri_ele,1] = len_parent_nodeXC + itr 
#		
#	_parent_triXC = np.concatenate( (_parent_triXC, _child_triXC) )
#	_parent_triYC = np.concatenate( (_parent_triYC, _child_triYC) )
#	_parent_triZC = np.concatenate( (_parent_triZC, _child_triZC) )
#	_parent_tri_verti_ = np.concatenate( (_parent_tri_verti_, _child_triVr) )
#	
#	
#	return _parent_triXC, _parent_triYC, _parent_triZC, _parent_tri_verti_
#	
	
	
#def _creating_child_triangles_(_parent_triXC, _parent_triYC, _parent_triZC, _parent_tri_verti_, _adding_tri_ele_, len_parent_nodeXC):
#	
#	_no_child_tri_to_add = len(_adding_tri_ele_)
#	
#	_child_nodeXC, _child_nodeYC, _child_nodeZC = \
#	_creating_child_nodes_(_parent_triXC, _parent_triYC, _parent_triZC, _adding_tri_ele_)
#	
#	( _triXC0, _triXC1, _triXC2 ) = ( [], [], [] )
#	( _triYC0, _triYC1, _triYC2 ) = ( [], [], [] )
#	( _triZC0, _triZC1, _triZC2 ) = ( [], [], [] )
#	( _triVr0, _triVr1, _triVr2 ) = ( [], [], [] )
#	
#	#p = -1
#	for itr in range( len(_adding_tri_ele_) ):
#		_tri_ele = _adding_tri_ele_[itr]
#		
#		_triXC0.append( _parent_triXC[_tri_ele,0] )
#		_triXC1.append( _parent_triXC[_tri_ele,1] )
#		_triXC2.append( _child_nodeXC[itr] )
#		
#		_triYC0.append( _parent_triYC[_tri_ele,0] )
#		_triYC1.append( _parent_triYC[_tri_ele,1] )
#		_triYC2.append( _child_nodeYC[itr] )
#		
#		_triZC0.append( _parent_triZC[_tri_ele,0] )
#		_triZC1.append( _parent_triZC[_tri_ele,1] )
#		_triZC2.append( _child_nodeZC[itr] )
#		
#		#p += 1 
#		_triVr0.append( _parent_tri_verti_[_tri_ele,0] )
#		_triVr1.append( _parent_tri_verti_[_tri_ele,1] )
#		_triVr2.append( len_parent_nodeXC + itr )
#####	
#		_triXC0.append( _parent_triXC[_tri_ele,1] )
#		_triXC1.append( _parent_triXC[_tri_ele,2] )
#		_triXC2.append( _child_nodeXC[itr] )
#		
#		_triYC0.append( _parent_triYC[_tri_ele,1] )
#		_triYC1.append( _parent_triYC[_tri_ele,2] )
#		_triYC2.append( _child_nodeYC[itr] )
#		
#		_triZC0.append( _parent_triZC[_tri_ele,1] )
#		_triZC1.append( _parent_triZC[_tri_ele,2] )
#		_triZC2.append( _child_nodeZC[itr] )
#		
#		#p += 1 
#		_triVr0.append( _parent_tri_verti_[_tri_ele,1] )
#		_triVr1.append( _parent_tri_verti_[_tri_ele,2] )
#		_triVr2.append( len_parent_nodeXC + itr )
#	
#	_len_ = len(_triXC0)
#	print('_len_ = ', _len_, ' _len_triVr = ', len(_triVr0))
#	
#	_child_triXC = np.zeros( (_len_,3) )
#	_child_triYC = np.zeros( (_len_,3) )
#	_child_triZC = np.zeros( (_len_,3) )
#	_child_triVr = np.zeros( (_len_,3) )
#	
#	for itr in range(_len_):
#		
#		( _child_triXC[itr,0], _child_triXC[itr,1], _child_triXC[itr,2] ) = ( _triXC0[itr], _triXC1[itr], _triXC2[itr] )
#		( _child_triYC[itr,0], _child_triYC[itr,1], _child_triYC[itr,2] ) = ( _triYC0[itr], _triYC1[itr], _triYC2[itr] )
#		( _child_triZC[itr,0], _child_triZC[itr,1], _child_triZC[itr,2] ) = ( _triZC0[itr], _triZC1[itr], _triZC2[itr] )
#		( _child_triVr[itr,0], _child_triVr[itr,1], _child_triVr[itr,2] ) = ( _triVr0[itr], _triVr1[itr], _triVr2[itr] )
#		
#		
#	for itr in range( len(_adding_tri_ele_) ):
#		_tri_ele = _adding_tri_ele_[itr]
#		
#		( _parent_triXC[_tri_ele,1], _parent_triYC[_tri_ele,1], _parent_triZC[_tri_ele,1] ) = \
#		( _child_nodeXC[itr], _child_nodeYC[itr], _child_nodeZC[itr] )
#		
#		_parent_tri_verti_[_tri_ele,1] = len_parent_nodeXC + itr 
#		
#	_parent_triXC = np.concatenate( (_parent_triXC, _child_triXC) )
#	_parent_triYC = np.concatenate( (_parent_triYC, _child_triYC) )
#	_parent_triZC = np.concatenate( (_parent_triZC, _child_triZC) )
#	_parent_tri_verti_ = np.concatenate( (_parent_tri_verti_, _child_triVr) )
#	
#	_parent_tri_verti_ = np.concatenate( (_parent_tri_verti_, _child_tri_verti_) )
#	
#	return _parent_triXC, _parent_triYC, _parent_triZC, _parent_tri_verti_
#		
	
