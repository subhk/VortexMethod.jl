import numpy as np
import scipy.io as sio
import matplotlib.tri as mtri
from scipy.interpolate import Rbf
from scipy.interpolate import RegularGridInterpolator

#from init_3d import _tri_coord_
from utility_3d_paral import _cal_Normal_Vector

def _euclid_3d_dist(x1, y1, z1, x2, y2, z2):

	_tmp_ = np.sqrt( (x1-x2)**2. + (y1-y2)**2. + (z1-z2)**2. )
	
	return _tmp_
	
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


def _sorting_triangles_( _edge_len_, criteria ):

	_no_tri_ = _edge_len_.shape[0]	
	_tri_sorted_ = []
	
	for _tri in range(_no_tri_):
		flag = any( i >= criteria for i in _edge_len_[itr,0:3] )
		if flag: _tri_sorted_.append( _tri )
		
	return _tri_sorted_ 

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
		
		
	def _tri_edge_len_( self, _TriVert_p1C, _TriVert_p2C, _TriVert_p3C ):
		'''	
		this function store the edge-length of all the elements
		'''	
		
		#_TriVert_p1C, _TriVert_p2C, _TriVert_p3C = self._TriVert_Cord()
		
		_tri_edge_len_ = np.zeros( (self._no_tri_,3) )
		
		_tri_edge_len_[:,0] = \
		_euclid_3d_dist( _TriVert_p1C[:,0],_TriVert_p1C[:,1],_TriVert_p1C[:,2], _TriVert_p2C[:,0],_TriVert_p2C[:,1],_TriVert_p2C[:,2] )
		
		_tri_edge_len_[:,1] = \
		_euclid_3d_dist( _TriVert_p2C[:,0],_TriVert_p2C[:,1],_TriVert_p2C[:,2], _TriVert_p3C[:,0],_TriVert_p3C[:,1],_TriVert_p3C[:,2] )
		
		_tri_edge_len_[:,2] = \
		_euclid_3d_dist( _TriVert_p3C[:,0],_TriVert_p3C[:,1],_TriVert_p3C[:,2], _TriVert_p1C[:,0],_TriVert_p1C[:,1],_TriVert_p1C[:,2] )
		
		return _tri_edge_len_ #_edge_Length_p1_p2_, _edge_Length_p2_p3_, _edge_Length_p3_p1_

	
	def _max_edge_length_dection_(self, _TriVert_p1C, _TriVert_p2C, _TriVert_p3C):
	
		_tri_edge_len_ = self._tri_edge_len_( _TriVert_p1C, _TriVert_p2C, _TriVert_p3C )
		
		_tri_sorted_ = _sorting_elements_( _edge_Length_p1_p2_, self.del_split)

		return _tri_sorted_
	
	
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
			
				
	
	def _each_node_unit_normal_(self, _node_share_cmn_eles_, _leng_):
	
		Obj_ = _cal_Normal_Vector( self._nodeXC, self._nodeYC, self._nodeZC, self._no_tri_, self.tri_verti_ ) 
		
		# get the normal vectors of all the elements
		_ele_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
		
		_triXC, _triYC, _triZC = _tri_coord_( self._nodeXC, self._nodeYC, self._nodeZC, self._no_tri_, self.tri_verti_ )
		
		_Centroid_ = np.zeros( (self._no_tri_,3) )	
			
		for it in range(self._no_tri_):

			p1_C = np.array( [_triXC[it,0], _triYC[it,0], _triZC[it,0]] )
			p2_C = np.array( [_triXC[it,1], _triYC[it,1], _triZC[it,1]] )		
			p3_C = np.array( [_triXC[it,2], _triYC[it,2], _triZC[it,2]] )  
			
			_Centroid_[it,0:3] = _Centroid_triangle_( p1_C, p2_C, p3_C ) 
			 
		_tot_nodes = len(self._nodeXC)
		_node_unit_normal_vec_ = np.zeros( (_tot_nodes,3) )
				
		for _node in range(_tot_nodes):
					
			_node_Coord = np.array( [self._triXC[_node], self._triYC[_node], self._triZC[_node]] )
			
			if _leng_[_node] == 1:				
				_node_unit_normal_vec_[_node,0:3] = _ele_unit_normal_vec_[_leng_[_node],0:3]
			else:
				_Centroid_Coord = np.zeros( (_leng_[_node], 3) )
				_Centroid_unit_vec_ = np.zeros( (_leng_[_node], 3) )
				
				for _len in range(_leng_[_node]):				
					#print('leng = ', _leng_[_node])
					
					_Centroid_Coord[_len, 0:3] = _Centroid_[_node_share_cmn_eles_[_node,_len], 0:3]
					_Centroid_unit_vec_[_len, 0:3] = _ele_unit_normal_vec_[_node_share_cmn_eles_[_node,_len], 0:3]
					
					#_Centroid_node_dist[_len] = _euclid_p2p_3d_dist(_Centroid_, _node_Coord)				
								
				rbfi = Rbf( _Centroid_Coord[:,0], _Centroid_Coord[:,1], _Centroid_Coord[:,2], _Centroid_unit_vec_[:,0])   
				_node_unit_normal_vec_[_node,0] = rbfi(_node_Coord[0], _node_Coord[1], _node_Coord[2])
				
				rbfi = Rbf( _Centroid_Coord[:,0], _Centroid_Coord[:,1], _Centroid_Coord[:,2], _Centroid_unit_vec_[:,1])   
				_node_unit_normal_vec_[_node,1] = rbfi(_node_Coord[0], _node_Coord[1], _node_Coord[2])
				
				rbfi = Rbf( _Centroid_Coord[:,0], _Centroid_Coord[:,1], _Centroid_Coord[:,2], _Centroid_unit_vec_[:,2])   
				_node_unit_normal_vec_[_node,2] = rbfi(_node_Coord[0], _node_Coord[1], _node_Coord[2])
				
		return _node_unit_normal_vec_
		

		
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
		
		
	def _remeshing_elements_(self):
		
		_TriVert_p1C, _TriVert_p2C, _TriVert_p3C = self._TriVert_Cord()
		
		_patrent_tri_sorted_ = _max_edge_length_dection_( _TriVert_p1C, _TriVert_p2C, _TriVert_p3C )
		_length_ = len(_patrent_tri_sorted_)
		
		_triXC, _triYC, _triZC = _tri_coord_( self._nodeXC, self._nodeYC, self._nodeZC, self._no_tri_, self.tri_verti_ )
		
		_child_nodeXC = []
		_child_nodeYC = []
		_child_nodeZC = []
		
		_no_child_node = _length_
		
		if _length_ > 0:
			
			_no_child_tri = _length_
			
			_child_triXC = np.zeros( (_length_,3,3) ) # 2nd gives the three trinagles and 3rd gives their X-coord
			_child_triYC = np.zeros( (_length_,3,3) )
			_child_triZC = np.zeros( (_length_,3,3) )
			
			for itr in range(_length_):
				_tri_ele_ = _patrent_tri_sorted_[itr]
				
				p1_C = np.array( _TriVert_p1C[_tri_ele_,0], _TriVert_p1C[_tri_ele_,1], _TriVert_p1C[_tri_ele_,2] )
				p2_C = np.array( _TriVert_p2C[_tri_ele_,0], _TriVert_p2C[_tri_ele_,1], _TriVert_p2C[_tri_ele_,2] )
				p3_C = np.array( _TriVert_p3C[_tri_ele_,0], _TriVert_p3C[_tri_ele_,1], _TriVert_p3C[_tri_ele_,2] )
				
				_centroid_ = _Centroid_triangle_(p1_C, p2_C, p3_C)
				
				_child_nodeXC.append( _centroid_[0] ) 
				_child_nodeYC.append( _centroid_[1] )
				_child_nodeZC.append( _centroid_[2] )
				
				_child_triXC[itr,0,0:3] = np.array( [ _TriVert_p1C[_tri_ele_,0], _TriVert_p2C[_tri_ele_,0], _centroid_[0] ] )
				_child_triYC[itr,0,0:3] = np.array( [ _TriVert_p1C[_tri_ele_,1], _TriVert_p2C[_tri_ele_,1], _centroid_[1] ] )
				_child_triZC[itr,0,0:3] = np.array( [ _TriVert_p1C[_tri_ele_,2], _TriVert_p2C[_tri_ele_,2], _centroid_[2] ] )
				
				_child_triXC[itr,1,0:3] = np.array( [ _TriVert_p2C[_tri_ele_,0], _TriVert_p3C[_tri_ele_,0], _centroid_[0] ] )
				_child_triYC[itr,1,0:3] = np.array( [ _TriVert_p2C[_tri_ele_,1], _TriVert_p3C[_tri_ele_,1], _centroid_[1] ] )
				_child_triZC[itr,1,0:3] = np.array( [ _TriVert_p2C[_tri_ele_,2], _TriVert_p3C[_tri_ele_,2], _centroid_[2] ] )
				
				_child_triXC[itr,2,0:3] = np.array( [ _TriVert_p3C[_tri_ele_,0], _TriVert_p1C[_tri_ele_,0], _centroid_[0] ] )
				_child_triYC[itr,2,0:3] = np.array( [ _TriVert_p3C[_tri_ele_,1], _TriVert_p1C[_tri_ele_,1], _centroid_[1] ] )
				_child_triZC[itr,2,0:3] = np.array( [ _TriVert_p3C[_tri_ele_,2], _TriVert_p1C[_tri_ele_,2], _centroid_[2] ] )
				
		return _no_child_tri, _no_child_node, _child_nodeXC, _child_nodeYC, _child_nodeZC, _child_triXC, _child_triYC, _child_triZC
		
		
	def _assign_child_vorticity_( self, _parent_tri_sorted_, _parent_eleGma_ ):
		
		_length_ = len(_patrent_tri_sorted_)
		_child_eleGma_ = np.zeros( (_length_,3) )
		
		for itr in range(_length_):	
			_tri = _parent_tri_sorted_[itr]
			_child_eleGma_[itr,0:3] = _parent_eleGma_[_tri,0:3]
			
		return _child_eleGma_
		
		
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
		
		
#	def _merging_nodes_(self, )
		
