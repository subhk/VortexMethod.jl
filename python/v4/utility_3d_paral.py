"""
utility function
"""

import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

#from remeshing_3d import _elements_splitting_
		
def _left_edge_BndryPts_(_input_, _imax_):

	gr_pts = len(_input_)
	Cnt = 0
	_LeftEdge_BndryPts_ = []
	for itr in range(gr_pts):
		if itr == Cnt*_imax_:
			_LeftEdge_BndryPts_.append(itr)
			Cnt += 1
	
	return _LeftEdge_BndryPts_
	
	
def _Enforce_Predic_Cond_on_Nodes_(_input_, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up, Xcnst, Ycnst):
	
	if len(_bndry_node_lt) != len(_bndry_node_rt): raise ValueError('No of nodes on left and right boundary should be same')
	if len(_bndry_node_dn) != len(_bndry_node_up): raise ValueError('No of nodes on lower and upper boundary should be same')
	
	for _tri in range( len(_bndry_node_lt) ):
		_input_[_bndry_node_lt[_tri]] = _input_[_bndry_node_rt[_tri]] - Xcnst
	
	for _tri in range( len(_bndry_node_dn) ):
		_input_[_bndry_node_dn[_tri]] = _input_[_bndry_node_up[_tri]] - Ycnst
	
	return _input_
	

def _Enforce_Predic_Cond_on_Tri(_input_, _bndry_ele_lt, _bndry_ele_rt, _bndry_ele_dn, _bndry_ele_up, Xcnst, Ycnst):
	
	if len(_bndry_ele_lt) != len(_bndry_ele_rt): raise ValueError('No of elements on left and right boundary should be same')
	if len(_bndry_ele_dn) != len(_bndry_ele_up): raise ValueError('No of elements on lower and upper boundary should be same')
		
	for _tri in range( len(_bndry_ele_lt) ):
		_input_[_bndry_ele_rt[_tri]] = _input_[_bndry_ele_lt[_tri]] + Xcnst
	
	for _tri in range( len(_bndry_ele_up) ):
		_input_[_bndry_ele_up[_tri]] = _input_[_bndry_ele_dn[_tri]]	+ Ycnst
		
	return _input_
	
	
def _Enforce_Predic_Cond_on_Vel_(_uXg_, _vYg_, _wZg_, _node_lt, _node_rt, _node_dn, _node_up):
	
	_Enforce_Predic_Cond_on_Nodes_(_uXg_, _node_lt, _node_rt, _node_dn, _node_up, 0., 0.)
	_Enforce_Predic_Cond_on_Nodes_(_vYg_, _node_lt, _node_rt, _node_dn, _node_up, 0., 0.)
	_Enforce_Predic_Cond_on_Nodes_(_wZg_, _node_lt, _node_rt, _node_dn, _node_up, 0., 0.)
	
	return _uXg_, _vYg_, _wZg_
	
	
def _Enforce_Predic_Cond_on_Gma_(_eleGmaX_, _eleGmaY_, _eleGmaZ_, _ele_lt, _ele_rt, _ele_dn, _ele_up):
	
	_Enforce_Predic_Cond_on_Tri(_eleGmaX_, _ele_lt, _ele_rt, _ele_dn, _ele_up, 0., 0.)
	_Enforce_Predic_Cond_on_Tri(_eleGmaY_, _ele_lt, _ele_rt, _ele_dn, _ele_up, 0., 0.)
	_Enforce_Predic_Cond_on_Tri(_eleGmaZ_, _ele_lt, _ele_rt, _ele_dn, _ele_up, 0., 0.)
	
	return _eleGmaX_, _eleGmaY_, _eleGmaZ_
	
	
def _Enforce_Predic_Cond_on_nodeCoord_(_nodeXC, _nodeYC, _nodeZC, _node_lt, _node_rt, _node_dn, _node_up, Lx, Ly):
	
	_Enforce_Predic_Cond_on_Nodes_(_nodeXC, _node_lt, _node_rt, _node_dn, _node_up, Lx, 0.)
	_Enforce_Predic_Cond_on_Nodes_(_nodeYC, _node_lt, _node_rt, _node_dn, _node_up, 0., Ly)
	_Enforce_Predic_Cond_on_Nodes_(_nodeZC, _node_lt, _node_rt, _node_dn, _node_up, 0., 0.)
	
	return _nodeXC, _nodeYC, _nodeZC
	
	
def _euclid_3d_dist(x1, y1, z1, x2, y2, z2):

	_tmp_ = np.sqrt( (x1-x2)**2. + (y1-y2)**2. + (z1-z2)**2. )
	
	return _tmp_
	
	
def _centroid_triangle_( p1_C, p2_C, p3_C ):

	_centroid_ = np.zeros(3)
	_centroid_[0] = (p1_C[0]+p2_C[0]+p3_C[0])/3.
	_centroid_[1] = (p1_C[1]+p2_C[1]+p3_C[1])/3.
	_centroid_[2] = (p1_C[2]+p2_C[2]+p3_C[2])/3.
		
	return _centroid_
	
def _allTriangle_Centroid_( _triXC, _triYC, _triZC ):

	_centroid_ = np.zeros( (len(_triXC[:,0]),3) )
	
	for it in range( len(_triXC[:,0]) ):
		
		p1_C = np.array( [_triXC[it,0], _triYC[it,0], _triZC[it,0]] )
		p2_C = np.array( [_triXC[it,1], _triYC[it,1], _triZC[it,1]] )
		p3_C = np.array( [_triXC[it,2], _triYC[it,2], _triZC[it,2]] )  
		
		_centroid_[it,0] = (p1_C[0]+p2_C[0]+p3_C[0])/3.
		_centroid_[it,1] = (p1_C[1]+p2_C[1]+p3_C[1])/3.
		_centroid_[it,2] = (p1_C[2]+p2_C[2]+p3_C[2])/3.
	
	return _centroid_
	
	
def _oneTriangle_Area_(p1_C, p2_C, p3_C):
	
	a = np.sqrt( (p1_C[0]-p2_C[0])**2. + (p1_C[1]-p2_C[1])**2. + (p1_C[2]-p2_C[2])**2. )
	b = np.sqrt( (p2_C[0]-p3_C[0])**2. + (p2_C[1]-p3_C[1])**2. + (p2_C[2]-p3_C[2])**2. )
	c = np.sqrt( (p3_C[0]-p1_C[0])**2. + (p3_C[1]-p1_C[1])**2. + (p3_C[2]-p1_C[2])**2. )	
	
	s = (a+b+c)/2.
	
	_Area_ = np.sqrt(s*(s-a)*(s-b)*(s-c))
	
	return _Area_
	

def _allTriangle_Area_( _triXC, _triYC, _triZC ):

	_Area_ = np.zeros( len(_triXC[:,0]) )
	
	for it in range( len(_triXC[:,0]) ):
		
		p1_C = np.array( [_triXC[it,0], _triYC[it,0], _triZC[it,0]] )
		p2_C = np.array( [_triXC[it,1], _triYC[it,1], _triZC[it,1]] )
		p3_C = np.array( [_triXC[it,2], _triYC[it,2], _triZC[it,2]] )
		
		a = np.sqrt( (p1_C[0]-p2_C[0])**2. + (p1_C[1]-p2_C[1])**2. + (p1_C[2]-p2_C[2])**2. )
		b = np.sqrt( (p2_C[0]-p3_C[0])**2. + (p2_C[1]-p3_C[1])**2. + (p2_C[2]-p3_C[2])**2. )
		c = np.sqrt( (p3_C[0]-p1_C[0])**2. + (p3_C[1]-p1_C[1])**2. + (p3_C[2]-p1_C[2])**2. )
		
		s = (a+b+c)/2.
		
		_Area_[it] = np.sqrt( s*(s-a)*(s-b)*(s-c) )
	
	return _Area_
	
	
def _D1_Peskin_function_( x, xp, dx, eps):
	
	if abs(x-xp)/dx <= eps:		
		delta = ( 1. + np.cos( np.pi*(x-xp)/(dx*eps) ) )/(2.*eps)
	else:
		delta = 0.
	
	return delta
		

#def _D3_Peskin_function_( x, y, z, xp, yp, zp, eps ):
#			
#	delta_x = _D1_Peskin_function_( x, xp, eps )
#	delta_y = _D1_Peskin_function_( y, yp, eps )
#	delta_z = _D1_Peskin_function_( x, zp, eps )
#	
#	return delta_x*delta_y*delta_z
	

class _cal_Normal_Vector(object):

	def __init__(self, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _no_tri_, tri_verti_):
		
		self._nodeXC = _nodeXC
		self._nodeYC = _nodeYC
		self._nodeZC = _nodeZC
		
		self._triXC = _triXC
		self._triYC = _triYC
		self._triZC = _triZC
		
		self._no_tri_ = _no_tri_
		self.tri_verti_ = tri_verti_
	
	
	def _all_triangle_surf_normal_(self):

		_unit_normal_vec_ = np.zeros( (self._no_tri_,3) )
		
#		_triXC, _triYC, _triZC = self._tri_coord_()
#		#sio.savemat('xT.mat', {'xT':xT})
#		#sio.savemat('yT.mat', {'yT':yT})
#		#sio.savemat('zT.mat', {'zT':zT})
		
		for it in range(self._no_tri_):
		
			p0 = np.array( [ self._triXC[it,0], self._triYC[it,0], self._triZC[it,0] ] )
			p1 = np.array( [ self._triXC[it,1], self._triYC[it,1], self._triZC[it,1] ] )
			p2 = np.array( [ self._triXC[it,2], self._triYC[it,2], self._triZC[it,2] ] )
			
			det = p0[0]*( p1[1]*p2[2] - p2[1]*p1[2] ) - p1[0]*( p0[1]*p2[2] - p2[1]*p0[2] ) + p2[0]*( p0[1]*p1[2] - p1[1]*p0[2] )
			
			#if det > 0:
			r01 = np.array( [ p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] ] )
			r12 = np.array( [ p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] ] )
			
			r01xr12 = np.array( [ r01[1]*r12[2] - r01[2]*r12[1], r12[0]*r01[2] - r12[2]*r01[0], r01[0]*r12[1] - r01[1]*r12[0] ] )
			norm = np.linalg.norm(r01xr12)
			if norm == 0.: raise ValueError('zero norm')
			
			_unit_normal_vec_[it,:] = r01xr12/norm
				
			#if det < 0:
			#	r02 = np.array( [ p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] ] )
			#	r21 = np.array( [ p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2] ] )
			#	
			#	r02xr21 = np.array( [ r02[1]*r21[2] - r02[2]*r21[1], r21[0]*r02[2] - r21[2]*r02[0], r02[0]*r21[1] - r02[1]*r21[0] ] )
			#	norm = np.linalg.norm(r02xr21)
			#	if norm == 0.: raise ValueError('zero norm')
				
			#_unit_normal_vec_[it,:] = r02xr21/norm
			
			if _unit_normal_vec_[it,2] < 0: _unit_normal_vec_[it,:] = -1.*_unit_normal_vec_[it,:]
			
#			poly = \
#			np.array([ [ self._triXC[it,0], self._triYC[it,0], self._triZC[it,0] ], \
#			[ self._triXC[it,1], self._triYC[it,1], self._triZC[it,1] ], \
#			[ self._triXC[it,2], self._triYC[it,2], self._triZC[it,2] ] ])
#			
#			n = np.array([0.0, 0.0, 0.0])
#			
#			for i, v_curr in enumerate(poly):
#				v_next = poly[(i+1) % len(poly),:]
#				n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2]) 
#				n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
#				n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])
#			
#			norm = np.linalg.norm(n)
#			
#			if norm==0:
#				print('itr = ', it)
#				print('n = ', n)
#				raise ValueError('zero norm')
#			else:
#				_unit_normal_vec_[it,:] = n/norm
#			
#			#_unit_normal_vec_[it,:] = n[:]
#			#print('normal vec = ', _unit_normal_vec_)
#			
		return _unit_normal_vec_
	
	
#	def _edge_length_(self):
#		
#		_triXC, _triYC, _triZC = self._tri_coord_()
#		
#		len_ = _euclid_3d_dist(_triXC[0,0], _triYC[0,0], _triZC[0,0], _triXC[0,1], _triYC[0,1], _triZC[0,1])	
#		
#		return len_
	
	
def Domain():
	
	( Lx, Ly, Lz ) = ( 1., 1., 1. )
	
	return Lx, Ly, Lz
	
	
def _SmoothGrid_Generation_():
	
	Lx, Ly, Lz = Domain()
	
	# nz = 2*nx-1 if Lz=Lx or, nz=2*(2*nx-1)-1 if Lz=2*Lx  
	nx = 25
	ny = 50
	if Lz == 1.*Lx: nz = 2*ny-1
	if Lz == 2.*Lx: nz = 2*(2*ny-1)-1
	
	x = np.linspace( 0., Lx, nx )
	y = np.linspace( 0., Ly, ny )
	z = np.linspace( -Lz, Lz, nz )
	
	return x, y, z
	

def _Interpolation_on_SmoothGrid_( _nodeXC, _nodeYC, _nodeZC, Vector ):

	Xgrid, Ygrid, Zgrid = _SmoothGrid_Generation_( _nodeXC, _nodeYC, _nodeZC )
	
	rbfi = Rbf( _nodeXC, _nodeYC, _nodeZC, Vector )
	
	func = rbfi( Xgrid, Ygrid, Zgrid )
	
	return func
	

def _Triangle_into_4SubTriangle(p1_C, p2_C, p3_C):
	
	ds2_p1p2 = np.zeros(3)
	ds2_p2p3 = np.zeros(3)
	ds2_p3p1 = np.zeros(3)
	
	# each edge has been divided into half
	for i in range(3): ds2_p1p2[i] = (p2_C[i] - p1_C[i])/2.	
	for i in range(3): ds2_p2p3[i] = (p3_C[i] - p2_C[i])/2.	
	for i in range(3): ds2_p3p1[i] = (p1_C[i] - p3_C[i])/2.	
	
	_triC = np.zeros( (4,3,3) )
	
	# create 1st triangle
	_triC[0,0,0:3] = np.array( [ p1_C[0], p1_C[1], p1_C[2]  ] )
	_triC[0,1,0:3] = np.array( [ p1_C[0]+ds2_p1p2[0], p1_C[1]+ds2_p1p2[1], p1_C[2]+ds2_p1p2[2]  ] )
	_triC[0,2,0:3] = np.array( [ p1_C[0]-ds2_p3p1[0], p1_C[1]-ds2_p3p1[1], p1_C[2]-ds2_p3p1[2]  ] )
	
	# create 2nd triangle
	_triC[1,0,0:3] = _triC[0,1,0:3]
	_triC[1,1,0:3] = np.array( [ p2_C[0]+ds2_p2p3[0], p2_C[1]+ds2_p2p3[1], p2_C[2]+ds2_p2p3[2]  ] )
	_triC[1,2,0:3] = _triC[0,2,0:3]
	
	# create 3rd triangle
	_triC[2,0,0:3] = np.array( [ p2_C[0], p2_C[1], p2_C[2]  ] )
	_triC[2,1,0:3] = _triC[1,1,0:3]
	_triC[2,2,0:3] = _triC[1,0,0:3]
	
	# create 4th triangle
	_triC[3,0,0:3] = _triC[1,1,0:3]
	_triC[3,1,0:3] = np.array( [ p3_C[0], p3_C[1], p3_C[2]  ] )
	_triC[3,2,0:3] = _triC[1,2,0:3]
	
	return _triC
	
	
def _test_():

	p1_C = np.array( [0.,0.,0.] )
	p2_C = np.array( [1.,1.,1.] )
	p3_C = np.array( [1.,1.,0.] )
	
	_triC = _Triangle_into_4SubTriangle(p1_C, p2_C, p3_C)
	#print( '4 tri = ', _triC)
	
	_triC1, _, _, _ = _Triangle_into_16SubTriangle(p1_C, p2_C, p3_C)
	#print( '16 tri = ', _triC1)
	
	#_centroid_ = _centroid_4subTriangle_(p1_C, p2_C, p3_C)
	#print(_centroid_)
	
	return _triC


def _Triangle_into_16SubTriangle(p1_C, p2_C, p3_C):
	
	_triC = _Triangle_into_4SubTriangle(p1_C, p2_C, p3_C)	 
	
	# 1st Triangle has been divided into 4 sub triangles
	p11_C = np.array( [ _triC[0,0,0], _triC[0,0,1], _triC[0,0,2] ] )
	p21_C = np.array( [ _triC[0,1,0], _triC[0,1,1], _triC[0,1,2] ] )
	p31_C = np.array( [ _triC[0,2,0], _triC[0,2,1], _triC[0,2,2] ] )
	_triC1 = _Triangle_into_4SubTriangle( p11_C, p21_C, p31_C )

	# 2nd Triangle has been divided into 4 sub triangles
	p11_C = np.array( [ _triC[1,0,0], _triC[1,0,1], _triC[1,0,2] ] )
	p21_C = np.array( [ _triC[1,1,0], _triC[1,1,1], _triC[1,1,2] ] )
	p31_C = np.array( [ _triC[1,2,0], _triC[1,2,1], _triC[1,2,2] ] )
	_triC2 = _Triangle_into_4SubTriangle( p11_C, p21_C, p31_C )
	
	# 3rd Triangle has been divided into 4 sub triangles
	p11_C = np.array( [ _triC[2,0,0], _triC[2,0,1], _triC[2,0,2] ] )
	p21_C = np.array( [ _triC[2,1,0], _triC[2,1,1], _triC[2,1,2] ] )
	p31_C = np.array( [ _triC[2,2,0], _triC[2,2,1], _triC[2,2,2] ] )
	_triC3 = _Triangle_into_4SubTriangle( p11_C, p21_C, p31_C )
	
	# 4th Triangle has been divided into 4 sub triangles
	p11_C = np.array( [ _triC[3,0,0], _triC[3,0,1], _triC[3,0,2] ] )
	p21_C = np.array( [ _triC[3,1,0], _triC[3,1,1], _triC[3,1,2] ] )
	p31_C = np.array( [ _triC[3,2,0], _triC[3,2,1], _triC[3,2,2] ] )
	_triC4 = _Triangle_into_4SubTriangle( p11_C, p21_C, p31_C )
	
	return _triC1, _triC2, _triC3, _triC4 
	

def _centroid_16subTriangle_(p1_C, p2_C, p3_C):
	
	_sub_triC1, _sub_triC2, _sub_triC3, _sub_triC4 = _Triangle_into_16SubTriangle(p1_C, p2_C, p3_C)
	
	_centroid_sub_tri = np.zeros( (16,3) ) 
	
	for i in range(4):
		sub_tri_p1C = np.array( [ _sub_triC1[i,0,0], _sub_triC1[i,0,1], _sub_triC1[i,0,2] ] )
		sub_tri_p2C = np.array( [ _sub_triC1[i,1,0], _sub_triC1[i,1,1], _sub_triC1[i,1,2] ] )
		sub_tri_p3C = np.array( [ _sub_triC1[i,2,0], _sub_triC1[i,2,1], _sub_triC1[i,2,2] ] )
		
		_centroid_sub_tri[i,0:3] = _centroid_triangle_( sub_tri_p1C, sub_tri_p2C, sub_tri_p3C  )

	for i in range(4):
		sub_tri_p1C = np.array( [ _sub_triC2[i,0,0], _sub_triC2[i,0,1], _sub_triC2[i,0,2] ] )
		sub_tri_p2C = np.array( [ _sub_triC2[i,1,0], _sub_triC2[i,1,1], _sub_triC2[i,1,2] ] )
		sub_tri_p3C = np.array( [ _sub_triC2[i,2,0], _sub_triC2[i,2,1], _sub_triC2[i,2,2] ] )
		
		_centroid_sub_tri[4+i,0:3] = _centroid_triangle_( sub_tri_p1C, sub_tri_p2C, sub_tri_p3C  )

	for i in range(4):
		sub_tri_p1C = np.array( [ _sub_triC3[i,0,0], _sub_triC3[i,0,1], _sub_triC3[i,0,2] ] )
		sub_tri_p2C = np.array( [ _sub_triC3[i,1,0], _sub_triC3[i,1,1], _sub_triC3[i,1,2] ] )
		sub_tri_p3C = np.array( [ _sub_triC3[i,2,0], _sub_triC3[i,2,1], _sub_triC3[i,2,2] ] )
		
		_centroid_sub_tri[8+i,0:3] = _centroid_triangle_( sub_tri_p1C, sub_tri_p2C, sub_tri_p3C  )
	
	for i in range(4):
		sub_tri_p1C = np.array( [ _sub_triC4[i,0,0], _sub_triC4[i,0,1], _sub_triC4[i,0,2] ] )
		sub_tri_p2C = np.array( [ _sub_triC4[i,1,0], _sub_triC4[i,1,1], _sub_triC4[i,1,2] ] )
		sub_tri_p3C = np.array( [ _sub_triC4[i,2,0], _sub_triC4[i,2,1], _sub_triC4[i,2,2] ] )
		
		_centroid_sub_tri[12+i,0:3] = _centroid_triangle_( sub_tri_p1C, sub_tri_p2C, sub_tri_p3C  )
	
	return _centroid_sub_tri
	
	
def _centroid_4subTriangle_(p1_C, p2_C, p3_C):

	_sub_triC = _Triangle_into_4SubTriangle(p1_C, p2_C, p3_C)
	
	_centroid_sub_tri = np.zeros( (4,3) ) 
	
	for i in range(4):
		
		sub_tri_p1C = np.array( [ _sub_triC[i,0,0], _sub_triC[i,0,1], _sub_triC[i,0,2] ] )
		sub_tri_p2C = np.array( [ _sub_triC[i,1,0], _sub_triC[i,1,1], _sub_triC[i,1,2] ] )
		sub_tri_p3C = np.array( [ _sub_triC[i,2,0], _sub_triC[i,2,1], _sub_triC[i,2,2] ] )
		
		_centroid_sub_tri[i,0:3] = _centroid_triangle_( sub_tri_p1C, sub_tri_p2C, sub_tri_p3C  )
		
	return _centroid_sub_tri
	
	
def _find_elements_nearby_( x, y, z, epsx, epsy, epsz, _tri_centroid_ ):
	
	#indx0 = [ i for i, value in enumerate(_tri_centroid_[:,0]) if abs(x-value) <= 1.0*epsx ]
	#indy0 = [ i for i, value in enumerate(_tri_centroid_[:,1]) if abs(y-value) <= 1.0*epsy ]
	#indz0 = [ i for i, value in enumerate(_tri_centroid_[:,2]) if abs(z-value) <= 1.0*epsz ]
	
	_len_ = len( _tri_centroid_[:,0] )
	_tri_centroid0_ = np.copy(_tri_centroid_)
	_tri_centroid0_[0:_len_,0] = np.abs( _tri_centroid0_[0:_len_,0] - x ) - epsx
	_tri_centroid0_[0:_len_,1] = np.abs( _tri_centroid0_[0:_len_,1] - y ) - epsy
	_tri_centroid0_[0:_len_,2] = np.abs( _tri_centroid0_[0:_len_,2] - z ) - epsz
	
	_tri_centroid0_[0:_len_,0][_tri_centroid0_[0:_len_,0] <= 0.] = 0. 
	_tri_centroid0_[0:_len_,1][_tri_centroid0_[0:_len_,1] <= 0.] = 0.
	_tri_centroid0_[0:_len_,2][_tri_centroid0_[0:_len_,2] <= 0.] = 0.
	
	tmp0 = np.array( _tri_centroid0_[0:_len_,0] )
	tmp1 = np.array( _tri_centroid0_[0:_len_,1] )
	tmp2 = np.array( _tri_centroid0_[0:_len_,2] )
	
	indx0 = np.where(tmp0 == 0.)[0]
	indy0 = np.where(tmp1 == 0.)[0]
	indz0 = np.where(tmp2 == 0.)[0]
	
	( s1, s2, s3 ) =  ( set(indx0), set(indy0), set(indz0) )
	set1 = s1.intersection(s2)
	result_set = set1.intersection(s3)
	final_list = list(result_set) 
	
	return final_list #_list_, length
	
	
def _find_elements_nearby_3dgrid_( x, y, z, epsx, epsy, epsz, _grid_coord_ ):
	
	#indx0 = [ i for i, value in enumerate(_grid_coord_[:,0]) if abs(x-value) <= 1.0*epsx ]
	#indy0 = [ i for i, value in enumerate(_grid_coord_[:,1]) if abs(y-value) <= 1.0*epsy ]
	#indz0 = [ i for i, value in enumerate(_grid_coord_[:,2]) if abs(z-value) <= 1.0*epsz ]
	
	_len_ = len( _grid_coord_[:,0] )
	_grid_coord0_ = np.copy(_grid_coord_)
	_grid_coord0_[0:_len_,0] = np.abs( _grid_coord0_[0:_len_,0] - x ) - epsx
	_grid_coord0_[0:_len_,1] = np.abs( _grid_coord0_[0:_len_,1] - y ) - epsy
	_grid_coord0_[0:_len_,2] = np.abs( _grid_coord0_[0:_len_,2] - z ) - epsz
	
	_grid_coord0_[0:_len_,0][_grid_coord0_[0:_len_,0] <= 0.] = 0. 
	_grid_coord0_[0:_len_,1][_grid_coord0_[0:_len_,1] <= 0.] = 0.
	_grid_coord0_[0:_len_,2][_grid_coord0_[0:_len_,2] <= 0.] = 0.
	
	tmp0 = np.array( _grid_coord0_[0:_len_,0] )
	tmp1 = np.array( _grid_coord0_[0:_len_,1] )
	tmp2 = np.array( _grid_coord0_[0:_len_,2] )
	
	indx0 = np.where(tmp0 == 0.)[0]
	indy0 = np.where(tmp1 == 0.)[0]
	indz0 = np.where(tmp2 == 0.)[0]
	
	( s1, s2, s3 ) =  ( set(indx0), set(indy0), set(indz0) )
	set1 = s1.intersection(s2)
	result_set = set1.intersection(s3)
	final_list = list(result_set) 
	
	return final_list
	
	
def _find_alltriangels_around_meshgridPoint_( _triXC, _triYC, _triZC ):
	
	x, y, z = _SmoothGrid_Generation_()
	ds = np.zeros(3)
	( ds[0], ds[1], ds[2] ) = ( abs(x[1]-x[0]), abs(y[1]-y[0]), abs(z[1]-z[0]) )
	x3d, y3d, z3d = np.meshgrid(x, y, z)
	x3d = x3d.flatten()
	y3d = y3d.flatten()
	z3d = z3d.flatten()
	
	_tri_list_ = np.zeros( (len(x3d), 50) )
	_len_list_ = np.zeros( len(x3d) ) 
	
	for itr in range( len(x3d) ):
	
		( x0, y0, z0 ) = ( x3d[itr], y3d[itr], z3d[itr] )
		_list_, _length_ =  _find_elements_neraby_( x0, y0, z0, ds, _triXC, _triYC, _triZC )
		
		_tri_list_[itr, 0:_length_] = _list_[0:_length_]
		_len_list_[itr] = _length_
		
	return _tri_list_, _len_list_
	
	
def _chk_GridPoint_close_boundary( x, y, z, delt):
	
	# Domain length
	Lx, Ly, Lz = Domain()
	
	flag = False
	
	# check close to west wall ( x=0, 0<y<Ly )
	dist = abs(x)
	if dist < delt: flag=True
	# check close to east wall ( x=0, 0<y<Ly )
	dist = Lx - x
	if dist < delt: flag=True
	
	# check close to south wall ( y=0, 0<x<Lx )
	dist = abs(y)
	if dist < delt: flag=True
	# check close to north wall ( y=0, 0<x<Lx )
	dist = Ly - y
	if dist < delt: flag=True

	# check close to down or up wall ( y=0, 0<x<Lx )
	dist = Lz - abs(z)
	if dist < delt: flag=True
	
#	flag_chk = _chk_GridPoint_on_corner_pt_( x, y, z )
#	if flag_chk: flag = False
	
#	flag_chk = _chk_GridPoint_on_corner_line_( x, y, z )
#	if flag_chk: flag = False
	
	flag_chk = _chk_GridPoint_on_boundary( x, y, z )
	if flag_chk: flag = False
	
#	flag_chk = _chk_GridPoint_cl_wall_corner_( x, y, z )
#	if flag_chk: flag = False
	
	return flag
	

def _chk_GridPoint_on_boundary( x, y, z ):
	
	# Domain length
	Lx, Ly, Lz = Domain()
	
	flag = False
	
	if x == 0:  flag=True	# check on west wall ( x=0, 0<y<Ly )
	if x == Lx: flag=True	# check on east wall ( x=0, 0<y<Ly )
	
	if y == 0:  flag=True	# check on south wall ( y=0, 0<x<Lx )
	if y == Ly: flag=True	# check on north wall ( y=0, 0<x<Lx )
	
	if z == abs(Lz): flag=True	# check on down or up wall ( y=0, 0<x<Lx )
	
	#flag_chk = _chk_GridPoint_on_corner_pt_( x, y, z )
	#if flag_chk: flag = False
	
	#flag_chk = _chk_GridPoint_on_corner_line_( x, y, z )
	#if flag_chk: flag = False
	
	flag_chk = _chk_GridPoint_cl_corner_line_( x, y, z )
	if flag_chk: flag = False
	
	#flag_chk = _chk_GridPoint_cl_wall_corner_( x, y, z )
	#if flag_chk: flag = False
	
	return flag
	
	
#def _chk_GridPoint_on_corner_pt_( x, y, z ):
#		
#	# Domain length
#	Lx, Ly, Lz = Domain()
#	
#	flag = False
#	
#	if x == 0. and y == 0. and z == 1.*Lz: flag = True
#	if x == 0. and y == 0. and z == -1.*Lz: flag = True
#	
#	if x == 0. and y == Ly and z == 1.*Lz: flag = True
#	if x == 0. and y == Ly and z == -1.*Lz: flag = True
#	
#	if x == Lx and y == 0. and z == 1.*Lz: flag = True
#	if x == Lx and y == 0. and z == -1.*Lz: flag = True
#	
#	if x == Lx and y == Ly and z == 1.*Lz: flag = True
#	if x == Lx and y == Ly and z == -1.*Lz: flag = True
#	
#	return flag
	
	
#def _chk_GridPoint_on_corner_line_( x, y, z ):
#	
#	# Domain length
#	Lx, Ly, Lz = Domain()
#	
#	flag = False
#	
#	if x == 0. and y == 0.: flag = True	
#	if x == 0. and y == Ly: flag = True	
#	if x == Lx and y == 0.: flag = True	
#	if x == Lx and y == Ly: flag = True
#	
#	return flag
	
	
def _chk_GridPoint_cl_corner_line_( x, y, z ):
	
	# Domain length
	Lx, Ly, Lz = Domain()
	xg, yg, _ = _SmoothGrid_Generation_()
	( nx, ny ) = ( len(xg), len(yg) )
	flag = False
	
	if x == xg[1] and y == yg[1]: flag = True
	if x == xg[1] and y == yg[ny-2]: flag = True
	if x == xg[nx-2] and y == yg[1]: flag = True
	if x == xg[nx-2] and y == yg[ny-2]: flag = True	
	
	return flag
	
	
#def _chk_GridPoint_cl_wall_corner_( x, y, z ):
#	
#	# Domain length
#	Lx, Ly, Lz = Domain()
#	xg, yg, _ = _SmoothGrid_Generation_()
#	( nx, ny ) = ( len(xg), len(yg) )
#	flag = False
#	
#	if x == xg[1] and y == yg[0]: flag = True
#	if x == xg[0] and y == yg[1]: flag = True
#	
#	if x == xg[0] and y == yg[ny-2]: flag = True
#	if x == xg[1] and y == yg[ny-1]: flag = True
#	
#	if x == xg[nx-2] and y == yg[0]:  flag = True
#	if x == xg[nx-1] and y == yg[1]: flag = True
#	
#	if x == xg[nx-1] and y == yg[ny-2]: flag = True
#	if x == xg[nx-2] and y == yg[ny-1]: flag = True	
#	
#	return flag	
	
	
def _find_tri_elems_todelete_at_boundary( _TriVert_p1C, _TriVert_p2C, _TriVert_p3C, tri_verti_, del_split ):
	
	delx = del_split
	Lx, Ly, Lz = Domain()
	
	# remove triangles from left boundary 
	_indx_l1 = [ i for i, value in enumerate(_TriVert_p1C[:,0]) if value < -delx ]
	_indx_l2 = [ i for i, value in enumerate(_TriVert_p2C[:,0]) if value < -delx ]
	_indx_l3 = [ i for i, value in enumerate(_TriVert_p3C[:,0]) if value < -delx ]
	_indx_l  = _indx_l1 + _indx_l2 + _indx_l3
	
	# remove triangles from right boundary 
	_indx_r1 = [ i for i, value in enumerate(_TriVert_p1C[:,0]) if value > Lx + delx ]
	_indx_r2 = [ i for i, value in enumerate(_TriVert_p2C[:,0]) if value > Lx + delx ]
	_indx_r3 = [ i for i, value in enumerate(_TriVert_p3C[:,0]) if value > Lx + delx ]
	_indx_r  = _indx_r1 + _indx_r2 + _indx_r3
	
	# remove triangles from southern boundary
	_indx_s1 = [ i for i, value in enumerate(_TriVert_p1C[:,1]) if value < -delx ]
	_indx_s2 = [ i for i, value in enumerate(_TriVert_p2C[:,1]) if value < -delx ]
	_indx_s3 = [ i for i, value in enumerate(_TriVert_p3C[:,1]) if value < -delx ]
	_indx_s  = _indx_s1 + _indx_s2 + _indx_s3
	
	# remove triangles from northern boundary
	_indx_n1 = [ i for i, value in enumerate(_TriVert_p1C[:,1]) if value < Ly + delx ]
	_indx_n2 = [ i for i, value in enumerate(_TriVert_p2C[:,1]) if value < Ly + delx ]
	_indx_n3 = [ i for i, value in enumerate(_TriVert_p3C[:,1]) if value < Ly + delx ]
	_indx_n  = _indx_n1 + _indx_n2 + _indx_n3
	
	_indx_l = np.unique(_indx_l).tolist()
	_indx_r = np.unique(_indx_r).tolist()
	_indx_s = np.unique(_indx_s).tolist()
	_indx_n = np.unique(_indx_n).tolist()
	
	return _indx_l, _indx_r, _indx_s, _indx_n
	

def _tri_elems_delete_at_boundary( _TriVert_p1C, _TriVert_p2C, _TriVert_p3C, tri_verti_, del_split ):
	
	_indx_l, _indx_r, _indx_s, _indx_n = \
	_find_tri_elems_todelete_at_boundary( _TriVert_p1C, _TriVert_p2C, _TriVert_p3C, tri_verti_, del_split )
	
	if len(_indx_l) > 0:
		for itr in range( len(_indx_l) ):
			tri_verti_[:,0].remove( tri_verti_[ _indx_l[itr],0 ] )
			tri_verti_[:,1].remove( tri_verti_[ _indx_l[itr],1 ] )
			tri_verti_[:,2].remove( tri_verti_[ _indx_l[itr],2 ] )
			
	if len(_indx_r) > 0:
		for itr in range( len(_indx_r) ):
			tri_verti_[:,0].remove( tri_verti_[ _indx_r[itr],0 ] )
			tri_verti_[:,1].remove( tri_verti_[ _indx_r[itr],1 ] )
			tri_verti_[:,2].remove( tri_verti_[ _indx_r[itr],2 ] )
			
	if len(_indx_s) > 0:
		for itr in range( len(_indx_r) ):
			tri_verti_[:,0].remove( tri_verti_[ _indx_s[itr],0 ] )
			tri_verti_[:,1].remove( tri_verti_[ _indx_s[itr],1 ] )
			tri_verti_[:,2].remove( tri_verti_[ _indx_s[itr],2 ] )
			
	if len(_indx_n) > 0:
		for itr in range( len(_indx_r) ):
			tri_verti_[:,0].remove( tri_verti_[ _indx_n[itr],0 ] )
			tri_verti_[:,1].remove( tri_verti_[ _indx_n[itr],1 ] )
			tri_verti_[:,2].remove( tri_verti_[ _indx_n[itr],2 ] )
	
	return tri_verti_
	
	
def _find_node_todelete_at_boundary( _nodeXC, _nodeYC, _nodeZC, del_split ):
	
	delx = del_split
	Lx, Ly, Lz = Domain()
	
	# remove nodes from left and right boundary 
	_indx_l = [ i for i, value in enumerate(_nodeXC) if value < -delx ]	
	_indx_r = [ i for i, value in enumerate(_nodeXC) if value > Lx + delx ]
	
	# remove nodes from southern and northern boundary
	_indx_s = [ i for i, value in enumerate(_nodeYC) if value < -delx ]
	_indx_n = [ i for i, value in enumerate(_nodeYC) if value < Ly + delx ]
	
	return _indx_l, _indx_r, _indx_s, _indx_n
	

def _delete_nodes_at_boundary( _nodeXC, _nodeYC, _nodeZC, del_split ):
	
	_indx_l, _indx_r, _indx_s, _indx_n = _find_node_todelete_at_boundary( _nodeXC, _nodeYC, _nodeZC, del_split )
	
	if len(_indx_l) > 0:
		for itr in range( len(_indx_l) ):
			_nodeXC.remove( _indx_l[itr] )
			_nodeYC.remove( _indx_l[itr] )
			_nodeZC.remove( _indx_l[itr] )
			
	if len(_indx_r) > 0:
		for itr in range( len(_indx_r) ):
			_nodeXC.remove( _indx_r[itr] )
			_nodeYC.remove( _indx_r[itr] )
			_nodeZC.remove( _indx_r[itr] )
			
	if len(_indx_s) > 0:
		for itr in range( len(_indx_s) ):
			_nodeXC.remove( _indx_s[itr] )
			_nodeYC.remove( _indx_s[itr] )
			_nodeZC.remove( _indx_s[itr] )
			
	if len(_indx_n) > 0:
		for itr in range( len(_indx_n) ):
			_nodeXC.remove( _indx_n[itr] )
			_nodeYC.remove( _indx_n[itr] )
			_nodeZC.remove( _indx_n[itr] )
	
	return _nodeXC, _nodeYC, _nodeZC 
	
	
def _delete_nodeVel_at_boundary( _nodeXC, _nodeYC, _nodeZC, del_split, _uXg_, _vYg_, _wZg_ ):
	
	_indx_l, _indx_r, _indx_s, _indx_n = _find_node_todelete_at_boundary( _nodeXC, _nodeYC, _nodeZC, del_split )
	
	if len(_indx_l) > 0:
		for itr in range( len(_indx_l) ):
			_uXg_.remove( _indx_l[itr] )
			_vYg_.remove( _indx_l[itr] )
			_wZg_.remove( _indx_l[itr] )
			
	if len(_indx_r) > 0:
		for itr in range( len(_indx_r) ):
			_uXg_.remove( _indx_r[itr] )
			_vYg_.remove( _indx_r[itr] )
			_wZg_.remove( _indx_r[itr] )
			
	if len(_indx_s) > 0:
		for itr in range( len(_indx_s) ):
			_uXg_.remove( _indx_s[itr] )
			_vYg_.remove( _indx_s[itr] )
			_wZg_.remove( _indx_s[itr] )
			
	if len(_indx_n) > 0:
		for itr in range( len(_indx_n) ):
			_uXg_.remove( _indx_n[itr] )
			_vYg_.remove( _indx_n[itr] )
			_wZg_.remove( _indx_n[itr] )
	
	return _uXg_, _vYg_, _wZg_
	
	
	
