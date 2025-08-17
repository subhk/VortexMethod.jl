
import numpy as np
import multiprocessing as mp

# to run in parallel core
from vor3d_paral_mod_a import _node_Circulation
#from vel3d_paral_mod_a import Vel3d_

# to run in single core
#from vor3d_single_mod import Vor3d_
#from vel3d_single_mod import Vel3d_

from remeshing_3d import _elements_splitting_
from utility_3d_paral import _cal_Normal_Vector


def _Sim_Gma_gRidp_(_nodeXC, _nodeYC, _nodeZC, _eleGmaX_, _eleGmaY_, _eleGmaZ_, _no_tri_, tri_verti_, At, dt):
		
	# normal vector at each node points
	#Obj_ = _elements_splitting_(_xg, _yg, _zg, _no_triangle_, tri_vertices_, _del_split)
	#_node_unit_normal_vec_ = Obj_._each_node_unit_normal_(_node_share_cmn_eles_, _leng_)
	
	Obj_ = _cal_Normal_Vector(_nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_)
	_tri_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
		
	_eleGmaX_[0:_no_tri_] += - 2.*At*_tri_unit_normal_vec_[0:_no_tri_,1]*dt
	_eleGmaY_[0:_no_tri_] += + 2.*At*_tri_unit_normal_vec_[0:_no_tri_,0]*dt
	_eleGmaZ_[0:_no_tri_] += + 0.
		
	return _eleGmaX_, _eleGmaY_, _eleGmaZ_
	

def _update_Circulation(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _nodeCirc, _no_tri_, tri_verti_, At, dt):

	Obj_ = _cal_Normal_Vector(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _no_tri_, tri_verti_)
	_tri_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
	
	_leng_ = len(_nodeXC)
	_nodeCoord = np.zeros( (_leng_,3) )
	_nodeCoord[:,0] = _nodeXC
	_nodeCoord[:,1] = _nodeYC
	_nodeCoord[:,2] = _nodeYC 

	del_eleGma = np.zeros( (_no_tri_,3) )
	
	# additional change of vorticity due to barcoclinic effets
	del_eleGma[0:_no_tri_,0] = - 2.*At*_tri_unit_normal_vec_[0:_no_tri_,1]*dt
	del_eleGma[0:_no_tri_,1] = + 2.*At*_tri_unit_normal_vec_[0:_no_tri_,0]*dt
	del_eleGma[0:_no_tri_,2] = 0.	
	
	# updation of node circulation
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, del_eleGma)
	tmp = Obj_._change_in_Circu()
	
	_nodeCirc[0:_no_tri_,0:3] += tmp[0:_no_tri_,0:3]

	return _nodeCirc
	

#def _Sim_Upv_gRidp_(_nodeXC, _nodeYC, _nodeZC, _eleGma, delta_, _Centroid_, _Triangle_Area_):
#
#	_leng_ = len(_nodeXC)
#	_nodeCoord = np.zeros( (_leng_,3) )
#	_nodeCoord[:,0] = _nodeXC
#	_nodeCoord[:,1] = _nodeYC
#	_nodeCoord[:,2] = _nodeYC 	
#	
#	Obj_ = Vel3d_(_nodeCoord, _eleGma, delta_, _Centroid_, _Triangle_Area_)
#	_uXg_, _vYg_, _wZg_ = Obj_._update_Upv_gRrip_()
#
#	return _uXg_, _vYg_, _wZg_
	

def _Sim_Xpv_gRidp_(_nodeXC, _nodeYC, _nodeZC, ux, vy, wz, dt): #, uxold, vyold, wzold, dt):

	gr_pts = len(_nodeXC)

#	_nodeXC[0:gr_pts] +=  dt*(1.5*ux[0:gr_pts] - 0.5*uxold[0:gr_pts])
#	_nodeYC[0:gr_pts] +=  dt*(1.5*vy[0:gr_pts] - 0.5*vyold[0:gr_pts])
#	_nodeZC[0:gr_pts] +=  dt*(1.5*wz[0:gr_pts] - 0.5*wzold[0:gr_pts])
	
	_nodeXC[0:gr_pts] +=  dt*ux[0:gr_pts] 
	_nodeYC[0:gr_pts] +=  dt*vy[0:gr_pts] 
	_nodeZC[0:gr_pts] +=  dt*wz[0:gr_pts] 
	
	return _nodeXC, _nodeYC, _nodeZC
	
	
def _Sim_Xpv_gRidp_AB_(_nodeXC, _nodeYC, _nodeZC, ux, vy, wz, uxold, vyold, wzold, dt):
	
	gr_pts = len(_nodeXC)
	
	_nodeXC[0:gr_pts] +=  dt*(1.5*ux[0:gr_pts] - 0.5*uxold[0:gr_pts])
	_nodeYC[0:gr_pts] +=  dt*(1.5*vy[0:gr_pts] - 0.5*vyold[0:gr_pts])
	_nodeZC[0:gr_pts] +=  dt*(1.5*wz[0:gr_pts] - 0.5*wzold[0:gr_pts])
	
#	_nodeXC[0:gr_pts] +=  dt*ux[0:gr_pts] 
#	_nodeYC[0:gr_pts] +=  dt*vy[0:gr_pts] 
#	_nodeZC[0:gr_pts] +=  dt*wz[0:gr_pts] 
	
	return _nodeXC, _nodeYC, _nodeZC


