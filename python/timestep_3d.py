
import numpy as np
import multiprocessing as mp

# to run in parallel core
from vor3d_paral_mod_a import _node_Circulation
from init_3d import _tri_coord_, _tri_coord_mod_
from utility_3d_paral import _cal_Normal_Vector, _SmoothGrid_Generation_
from vor3d_paral_mod_a import _node_Circulation, _cal_nodeVelocity_frm_gridVelocity, _cal_eleVorStrgth_frm_nodeCirculation


def _Sim_Gma_gRidp_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGmaX_, _eleGmaY_, _eleGmaZ_, _no_tri_, tri_verti_, Atg, dt):
	
	Obj_ = _cal_Normal_Vector(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _no_tri_, tri_verti_)
	_tri_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
		
	_eleGmaX_[0:_no_tri_] += + 2.*Atg*_tri_unit_normal_vec_[0:_no_tri_,1]*dt
	_eleGmaY_[0:_no_tri_] += - 2.*Atg*_tri_unit_normal_vec_[0:_no_tri_,0]*dt
	_eleGmaZ_[0:_no_tri_] += + 0.
		
	return _eleGmaX_, _eleGmaY_, _eleGmaZ_
	
	
def _change_in_VortexStrength_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGma, _nodeCirc, _no_tri_, tri_verti_, Atg):
	
	VortexStretch_ = np.zeros( (_no_tri_,3) )
	
	_eleGma_x, _eleGma_y, _eleGma_z = \
	_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC )
	
	VortexStretch_[:,0] = _eleGma_x - _eleGma[:,0]
	VortexStretch_[:,1] = _eleGma_y - _eleGma[:,1]
	VortexStretch_[:,2] = _eleGma_z - _eleGma[:,2]
	
	barcolinic_gen = np.zeros( (_no_tri_,3) )
	Obj_ = _cal_Normal_Vector(_nodeXC, _nodeYC, _nodeZC,  _triXC, _triYC, _triZC, _no_tri_, tri_verti_)
	_tri_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
	
	barcolinic_gen[0:_no_tri_,0] = + 2.*Atg*_tri_unit_normal_vec_[0:_no_tri_,1]
	barcolinic_gen[0:_no_tri_,1] = - 2.*Atg*_tri_unit_normal_vec_[0:_no_tri_,0]
	barcolinic_gen[0:_no_tri_,2] = + 0.
	
	VortexStretch_[:,0] = VortexStretch_[:,0] + barcolinic_gen[:,0] 
	VortexStretch_[:,1] = VortexStretch_[:,1] + barcolinic_gen[:,1]
	VortexStretch_[:,2] = VortexStretch_[:,2] + barcolinic_gen[:,2]
	
	return VortexStretch_
	
	
def _change_in_VortexStrength_baroclinic(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, Atg):
	
	barcolinic_gen = np.zeros( (_no_tri_,3) )
	Obj_ = _cal_Normal_Vector(_nodeXC, _nodeYC, _nodeZC,  _triXC, _triYC, _triZC, _no_tri_, tri_verti_)
	_tri_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
	
	barcolinic_gen[0:_no_tri_,0] = + 2.*Atg*_tri_unit_normal_vec_[0:_no_tri_,1]
	barcolinic_gen[0:_no_tri_,1] = - 2.*Atg*_tri_unit_normal_vec_[0:_no_tri_,0]
	barcolinic_gen[0:_no_tri_,2] = + 0.
	
	return barcolinic_gen
	
	
def _update_Circulation(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _nodeCirc, _no_tri_, tri_verti_, At, dt):

	Obj_ = _cal_Normal_Vector(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _no_tri_, tri_verti_)
	_tri_unit_normal_vec_ = Obj_._all_triangle_surf_normal_()
	
	_leng_ = len(_nodeXC)
	_nodeCoord = np.zeros( (_leng_,3) )
	( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )

	del_eleGma = np.zeros( (_no_tri_,3) )
	
	# additional changes of vorticity due to barcoclinic effects
	del_eleGma[0:_no_tri_,0] = + 2.*At*_tri_unit_normal_vec_[0:_no_tri_,1]*dt
	del_eleGma[0:_no_tri_,1] = - 2.*At*_tri_unit_normal_vec_[0:_no_tri_,0]*dt
	del_eleGma[0:_no_tri_,2] = 0.
	
	# updation of  node circulation
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, del_eleGma)
	tmp = Obj_._change_in_Circu()
	
	_nodeCirc[0:_no_tri_,0:3] += tmp[0:_no_tri_,0:3]

	return _nodeCirc
	
	
def _Sim_Xpv_gRidp_(_nodeXC, _nodeYC, _nodeZC, ux, vy, wz, dt): #, uxold, vyold, wzold, dt):

	gr_pts = len(_nodeXC)
	
	_nodeXC[0:gr_pts] +=  dt*ux[0:gr_pts] 
	_nodeYC[0:gr_pts] +=  dt*vy[0:gr_pts] 
	_nodeZC[0:gr_pts] +=  dt*wz[0:gr_pts] 
	
	return _nodeXC, _nodeYC, _nodeZC
	
	
def _Sim_Xpv_gRidp_AB_(_nodeXC, _nodeYC, _nodeZC, ux, vy, wz, uxold, vyold, wzold, dt):
	
	gr_pts = len(_nodeXC)
	
	_nodeXC[0:gr_pts] +=  dt*(1.5*ux[0:gr_pts] - 0.5*uxold[0:gr_pts])
	_nodeYC[0:gr_pts] +=  dt*(1.5*vy[0:gr_pts] - 0.5*vyold[0:gr_pts])
	_nodeZC[0:gr_pts] +=  dt*(1.5*wz[0:gr_pts] - 0.5*wzold[0:gr_pts]) 
	
	return _nodeXC, _nodeYC, _nodeZC
	
	
def _RK2_timestep_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGma, _no_tri_, tri_verti_, At):
	
	gr_pts = len(_nodeXC)
#	_eleGma = np.zeros( (_no_tri_,3) )
	_leng_ = len(_nodeXC)
	_nodeCoord = np.zeros( (_leng_,3) )
	
	x, y, z = _SmoothGrid_Generation_()
	dx = min( abs( x[1] - x[0] ), abs( y[1] - y[0] ) )
	Courant_Num = 0.5
	
############### time-step : 1st part
	
	# (1a) calculating circulation from vortex strength
	( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma)
	_nodeCirc = Obj_._change_in_Circu()
	
	# (2a) calculating node velocity from vortex strength
	_uXg0_, _vYg0_, _wZg0_, _, max_vel = \
	_cal_nodeVelocity_frm_gridVelocity(_eleGma, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
	
	dt = Courant_Num*dx/max_vel
	
	# (3a) advect nodes
	_nodeXC[0:gr_pts] +=  0.5*dt*_uXg0_[0:gr_pts]
	_nodeYC[0:gr_pts] +=  0.5*dt*_vYg0_[0:gr_pts]
	_nodeZC[0:gr_pts] +=  0.5*dt*_wZg0_[0:gr_pts]
	
	# update vortex strength due to 
	# (4a) -> vortex stretching + barcoclinic energy
	_triXC, _triYC, _triZC = _tri_coord_mod_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
	
	_nodeCirc = \
	_update_Circulation(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _nodeCirc, _no_tri_, tri_verti_, At, 0.5*dt)
	
	_eleGma[:,0], _eleGma[:,1], _eleGma[:,2] = \
	_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC )
	
############### time-step : 2nd part
	
	# (1b) calculating node velocity from vortex strength
	_uXg0_, _vYg0_, _wZg0_, _, _ = \
	_cal_nodeVelocity_frm_gridVelocity(_eleGma, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
	
	# (2b) advect nodes
	_nodeXC[0:gr_pts] +=  0.5*dt*_uXg0_[0:gr_pts]
	_nodeYC[0:gr_pts] +=  0.5*dt*_vYg0_[0:gr_pts]
	_nodeZC[0:gr_pts] +=  0.5*dt*_wZg0_[0:gr_pts]
	
	# update vortex strength due to 
	# (3b) -> vortex stretching + barcoclinic energy
	_triXC, _triYC, _triZC = _tri_coord_mod_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
	
	_nodeCirc = \
	_update_Circulation(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _nodeCirc, _no_tri_, tri_verti_, At, 0.5*dt)
	
	_eleGma[:,0], _eleGma[:,1], _eleGma[:,2] = \
	_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC )
	
	return _eleGma, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, dt 
	
	
def _RK2h_timestep_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGma, _no_tri_, tri_verti_, At):
	
	gr_pts = len(_nodeXC)
#	_eleGma = np.zeros( (_no_tri_,3) )
	_leng_ = len(_nodeXC)
	_nodeCoord = np.zeros( (_leng_,3) )
	
	_nodeXC0 = np.zeros(gr_pts)
	_nodeYC0 = np.zeros(gr_pts)
	_nodeZC0 = np.zeros(gr_pts)
	
	_eleGma0 = np.zeros( (_no_tri_,3))
	
	x, y, z = _SmoothGrid_Generation_()
	dx = min( abs( x[1] - x[0] ), abs( y[1] - y[0] ) )
	Courant_Num = 0.25
	
############### time-step : 1st part
	
	# (1a) calculating circulation from vortex strength
	( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma)
	_nodeCirc = Obj_._change_in_Circu()
	
	# (2a) calculating node velocity from vortex strength
	_uXg0_, _vYg0_, _wZg0_, _, max_vel = \
	_cal_nodeVelocity_frm_gridVelocity(_eleGma, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
	
	dt = Courant_Num*dx/max_vel
	
	# (3a) advect nodes
	_nodeXC0[0:gr_pts] = _nodeXC[0:gr_pts] + dt*_uXg0_[0:gr_pts]
	_nodeYC0[0:gr_pts] = _nodeYC[0:gr_pts] + dt*_vYg0_[0:gr_pts]
	_nodeZC0[0:gr_pts] = _nodeZC[0:gr_pts] + dt*_wZg0_[0:gr_pts]
	
	# update vortex strength due to 
	# (4a) -> vortex stretching + barcoclinic energy
	_triXC, _triYC, _triZC = _tri_coord_mod_( _nodeXC0, _nodeYC0, _nodeZC0, _no_tri_, tri_verti_ )
	
	barcolinic_gen0 = \
	_change_in_VortexStrength_baroclinic(_nodeXC0, _nodeYC0, _nodeZC0, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, At)
	
	_eleGma0[:,0], _eleGma0[:,1], _eleGma0[:,2] = \
	_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _nodeXC0, _nodeYC0, _nodeZC0, _triXC, _triYC, _triZC )
	
	_eleGma0[:,0] += dt*barcolinic_gen0[:,0]
	_eleGma0[:,1] += dt*barcolinic_gen0[:,1]
	_eleGma0[:,2] += dt*barcolinic_gen0[:,2]
	
#	( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC0, _nodeYC0, _nodeZC0 )
#	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma0)
#	_nodeCirc = Obj_._change_in_Circu()
	
############### time-step : 2nd part
	
	# (1b) calculating node velocity from vortex strength
	_uXg1_, _vYg1_, _wZg1_, _, _ = \
	_cal_nodeVelocity_frm_gridVelocity(_eleGma0, _no_tri_, tri_verti_, _nodeXC0, _nodeYC0, _nodeZC0, _triXC, _triYC, _triZC)
	
	# (2b) advect nodes
	_nodeXC[0:gr_pts] +=  0.5*dt*( _uXg0_[0:gr_pts] + _uXg1_[0:gr_pts] )
	_nodeYC[0:gr_pts] +=  0.5*dt*( _vYg0_[0:gr_pts] + _vYg1_[0:gr_pts] )
	_nodeZC[0:gr_pts] +=  0.5*dt*( _wZg0_[0:gr_pts] + _wZg1_[0:gr_pts] )
	
	# update vortex strength due to 
	# (3b) -> vortex stretching + barcoclinic energy
	_triXC, _triYC, _triZC = _tri_coord_mod_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
	
	barcolinic_gen1 = \
	_change_in_VortexStrength_baroclinic(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, At)
	
	_eleGma0[:,0], _eleGma0[:,1], _eleGma0[:,2] = \
	_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _nodeXC0, _nodeYC0, _nodeZC0, _triXC, _triYC, _triZC )
	
	_eleGma[:,0] = _eleGma0[:,0] + 0.5*dt*( barcolinic_gen0[:,0] + barcolinic_gen1[:,0] )
	_eleGma[:,1] = _eleGma0[:,1] + 0.5*dt*( barcolinic_gen0[:,1] + barcolinic_gen1[:,1] )
	_eleGma[:,2] = _eleGma0[:,2] + 0.5*dt*( barcolinic_gen0[:,2] + barcolinic_gen1[:,2] )
	
#	_eleGma[:,0], _eleGma[:,1], _eleGma[:,2] = \
#	_cal_eleVorStrgth_frm_nodeCirculation( _nodeCirc, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC )
	
	
	return _eleGma, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, dt 
	
	
def _RK44_timestep_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGma, _no_tri_, tri_verti_, At):
	
	gr_pts = len(_nodeXC)
	_eleGma0 = np.zeros( (_no_tri_,3) )
	_leng_ = len(_nodeXC)
	_nodeCoord = np.zeros( (_leng_,3) )
	
	x, y, z = _SmoothGrid_Generation_()
	dx = abs(x[1]-x[0])
	Courant_Num = 0.4
	
	ds1 = np.zeros( (gr_pts, 3) )
	ds2 = np.zeros( (gr_pts, 3) )
	ds3 = np.zeros( (gr_pts, 3) )
	ds4 = np.zeros( (gr_pts, 3) )
	
	d_gma1 = np.zeros( (_no_tri_, 3) )
	d_gma2 = np.zeros( (_no_tri_, 3) )
	d_gma3 = np.zeros( (_no_tri_, 3) )
	d_gma4 = np.zeros( (_no_tri_, 3) )
	
############### time-step : 1st part
	
	# (1a) calculating circulation from vortex strength
	( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma)
	_nodeCirc = Obj_._change_in_Circu()
	
	# (2a) calculating node velocity from vortex strength
	_uXg0_, _vYg0_, _wZg0_, _, max_vel = \
	_cal_nodeVelocity_frm_gridVelocity(_eleGma, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC)
	
	dt = Courant_Num*dx/max_vel
	
	# (3a) advect nodes
	ds1[0:gr_pts,0] = dt*_uXg0_[0:gr_pts]
	ds1[0:gr_pts,1] = dt*_vYg0_[0:gr_pts]
	ds1[0:gr_pts,2] = dt*_wZg0_[0:gr_pts]
	
	# update vortex strength due to 
	# (4a) -> vortex stretching + barcoclinic torque
	VortexStretch0_ = \
	_change_in_VortexStrength_baroclinic(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, _eleGma, _nodeCirc, _no_tri_, tri_verti_, At)
	
	d_gma1[:,0] = dt*VortexStretch0_[:,0]
	d_gma1[:,1] = dt*VortexStretch0_[:,1]
	d_gma1[:,2] = dt*VortexStretch0_[:,2]
	
############### time-step : 2nd part
	
	# (1b) calculating node velocity from vortex strength
	( _eleGma0[:,0], _eleGma0[:,1], _eleGma0[:,2] ) = ( _eleGma[:,0]+0.5*d_gma1[:,0], _eleGma[:,1]+0.5*d_gma1[:,1], _eleGma[:,2]+0.5*d_gma1[:,2] )
	_triXC, _triYC, _triZC = _tri_coord_mod_( _nodeXC+0.5*ds1[:,0], _nodeYC+0.5*ds1[:,1], _nodeZC+0.5*ds1[:,2], _no_tri_, tri_verti_ )
	_duX_, _dvY_, _dwZ_, _, _ = \
	_cal_nodeVelocity_frm_gridVelocity\
	(_eleGma0, _no_tri_, tri_verti_, _nodeXC+0.5*ds1[:,0], _nodeYC+0.5*ds1[:,1], _nodeZC+0.5*ds1[:,2], _triXC, _triYC, _triZC)
	
	# (2b) advect nodes
	ds2[0:gr_pts,0] = dt*( _uXg0_[0:gr_pts] + 0.5*_duX_ )
	ds2[0:gr_pts,1] = dt*( _vYg0_[0:gr_pts] + 0.5*_dvY_ )
	ds2[0:gr_pts,2] = dt*( _wZg0_[0:gr_pts] + 0.5*_dwZ_ )
	
	# update vortex strength due to 
	# (3b) -> vortex stretching + barcoclinic torque
	VortexStretch0_ = \
	_change_in_VortexStrength_\
	(_nodeXC+0.5*ds1[:,0], _nodeYC+0.5*ds1[:,1], _nodeZC+0.5*ds1[:,2], _triXC, _triYC, _triZC, _eleGma0, _nodeCirc, _no_tri_, tri_verti_, At)
	
	d_gma2[:,0] = dt*VortexStretch0_[:,0]
	d_gma2[:,1] = dt*VortexStretch0_[:,1]
	d_gma2[:,2] = dt*VortexStretch0_[:,2]
	
	( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma0)
	_nodeCirc = Obj_._change_in_Circu()
	
############### time-step : 3rd part
	
	# (1c) calculating node velocity from vortex strength
	( _eleGma0[:,0], _eleGma0[:,1], _eleGma0[:,2] ) = ( _eleGma[:,0]+0.5*d_gma2[:,0], _eleGma[:,1]+0.5*d_gma2[:,1], _eleGma[:,2]+0.5*d_gma2[:,2] )
	_triXC, _triYC, _triZC = _tri_coord_mod_( _nodeXC+0.5*ds2[:,0], _nodeYC+0.5*ds2[:,1], _nodeZC+0.5*ds2[:,2], _no_tri_, tri_verti_ )
	_duX_, _dvY_, _dwZ_, _, _ = \
	_cal_nodeVelocity_frm_gridVelocity\
	(_eleGma0, _no_tri_, tri_verti_, _nodeXC+0.5*ds2[:,0], _nodeYC+0.5*ds2[:,1], _nodeZC+0.5*ds2[:,2], _triXC, _triYC, _triZC)
	
	# (2c) advect nodes
	ds3[0:gr_pts,0] = dt*( _uXg0_[0:gr_pts] + 0.5*_duX_ )
	ds3[0:gr_pts,1] = dt*( _vYg0_[0:gr_pts] + 0.5*_dvY_ )
	ds3[0:gr_pts,2] = dt*( _wZg0_[0:gr_pts] + 0.5*_dwZ_ )
	
	# update vortex strength due to 
	# (3c) -> vortex stretching + barcoclinic torque
	VortexStretch0_ = \
	_change_in_VortexStrength_\
	(_nodeXC+0.5*ds2[:,0], _nodeYC+0.5*ds2[:,1], _nodeZC+0.5*ds2[:,2], _triXC, _triYC, _triZC, _eleGma0, _nodeCirc, _no_tri_, tri_verti_, At)
	
	d_gma3[:,0] = dt*VortexStretch0_[:,0]
	d_gma3[:,1] = dt*VortexStretch0_[:,1]
	d_gma3[:,2] = dt*VortexStretch0_[:,2]
	
	( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
	Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma0)
	_nodeCirc = Obj_._change_in_Circu()	
	
############### time-step : 4th part
	
	# (1d) calculating node velocity from vortex strength
	( _eleGma0[:,0], _eleGma0[:,1], _eleGma0[:,2] ) = ( _eleGma[:,0]+d_gma3[:,0], _eleGma[:,1]+d_gma3[:,1], _eleGma[:,2]+d_gma3[:,2] )
	_triXC, _triYC, _triZC = _tri_coord_mod_( _nodeXC+ds3[:,0], _nodeYC+ds3[:,1], _nodeZC+ds3[:,2], _no_tri_, tri_verti_ )
	_duX_, _dvY_, _dwZ_, _, _ = \
	_cal_nodeVelocity_frm_gridVelocity\
	(_eleGma0, _no_tri_, tri_verti_, _nodeXC+ds3[:,0], _nodeYC+ds3[:,1], _nodeZC+ds3[:,2], _triXC, _triYC, _triZC)
	
	# (2d) advect nodes
	ds4[0:gr_pts,0] = dt*( _uXg0_[0:gr_pts] + _duX_ )
	ds4[0:gr_pts,1] = dt*( _vYg0_[0:gr_pts] + _dvY_ )
	ds4[0:gr_pts,2] = dt*( _wZg0_[0:gr_pts] + _dwZ_ )
	
	# update vortex strength due to 
	# (3d) -> vortex stretching + barcoclinic torque
	VortexStretch0_ = \
	_change_in_VortexStrength_\
	(_nodeXC+ds3[:,0], _nodeYC+ds3[:,1], _nodeZC+ds3[:,2], _triXC, _triYC, _triZC, _eleGma0, _nodeCirc, _no_tri_, tri_verti_, At)
	
	d_gma4[:,0] = dt*VortexStretch0_[:,0]
	d_gma4[:,1] = dt*VortexStretch0_[:,1]
	d_gma4[:,2] = dt*VortexStretch0_[:,2]
	
	_nodeXC[:] += ( ds1[:,0] + 2.*ds2[:,0] + 2.*ds3[:,0] + ds4[:,0] )/6.
	_nodeYC[:] += ( ds1[:,1] + 2.*ds2[:,1] + 2.*ds3[:,1] + ds4[:,1] )/6.
	_nodeZC[:] += ( ds1[:,2] + 2.*ds2[:,2] + 2.*ds3[:,2] + ds4[:,2] )/6.
	
	_eleGma[:,0] += ( d_gma1[:,0] + 2.*d_gma2[:,0] + 2.*d_gma3[:,0] + d_gma4[:,0] )/6.
	_eleGma[:,1] += ( d_gma1[:,1] + 2.*d_gma2[:,1] + 2.*d_gma3[:,1] + d_gma4[:,1] )/6.
	_eleGma[:,2] += ( d_gma1[:,2] + 2.*d_gma2[:,2] + 2.*d_gma3[:,2] + d_gma4[:,2] )/6.
	
	_triXC, _triYC, _triZC = _tri_coord_mod_( _nodeXC, _nodeYC, _nodeZC, _no_tri_, tri_verti_ )
	
	return _eleGma, _nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, dt
	
	
	
	
	
