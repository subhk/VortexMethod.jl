# solve a 2-d Poisson equation by differentiating the discretized
# Poisson equation and then substituting in the inverse Fourier
# transform and solving for the amplitudes in Fourier space.
#
# This is the standard way that texts deal with the Poisson equation
# (see, e.g., Garcia and NR)
#
# Note: we need a periodic problem for an FFT

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl

# Use LaTeX for rendering
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

# the analytic solution
def true(x,y):
    pi = np.pi
    return np.sin(2.0*pi*x)**2*np.cos(4.0*pi*y) + \
        np.sin(4.0*pi*x)*np.cos(2.0*pi*y)**2


# the righthand side
def frhs(x,y):
    pi = np.pi
    return 8.0*pi**2*np.cos(4.0*pi*y)*(np.cos(4.0*pi*x) - 
                                          np.sin(4.0*pi*x)) - \
           16.0*pi**2*(np.sin(4.0*pi*x)*np.cos(2.0*pi*y)**2 + 
                       np.sin(2.0*pi*x)**2 * np.cos(4.0*pi*y))


def doit(Nx, Ny, do_plot=False):

    # create the domain -- cell-centered finite-difference / finite-volume
    xmin = 0.0
    xmax = 1.0

    ymin = 0.0
    ymax = 1.0

    dx = (xmax - xmin)/Nx
    dy = (ymax - ymin)/Ny

    x = (np.arange(Nx) + 0.5)*dx
    y = (np.arange(Ny) + 0.5)*dy

    x2d = np.repeat(x, Ny)
    x2d.shape = (Nx, Ny)

    y2d = np.repeat(y, Nx)
    y2d.shape = (Ny, Nx)
    y2d = np.transpose(y2d)


    # create the RHS
    f = frhs(x2d, y2d)
    print('size = ', np.shape(f))

    # FFT of RHS
    F = np.fft.fft2(f)

    # get the wavenumbers -- we need these to be physical, so divide by dx
    kx = np.fft.fftfreq(Nx)/dx
    ky = np.fft.fftfreq(Ny)/dy

    # make 2-d arrays for the wavenumbers
    kx2d = np.repeat(kx, Ny)
    kx2d.shape = (Nx, Ny)

    ky2d = np.repeat(ky, Nx)
    ky2d.shape = (Ny, Nx)
    ky2d = np.transpose(ky2d)

    # here the FFT frequencies are in the order 0 ... N/2-1, -N/2, ...
    # the 0 component is not a physical frequency, but rather it is
    # the DC signal.  Don't mess with it, since we'll divide by zero
    oldDC = F[0,0]
    F = 0.5*F/( (np.cos(2.0*np.pi*kx2d/Nx) - 1.0)/dx**2 +
                (np.cos(2.0*np.pi*ky2d/Ny) - 1.0)/dy**2)

    F[0,0] = oldDC

    # transform back to real space
    fsolution = np.real(np.fft.ifft2(F))


    # since x is our row in the array, we transpose for the
    # plot
    if do_plot:
        plt.imshow(np.transpose(fsolution),
                   origin="lower", interpolation="nearest",
                   extent=[xmin, xmax, ymin, ymax])

        plt.xlabel("x")
        plt.ylabel("y")

        plt.colorbar()

        plt.tight_layout()

        plt.savefig("poisson_fft.pdf")
        

    # return the error, compared to the true solution
    return np.sqrt(dx*dx*np.sum( ( (fsolution - true(x2d,y2d))**2).flat))


N = [16, 32, 64, 128, 256]
plot = [False]*len(N)
plot[N.index(64)] = True

err = []
for n, p in zip(N, plot):
    err.append(doit(n, 2*n, do_plot=p))


# plot the convergence
plt.clf()

N = np.array(N, dtype=np.float64)
err = np.array(err, dtype=np.float64)

print(N)
print(type(N))
print(err)

plt.scatter(N, err, marker="x", color="r")
plt.plot(N, err[0]*(N[0]/N)**2, "--", color="k", label="$\mathcal{O}(\Delta x^2)$")

ax = plt.gca()

ax.set_xscale('log')
ax.set_yscale('log')

plt.legend(frameon=False)

plt.xlabel("number of zones")
plt.ylabel("L2 norm of abs error")
                        
f = plt.gcf()
f.set_size_inches(5.0,5.0)

plt.savefig("fft-poisson-converge.pdf", bbox_inches="tight")







def _finding_boundary_tri_elements_(_triXC, _triYC, _triZC, tri_verti_, Lx, Ly):
	
	_no_tri_ = len(_triXC[:,0])
	
	_boundary_elements_ = []
	
	for itr in range(_no_tri_):
		indx = np.where( _triXC[itr,0] == _triXC )[0][0]
		indy = np.where( _triYC[itr,0] == _triYC )[0][0]
		ind0 = np.intersect1d( indx, indy )
		
		indx = np.where( _triXC[itr,1] == _triXC )[0][0]
		indy = np.where( _triYC[itr,1] == _triYC )[0][0]
		ind1 = np.intersect1d( indx, indy )
		
		indx = np.where( _triXC[itr,2] == _triXC )[0][0]
		indy = np.where( _triYC[itr,2] == _triYC )[0][0]
		ind2 = np.intersect1d( indx, indy )
		
		flag = True
		if len(ind0) == 1:
			#print( 'ind0 = ', ind0 )
			_boundary_elements_.append(itr)
			flag = False
		
		if flag:
			if len(ind1) == 1:
				#print( 'ind1 = ',  ind1 ) 
				_boundary_elements_.append(itr)
				flag = False
		
		if flag:
			if len(ind2) == 1: 
				#print( 'ind2 = ', ind2 )
				_boundary_elements_.append(itr)
				
	_boundary_elements_ = np.unique(_boundary_elements_).tolist()
	
	print( 'no of boundary elements = ', len(_boundary_elements_) )
	
	_west_eles_ = []
	_east_eles_ = []
	
	_north_eles_ = []
	_south_eles_ = []
	
	for itr in range( len(_boundary_elements_) ):
		_ele_ = _boundary_elements_[itr]
		
		# figuring out elements at western boundary
		p = np.array([ _triXC[_ele_,0], _triXC[_ele_,1], _triXC[_ele_,2] ])
		if abs(np.min(p)) < 0.1*Lx: _west_eles_.append(_ele_)
		
		# figuring out elements at eastern boundary
		if abs(np.max(p)) > 0.9*Lx : _east_eles_.append(_ele_)
		
		# figuring out elements at southern boundary
		p = np.array([ _triYC[_ele_,0], _triYC[_ele_,1], _triYC[_ele_,2] ])
		if abs(np.min(p)) < 0.1*Ly: _south_eles_.append(_ele_)
		
		# figuring out elements at northern boundary
		if abs(np.max(p)) > 0.9*Ly : _north_eles_.append(_ele_)
		
	print( 'west + east + south + north = ', len(_west_eles_)+len(_east_eles_)+len(_south_eles_)+len(_north_eles_) )
	
	return _west_eles_, _east_eles_, _south_eles_, _north_eles_
	
	
def _elements_outof_left_domain_(_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC, delta, Lx):
	
	for itr in range( len(_nodeXC) ):
		
		# on left boundary ->
		if _nodeXC[itr] < - delta:
			_nodep_= itr
			indx0 = np.where( _triXC[:,0] == _nodeXC[ _nodep_ ] )[0]
			indx1 = np.where( _triXC[:,1] == _nodeXC[ _nodep_ ] )[0]
			indx2 = np.where( _triXC[:,2] == _nodeXC[ _nodep_ ] )[0]
			
			_nodeXC[_nodep_] = 0.
			_nodes_ = \
			_find_nearest_two_nodes_( _nodeXC, _nodeYC, _nodeZC, _nodeXC[_nodep_], _nodeYC[_nodep_], _nodeZC[_nodep_] )
			assert len(_nodes_) == 2
			
			# move the left-node to the right side of the domain
			if len(indx0) > 0:
				for ptr in range( len(indx0) ):
					jtr = indx0[ptr]
					( _triXC[jtr,0], _triYC[jtr,0], _triZC[jtr,0] ) = ( _triXC[jtr,0]+Lx, _triYC[jtr,0], _triZC[jtr,0] )
					( _triXC[jtr,1], _triYC[jtr,1], _triZC[jtr,1] ) = ( _nodeXC[_nodes_[0]], _nodeYC[_nodes_[0]], _nodeZC[_nodes_[0]] )
					( _triXC[jtr,2], _triYC[jtr,2], _triZC[jtr,2] ) = ( _nodeXC[_nodes_[1]], _nodeYC[_nodes_[1]], _nodeZC[_nodes_[1]] )
					( tri_verti_[jtr,0], tri_verti_[jtr,1], tri_verti_[jtr,2] ) = ( tri_verti_[jtr,0], _nodes_[0], _nodes_[1] )
			
			if len(indx1) > 0:
				for ptr in range( len(indx1) ):
					jtr = indx1[ptr]
					( _triXC[jtr,1], _triYC[jtr,1], _triZC[jtr,1] ) = ( _triXC[jtr,1]+Lx, _triYC[jtr,1], _triZC[jtr,1] )
					( _triXC[jtr,0], _triYC[jtr,0], _triZC[jtr,0] ) = ( _nodeXC[_nodes_[0]], _nodeYC[_nodes_[0]], _nodeZC[_nodes_[0]] )
					( _triXC[jtr,2], _triYC[jtr,2], _triZC[jtr,2] ) = ( _nodeXC[_nodes_[1]], _nodeYC[_nodes_[1]], _nodeZC[_nodes_[1]] )
					( tri_verti_[jtr,1], tri_verti_[jtr,0], tri_verti_[jtr,2] ) = ( tri_verti_[jtr,1], _nodes_[0], _nodes_[1] )
			
			if len(indx2) > 0:
				for ptr in range( len(indx2) ):
					jtr = indx2[ptr]
					( _triXC[jtr,2], _triYC[jtr,2], _triZC[jtr,2] ) = ( _triXC[jtr,2]+Lx, _triYC[jtr,2], _triZC[jtr,2] )
					( _triXC[jtr,0], _triYC[jtr,0], _triZC[jtr,0] ) = ( _nodeXC[_nodes_[0]], _nodeYC[_nodes_[0]], _nodeZC[_nodes_[0]] )
					( _triXC[jtr,1], _triYC[jtr,1], _triZC[jtr,1] ) = ( _nodeXC[_nodes_[1]], _nodeYC[_nodes_[1]], _nodeZC[_nodes_[1]] )
					( tri_verti_[jtr,2], tri_verti_[jtr,0], tri_verti_[jtr,1] ) = ( tri_verti_[jtr,2], _nodes_[0], _nodes_[1] )
		
		
	return _triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC
	
	
def _elements_outof_right_domain_(_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC, delta, Lx):
	
	for itr in range( len(_nodeXC) ):
		
		# on right boundary ->
		if _nodeXC[itr] > Lx + delta:
			_nodep_= itr
			indx0 = np.where( _triXC[:,0] == _nodeXC[ _nodep_ ] )[0]
			indx1 = np.where( _triXC[:,1] == _nodeXC[ _nodep_ ] )[0]
			indx2 = np.where( _triXC[:,2] == _nodeXC[ _nodep_ ] )[0]
			
			_nodeXC[_nodep_] = Lx
			_nodes_ = \
			_find_nearest_two_nodes_( _nodeXC, _nodeYC, _nodeZC, _nodeXC[_nodep_], _nodeYC[_nodep_], _nodeZC[_nodep_] )
			assert len(_nodes_) == 2
			
			# move the right-node to the left side of the domain
			if len(indx0) > 0:
				for ptr in range( len(indx0) ):
					jtr = indx0[ptr]
					( _triXC[jtr,0], _triYC[jtr,0], _triZC[jtr,0] ) = ( _triXC[jtr,0]-Lx, _triYC[jtr,0], _triZC[jtr,0] )
					( _triXC[jtr,1], _triYC[jtr,1], _triZC[jtr,1] ) = ( _nodeXC[_nodes_[0]], _nodeYC[_nodes_[0]], _nodeZC[_nodes_[0]] )
					( _triXC[jtr,2], _triYC[jtr,2], _triZC[jtr,2] ) = ( _nodeXC[_nodes_[1]], _nodeYC[_nodes_[1]], _nodeZC[_nodes_[1]] )
					( tri_verti_[jtr,0], tri_verti_[jtr,1], tri_verti_[jtr,2] ) = ( tri_verti_[jtr,0], _nodes_[0], _nodes_[1] )
			
			if len(indx1) > 0:
				for ptr in range( len(indx1) ):
					jtr = indx1[ptr]
					( _triXC[jtr,1], _triYC[jtr,1], _triZC[jtr,1] ) = ( _triXC[jtr,1]-Lx, _triYC[jtr,1], _triZC[jtr,1] )
					( _triXC[jtr,0], _triYC[jtr,0], _triZC[jtr,0] ) = ( _nodeXC[_nodes_[0]], _nodeYC[_nodes_[0]], _nodeZC[_nodes_[0]] )
					( _triXC[jtr,2], _triYC[jtr,2], _triZC[jtr,2] ) = ( _nodeXC[_nodes_[1]], _nodeYC[_nodes_[1]], _nodeZC[_nodes_[1]] )
					( tri_verti_[jtr,1], tri_verti_[jtr,0], tri_verti_[jtr,2] ) = ( tri_verti_[jtr,1], _nodes_[0], _nodes_[1] )
			
			if len(indx2) > 0:
				for ptr in range( len(indx2) ):
					jtr = indx2[ptr]
					( _triXC[jtr,2], _triYC[jtr,2], _triZC[jtr,2] ) = ( _triXC[jtr,2]-Lx, _triYC[jtr,2], _triZC[jtr,2] )
					( _triXC[jtr,0], _triYC[jtr,0], _triZC[jtr,0] ) = ( _nodeXC[_nodes_[0]], _nodeYC[_nodes_[0]], _nodeZC[_nodes_[0]] )
					( _triXC[jtr,1], _triYC[jtr,1], _triZC[jtr,1] ) = ( _nodeXC[_nodes_[1]], _nodeYC[_nodes_[1]], _nodeZC[_nodes_[1]] )
					( tri_verti_[jtr,2], tri_verti_[jtr,0], tri_verti_[jtr,1] ) = ( tri_verti_[jtr,2], _nodes_[0], _nodes_[1] )
		
	return _triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC
	
	
	
	
#			_nodeXC, _nodeYC, _nodeZC, _delete_nodes = \
#			_delete_straddle_nodes_(_nodeXC, _nodeYC, _nodeZC, ds, max(x), max(y))
#			
#			if len(_delete_nodes) > 0:
#				_triXC, _triYC, _triZC, _no_tri_, tri_verti_, _tri_ele_share_comm_node_ = \
#				_delete_straddle_elemenents_(_triXC, _triYC, _triZC, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _delete_nodes) 
#				
#				_uXg_, _vYg_, _wZg_= _delete_straddle_nodeVel_(_uXg_, _vYg_, _wZg_, _delete_nodes)
#				_nodeCirc = _delete_straddle_nodeCirculation_( _nodeCirc, _tri_ele_share_comm_node_ )
#				_eleGma = _delete_straddle_eleGma_( _eleGma, _tri_ele_share_comm_node_ )
#				
#				print('No of triangle element = ', _no_tri_)
				
#			_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC = \
#			_elemenst_outof_domain_(_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC, ds_max)
#			
#			_eleGma = _recalculate_vortex_strength( _eleGma, _Triangle_Area_, _triXC, _triYC, _triZC, _no_tri_, tri_verti_)
#			
#			_leng_ = len(_nodeXC)
#			_nodeCoord = np.zeros( (_leng_,3) )
#			( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
#			Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma)
#			_nodeCirc = Obj_._change_in_Circu()
	
	
#				_nodeXC, _nodeYC, _nodeZC, _delete_nodes = \
#				_delete_straddle_nodes_(_nodeXC, _nodeYC, _nodeZC, ds, max(x), max(y))
#				
#				if len(_delete_nodes) > 0:
#					_triXC, _triYC, _triZC, _no_tri_, tri_verti_, _tri_ele_share_comm_node_ = \
#					_delete_straddle_elemenents_(_triXC, _triYC, _triZC, _no_tri_, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _delete_nodes) 
#					
#					_uXg_, _vYg_, _wZg_= _delete_straddle_nodeVel_(_uXg_, _vYg_, _wZg_, _delete_nodes)
#					_nodeCirc = _delete_straddle_nodeCirculation_( _nodeCirc, _tri_ele_share_comm_node_ )
#					_eleGma = _delete_straddle_eleGma_( _eleGma, _tri_ele_share_comm_node_ )
#					
#					print('No of triangle element = ', _no_tri_)
				
#				_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC = \
#				_elemenst_outof_domain_(_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC, ds_max)
#				
#				_eleGma = _recalculate_vortex_strength( _eleGma, _Triangle_Area_, _triXC, _triYC, _triZC, _no_tri_, tri_verti_)
#				
#				_leng_ = len(_nodeXC)
#				_nodeCoord = np.zeros( (_leng_,3) )
#				( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
#				Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma)
#				_nodeCirc = Obj_._change_in_Circu()



#			_bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up = \
#			_finding_boundary_nodes_(_nodeXC, _nodeYC, xmin=min(x), xmax=max(x), ymin=min(y), ymax=max(y))
			
#			if len(ind_ele_l) > 0: _bndry_ele_lt = np.concatenate( (_bndry_ele_lt, ind_ele_l) )
#			if len(ind_ele_r) > 0: _bndry_ele_rt = np.concatenate( (_bndry_ele_rt, ind_ele_r) )
#			if len(ind_ele_d) > 0: _bndry_ele_dn = np.concatenate( (_bndry_ele_up, ind_ele_d) )
#			if len(ind_ele_u) > 0: _bndry_ele_up = np.concatenate( (_bndry_ele_dn, ind_ele_u) )
			
#			_bndry_ele_lt = np.unique(_bndry_ele_lt).tolist() 
#			_bndry_ele_rt = np.unique(_bndry_ele_rt).tolist()
#			_bndry_ele_dn = np.unique(_bndry_ele_dn).tolist()
#			_bndry_ele_up = np.unique(_bndry_ele_up).tolist()

#		ind_ele_l, ind_ele_r, ind_ele_d, ind_ele_u = \
#		_max_edge_length_dection_at_boundary_ele_( _bndry_ele_lt, _bndry_ele_rt, _bndry_ele_dn, _bndry_ele_up, _tri_ele_max_edge_ )


















###############################################################


def _element_merging_below_critical_length_\
(_triXC, _triYC, _triZC, tri_verti_, _delete_tri_ele_, _edge_sorted_, _parent_nodeXC, _parent_nodeYC, _parent_nodeZC, _Triangle_Area_, _eleGma):
	
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
			( _coord1_[itr, 0], _coord1_[itr, 1], _coord1_[itr, 2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			
		if _edge_sorted_[itr] == 1: 
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			( _coord1_[itr, 0], _coord1_[itr, 1], _coord1_[itr, 2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			
		if _edge_sorted_[itr] == 2: 
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			( _coord1_[itr, 0], _coord1_[itr, 1], _coord1_[itr, 2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
	
	
	for itr in range( len(_delete_tri_ele_) ):
		
		_tri_ele_ = _delete_tri_ele_[itr]
#		print( '_tri_ele_ = ', _tri_ele_ )
		
		indx1 = np.where( _coord0_[itr,0] == _triXC )[0]
		indy1 = np.where( _coord0_[itr,1] == _triYC )[0]
		
		indx2 = np.where( _coord1_[itr,0] == _triXC )[0]
		indy2 = np.where( _coord1_[itr,1] == _triYC )[0]
		
		p0 = np.intersect1d( indx1, indy1 )
		p1 = np.intersect1d( indx2, indy2 )
		
		_comm_tri_ele_ = np.intersect1d( p0, p1 )
		
		if len(_comm_tri_ele_) != 2: 
			raise ValueError( 'two common triangles could not be found' )
		
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
		if len(p0) > 0: p.append(p0[0])
		
	_nw_nodeXC = np.delete( _nw_nodeXC, p )
	_nw_nodeYC = np.delete( _nw_nodeYC, p )
	_nw_nodeZC = np.delete( _nw_nodeZC, p )
	
#	print('no of nodes = ', len(_nw_nodeXC))
	
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
		
		if len(p0) > 0: _triVt0.append(p0[0])
		if len(p1) > 0: _triVt1.append(p1[0])
		if len(p2) > 0: _triVt2.append(p2[0])
	
	_triVT0 = np.zeros( (len(_triVt0), 3) )
	_triVT0[0:len(_triVt0),0] = _triVt0[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),1] = _triVt1[0:len(_triVt0)]
	_triVT0[0:len(_triVt0),2] = _triVt2[0:len(_triVt0)]
	
	return _nw_triXC, _nw_triYC, _nw_triZC, _triVT0, _nw_nodeXC, _nw_nodeYC, _nw_nodeZC, _nw_eleGma
























#		for jtr in range( len(p1) ):
#			_tri_ele_ = p1[jtr]
#			if _nw_triXC[_tri_ele_,0] == _coord1_[itr, 0] and _nw_triYC[_tri_ele_,0] == _coord1_[itr, 1]:
#				_nw_triXC[_tri_ele_,0] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
#				_nw_triYC[_tri_ele_,0] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
#				_nw_triZC[_tri_ele_,0] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
#			
#			if _nw_triXC[_tri_ele_,1] == _coord1_[itr, 0] and _nw_triYC[_tri_ele_,1] == _coord1_[itr, 1]:
#				_nw_triXC[_tri_ele_,1] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
#				_nw_triYC[_tri_ele_,1] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
#				_nw_triZC[_tri_ele_,1] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
#			
#			if _nw_triXC[_tri_ele_,2] == _coord1_[itr, 0] and _nw_triYC[_tri_ele_,2] == _coord1_[itr, 1]:
#				_nw_triXC[_tri_ele_,2] = 0.5*( _coord0_[itr, 0] + _coord1_[itr, 0] )
#				_nw_triYC[_tri_ele_,2] = 0.5*( _coord0_[itr, 1] + _coord1_[itr, 1] )
#				_nw_triZC[_tri_ele_,2] = 0.5*( _coord0_[itr, 2] + _coord1_[itr, 2] )
#			
#			p1C = np.array( [ _nw_triXC[_tri_ele_,0], _nw_triYC[_tri_ele_,0], _nw_triZC[_tri_ele_,0] ] )
#			p2C = np.array( [ _nw_triXC[_tri_ele_,1], _nw_triYC[_tri_ele_,1], _nw_triZC[_tri_ele_,1] ] )
#			p3C = np.array( [ _nw_triXC[_tri_ele_,2], _nw_triYC[_tri_ele_,2], _nw_triZC[_tri_ele_,2] ] )
#			
#			_Area_ = _oneTriangle_Area_( p1C, p2C, p3C )
#			_nw_eleGma[_tri_ele_] = tot_vor_/_Area_
#			tot_area_ += _Area_
		 
		
#			_nw_eleGma[_tri_ele_] = \
#			( _area1_*_eleGma[ _comm_tri_ele_[0] ] + _area2_*_eleGma[ _comm_tri_ele_[1] ] + _Triangle_Area_[_tri_ele_]*_eleGma[_tri_ele_] )/_Area_





















def _element_merging_below_critical_length_\
(_triXC, _triYC, _triZC, tri_verti_, _delete_tri_ele_, _edge_sorted_, _nodeXC, _nodeYC, _nodeZC, _Triangle_Area_, _eleGma, index, ds_min):
	
	_equal_len_ = np.array_equal( len(_delete_tri_ele_), len(_edge_sorted_) )
	
	_nw_triXC = np.copy( _triXC )
	_nw_triYC = np.copy( _triYC )
	_nw_triZC = np.copy( _triZC )
	
	_nw_eleGma = np.copy( _eleGma )
	
	if not _equal_len_: raise ValueError  ( ' mismatch between no of triangles and no of min edges ' )
	
	_num_triVertex_ = np.zeros(2)
	
	_coord0_ = np.zeros( (len(_delete_tri_ele_), 3) )
	_coord1_ = np.zeros( (len(_delete_tri_ele_), 3) )
	
	for itr in range( len(_delete_tri_ele_) ):
		_tri_ele_ = _delete_tri_ele_[itr]
		
		if _edge_sorted_[itr] == 0: 
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
			( _coord1_[itr, 0], _coord1_[itr, 1], _coord1_[itr, 2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			
		if _edge_sorted_[itr] == 1: 
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 1], _triYC[_tri_ele_, 1], _triZC[_tri_ele_, 1] )
			( _coord1_[itr, 0], _coord1_[itr, 1], _coord1_[itr, 2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			
		if _edge_sorted_[itr] == 2: 
			( _coord0_[itr, 0], _coord0_[itr, 1], _coord0_[itr, 2] ) = ( _triXC[_tri_ele_, 2], _triYC[_tri_ele_, 2], _triZC[_tri_ele_, 2] )
			( _coord1_[itr, 0], _coord1_[itr, 1], _coord1_[itr, 2] ) = ( _triXC[_tri_ele_, 0], _triYC[_tri_ele_, 0], _triZC[_tri_ele_, 0] )
	
	p0 = []
	# filtering out the repeated coordinates
	for itr in range( len(_coord0_[:,0]) ):
		indx = np.where( _coord0_[itr,0] == _coord0_[:,0] )[0]
		indy = np.where( _coord0_[itr,1] == _coord0_[:,1] )[0]
		ind = np.intersect1d(indx, indy)
		if len(ind) > 1: p0.append(ind[1])
	
	p1 = []
	for itr in range( len(_coord1_[:,0]) ):
		indx = np.where( _coord1_[itr,0] == _coord1_[:,0] )[0]
		indy = np.where( _coord1_[itr,1] == _coord1_[:,1] )[0]
		ind = np.intersect1d(indx, indy)
		if len(ind) > 1: p1.append(ind[1])
	
	_coord0_ = np.delete( _coord0_, [p0, p1], axis=0 )
	_coord1_ = np.delete( _coord1_, [p0, p1], axis=0 )
	
	_dist_ = np.zeros( len(_coord0_) )
	for itr in range( len(_coord0_) ):
		_dist_[itr] = _euclid_3d_dist( _coord0_[itr,0], _coord0_[itr,1], _coord0_[itr,2], _coord1_[itr,0], _coord1_[itr,1], _coord1_[itr,2] )
	
	p = []
	for itr in range( len(_coord0_) ):
		ind = np.where( _dist_[itr] == _dist_ )[0]
		if len(ind) > 1: p.append(ind[1])
	
	_coord0_ = np.delete( _coord0_, p, axis=0 )
	_coord1_ = np.delete( _coord1_, p, axis=0 )
	
	p0 = []
	for itr in range( len(_coord0_) ):
		for jtr in range( len(_coord1_) ):
				if itr != jtr:
					if _coord0_[itr,0] == _coord1_[jtr,0] and _coord0_[itr,1] == _coord1_[jtr,1]:
						p0.append(jtr)
	
	_coord0_ = np.delete( _coord0_, p0, axis=0 )
	_coord1_ = np.delete( _coord1_, p0, axis=0 )
	
	_child_nodeXC = []
	_child_nodeYC = []
	_child_nodeZC = []
	
	for itr in range( len(_coord0_[:,0]) ):
		_child_nodeXC.append( _coord1_[itr,0] )
		_child_nodeYC.append( _coord1_[itr,1] )
		_child_nodeZC.append( _coord1_[itr,2] )
		
	sio.savemat( 'coord0.mat', {'c0':_coord0_} )
	sio.savemat( 'coord1.mat', {'c1':_coord1_} )
	
	_tri_ele_to_delete_ = []
	
	for itr in range( len(_coord0_) ):
		
		indx1 = np.where( _coord0_[itr,0] == _triXC )[0]
		indy1 = np.where( _coord0_[itr,1] == _triYC )[0]
		
		indx2 = np.where( _coord1_[itr,0] == _triXC )[0]
		indy2 = np.where( _coord1_[itr,1] == _triYC )[0]
		
		p0 = np.intersect1d( indx1, indy1 )
		p1 = np.intersect1d( indx2, indy2 )
		
		flag = _inquire_min_edge_at_boundary_( _coord0_[itr,0], _coord0_[itr,1], _coord1_[itr,0], _coord1_[itr,1] )
		
		_comm_tri_ele_ = np.intersect1d( p0, p1 )
		
		if len(_comm_tri_ele_) != 2: raise ValueError( 'two common triangles could not be found' )
		
		_tri_ele_to_delete_.append( _comm_tri_ele_[0] )
		_tri_ele_to_delete_.append( _comm_tri_ele_[1] )
		
		_area1_ = _Triangle_Area_[ _comm_tri_ele_[0] ]
		_area2_ = _Triangle_Area_[ _comm_tri_ele_[1] ]
		
		_delete_ = []
		_len_ = len(p0)
		for i in range( len(p0) ):
			if p0[i] == _comm_tri_ele_[0]: _delete_.append(i)
			if p0[i] == _comm_tri_ele_[1]: _delete_.append(i)
		p0 = np.delete(p0, _delete_) 
#		print( 'len of p0 has been decreased by ', _len_-len(p0) )
		
		_delete_ = []
		_len_ = len(p1)
		for i in range( len(p1) ):
			if p1[i] == _comm_tri_ele_[0]: _delete_.append(i)
			if p1[i] == _comm_tri_ele_[1]: _delete_.append(i)
		p1 = np.delete(p1, _delete_)
#		print( 'len of p1 has been decreased by ', _len_-len(p1) )
		
		tot_ele_ = np.concatenate( (p0,p1) )
		tot_vor_ = 0.
		_area_ = 0.
		for jtr in range( len(tot_ele_) ):
			_tri_ele_ = tot_ele_[jtr]
			_area_ += _Triangle_Area_[_tri_ele_]
			tot_vor_ += _Triangle_Area_[_tri_ele_]*_eleGma[_tri_ele_]
		
		avg_gamma = tot_vor_/_area_
		
#		tot_area_ = 0.
		for jtr in range( len(p0) ):
			_tri_ele_ = p0[jtr]
			if _nw_triXC[_tri_ele_,0] == _coord0_[itr, 0] and _nw_triYC[_tri_ele_,0] == _coord0_[itr, 1]:
				( _nw_triXC[_tri_ele_,0], _nw_triYC[_tri_ele_,0], _nw_triZC[_tri_ele_,0] ) = \
				( _coord1_[itr,0], _coord1_[itr,1], _coord1_[itr,2] )
			
			if _nw_triXC[_tri_ele_,1] == _coord0_[itr, 0] and _nw_triYC[_tri_ele_,1] == _coord0_[itr, 1]:
				( _nw_triXC[_tri_ele_,1], _nw_triYC[_tri_ele_,1], _nw_triZC[_tri_ele_,1] ) = \
				( _coord1_[itr,0], _coord1_[itr,1], _coord1_[itr,2] )
			
			if _nw_triXC[_tri_ele_,2] == _coord0_[itr, 0] and _nw_triYC[_tri_ele_,2] == _coord0_[itr, 1]:
				( _nw_triXC[_tri_ele_,2], _nw_triYC[_tri_ele_,2], _nw_triZC[_tri_ele_,2] ) = \
				( _coord1_[itr,0], _coord1_[itr,1], _coord1_[itr,2] )
			
			p1C = np.array( [ _nw_triXC[_tri_ele_,0], _nw_triYC[_tri_ele_,0], _nw_triZC[_tri_ele_,0] ] )
			p2C = np.array( [ _nw_triXC[_tri_ele_,1], _nw_triYC[_tri_ele_,1], _nw_triZC[_tri_ele_,1] ] )
			p3C = np.array( [ _nw_triXC[_tri_ele_,2], _nw_triYC[_tri_ele_,2], _nw_triZC[_tri_ele_,2] ] )
			
			_Area_ = _oneTriangle_Area_( p1C, p2C, p3C )
			_nw_eleGma[_tri_ele_] += avg_gamma*( _Area_ - _Triangle_Area_[_tri_ele_] )
#			tot_area_ += _Area_ 
		
		
	# deleting the Lagrangian nodes
	_coord0_ = np.concatenate( (_coord0_, _coord1_) )
	_delete_ = []
	for itr in range( len(_coord0_) ):
		indx1 = np.where( _coord0_[itr,0] == _nodeXC )[0]
		indy1 = np.where( _coord0_[itr,1] == _nodeYC )[0]
		p0 = np.intersect1d( indx1, indy1 )
		_delete_.append(p0[0])
		
	_delete_ = np.unique(_delete_).tolist()
	
	_nodeXC = np.delete( _nodeXC, _delete_ )
	_nodeYC = np.delete( _nodeYC, _delete_ )
	_nodeZC = np.delete( _nodeZC, _delete_ )
	
	print( 'no of Lagrangian node has been deleted ', len(_delete_) )
	print( 'no of Lagrangian node has been introduced ', len(_child_nodeXC) )
	
	_nodeXC = np.concatenate( (_nodeXC, _child_nodeXC) )
	_nodeYC = np.concatenate( (_nodeYC, _child_nodeYC) )
	_nodeZC = np.concatenate( (_nodeZC, _child_nodeZC) )
	
	_nw_triXC  = np.delete( _nw_triXC,  _tri_ele_to_delete_, axis=0 )
	_nw_triYC  = np.delete( _nw_triYC,  _tri_ele_to_delete_, axis=0 )
	_nw_triZC  = np.delete( _nw_triZC,  _tri_ele_to_delete_, axis=0 )
	_nw_eleGma = np.delete( _nw_eleGma, _tri_ele_to_delete_, axis=0 )
	
	print( 'no of triangle after scrapping the garbage elemenst ', len(_nw_triXC[:,0]) )
	
	_triVt0 = []
	_triVt1 = []
	_triVt2 = []
	for ktr in range( len(_nw_triXC[:,0]) ):
		
		indx0 = np.where(_nw_triXC[ktr,0] == _nodeXC)[0]
		indx1 = np.where(_nw_triXC[ktr,1] == _nodeXC)[0]
		indx2 = np.where(_nw_triXC[ktr,2] == _nodeXC)[0]
		
		indy0 = np.where(_nw_triYC[ktr,0] == _nodeYC)[0]
		indy1 = np.where(_nw_triYC[ktr,1] == _nodeYC)[0]
		indy2 = np.where(_nw_triYC[ktr,2] == _nodeYC)[0]
		
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
	
	return _nw_triXC, _nw_triYC, _nw_triZC, _triVT0, _nodeXC, _nodeYC, _nodeZC, _nw_eleGma
	
	
	
	
		if _tri_ele_max_edge_ >= 0:
			
			print( 'no of triangles have max edge length = ', len(_tri_ele_max_edge_) )
			_len_ = len(_nodeXC)
			
			_nodeXC, _nodeYC, _nodeZC = \
			_merging_parent_child_nodes_(_triXC, _triYC, _triZC, _tri_ele_max_edge_, _max_edge_sorted_, _nodeXC, _nodeYC, _nodeZC)
			
			_triXC, _triYC, _triZC, tri_verti_ = \
			_creating_child_triangles_(_triXC, _triYC, _triZC, tri_verti_, _tri_ele_max_edge_, _max_edge_sorted_, _len_)
			
			tri_verti_ = tri_verti_.astype(int) 
			_no_tri_ = np.shape(tri_verti_)[0]
#			print('_no_tri_ = ', _no_tri_)
			
			_eleGma = _reassign_eleGma_on_added_tri_( _eleGma, _tri_ele_max_edge_ )
			
			_leng_ = len(_nodeXC)
			_nodeCoord = np.zeros( (_leng_,3) )
			( _nodeCoord[:,0], _nodeCoord[:,1], _nodeCoord[:,2] ) = ( _nodeXC, _nodeYC, _nodeZC )
			
			Obj_ = _node_Circulation(_nodeCoord, _triXC, _triYC, _triZC, _no_tri_, tri_verti_, _eleGma)
			_nodeCirc = Obj_._change_in_Circu()
		
		_tri_ele_min_edge_, _min_edge_sorted_ = _min_edge_length_dection_( _triXC, _triYC, _triZC, ds_min )
		
		_bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up = \
		_finding_boundary_nodes_(_nodeXC, _nodeYC, xmin=min(x), xmax=max(x), ymin=min(y), ymax=max(y))
		
		_triXC, _triYC, _triZC, _nodeXC, _nodeYC, _nodeZC = _repositioning_boundary_eles_\
		(_triXC, _triYC, _triZC, tri_verti_, _nodeXC, _nodeYC, _nodeZC, _bndry_node_lt, _bndry_node_rt, _bndry_node_dn, _bndry_node_up, max(x), max(y))
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		if indw >= 0: 
			( coord[0], coord[1], coord[2] ) = ( _nodeXC[indw], _nodeYC[indw], _nodeZC[indw] )
			p = _del_ele_contains_straddle_node_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, tri_verti_, coord)
			node_del.append(indw)
			eles_del.append(p)
		
		if inde >= 0: 
			( coord[0], coord[1], coord[2] ) = ( _nodeXC[inde], _nodeYC[inde], _nodeZC[inde] )
			p = _del_ele_contains_straddle_node_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, tri_verti_, coord)
			node_del.append(inde)
			eles_del.append(p)
		
		if inds >= 0: 
			( coord[0], coord[1], coord[2] ) = ( _nodeXC[inds], _nodeYC[inds], _nodeZC[inds] )
			p = _del_ele_contains_straddle_node_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, tri_verti_, coord)
			node_del.append(inds)
			eles_del.append(p)
		
		if indn >= 0: 
			( coord[0], coord[1], coord[2] ) = ( _nodeXC[indn], _nodeYC[indn], _nodeZC[indn] )
			p = _del_ele_contains_straddle_node_(_nodeXC, _nodeYC, _nodeZC, _triXC, _triYC, _triZC, tri_verti_, coord)
			node_del.append(indn)
			eles_del.append(p)
		
		indw, inde, inds, indn = \
		_node_outside_domain(_nodeXC, _nodeYC, node_lt, node_rt, node_dn, node_up)
		
	if len(node_del) > 0: _nodeXC, _nodeYC, _nodeZC = node_to_del(_nodeXC, _nodeYC, _nodeZC, node_del)
	if len(eles_del) > 0: _triXC, _triYC, _triZC, tri_verti_, _eleGma = eles_to_del(_triXC, _triYC, _triZC, tri_verti_, _eleGma, eles_del)
	
	if len(node_del) > 0:
		node_lt, node_rt, node_dn, node_up = \
		_rearranging_boundary_node_list_(node_lt, node_rt, node_dn, node_up, node_del)
		
		
		
		
	for itr in range( len(_bndry_node_dn) ):
		_node_ = _bndry_node_dn[itr]
		ind = np.intersect1d( np.where( _nodeXC[_node_] == _triXC )[0], np.where( _nodeYC[_node_] == _triYC )[0] )
		
		if len(ind) > 0:
			for jtr in range( len(ind) ):
				VAL = _triYC[ ind[jtr], 0:3]
				POS = [ i for i, j in enumerate(VAL) if j == _nodeYC[_node_] ]
				_triYC[ ind[jtr], POS[0] ] = 0.
		_nodeYC[ _node_ ] = 0.
