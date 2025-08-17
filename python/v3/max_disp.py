import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LightSource

from scipy.io import loadmat
import scipy.io as sio

z_max = []
for itr in range(0, 155, 5):
	print('itr = ', itr)
	x = loadmat('nodex%s.mat' %itr)['x'] 
	y = loadmat('nodey%s.mat' %itr)['y']
	z = loadmat('nodez%s.mat' %itr)['z']
	tri = loadmat('tri%s.mat' %itr)['tri']
	
	x1 = np.zeros( x.shape[1] )
	y1 = np.zeros( x.shape[1] )
	z1 = np.zeros( x.shape[1] )
	
	x1[0:x.shape[1]] = x[0, 0:x.shape[1]] 
	y1[0:x.shape[1]] = y[0, 0:x.shape[1]]
	z1[0:x.shape[1]] = z[0, 0:x.shape[1]] 
	
	max_x = np.max(x1)
	min_x = np.min(x1)
	
	max_y = np.max(y1)
	min_y = np.min(y1)
	
	max_z = np.max(z)
	min_z = np.min(z)
	
	z_max.append(max_z)
	
	print('max_x = ', max_x, ' min_x = ', min_x)
	print('max_y = ', max_y, ' min_y = ', min_y)
	print('max_y = ', max_z, ' min_y = ', min_z)
	

sio.savemat( 'z_max.mat', {'zmax':z_max} )



