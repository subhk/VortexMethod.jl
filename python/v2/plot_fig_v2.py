import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LightSource

from scipy.io import loadmat

itr = 200

x = loadmat('nodex%s.mat' %itr)['x']
y = loadmat('nodey%s.mat' %itr)['y']
z = loadmat('nodez%s.mat' %itr)['z']
uz = loadmat('ux.mat')['ux']
tri = loadmat('tri%s.mat' %itr)['tri']

x1 = np.zeros( x.shape[1] )
y1 = np.zeros( x.shape[1] )
z1 = np.zeros( x.shape[1] )

uz1 = np.zeros( uz.shape[1] )

for i in range( x.shape[1] ):
	x1[i] = x[0,i] 
	y1[i] = y[0,i]
	z1[i] = z[0,i] 
	
	uz1[i] = uz[0,i] 
	
( max_x, min_x ) = ( np.max(x1), np.min(x1) )
( max_y, min_y ) = ( np.max(y1), np.min(y1) )
( max_z, min_z ) = ( np.max(z1), np.min(z1) )

print('max_x = ', max_x, ' min_x = ', min_x)
print('max_y = ', max_y, ' min_y = ', min_y)
print('max_z = ', max_z, ' min_z = ', min_z)

fig = plt.figure(num=None, figsize=(3, 6), dpi=500, facecolor='w', edgecolor='w')
fig.set_size_inches(11.5, 8.5)
ax = fig.add_subplot(111, projection='3d')

#ax = plt.plot(111, projection='3d')

# Create light source object.
ls = LightSource(azdeg=0, altdeg=65)
# Shade data, creating an rgb array.
#rgb = ls.shade(z, plt.cm.RdYlBu)
 
ax.plot_trisurf( x1, y1, z1,  triangles=tri,  lw=0.1, edgecolor="black",  antialiased=True, color="lightgrey", alpha=0.4)
#ax.plot_trisurf( x1, y1, z1,  triangles=tri, lw=None, linestyle='None', color='black', alpha=0.5 )
ax.view_init(34, -53)
ax.axis('off')
#plt.title(' ', fontsize=18)
#plt.rcParams.update({'font.size': 10})
#plt.xlabel('X', fontsize=10)
#plt.ylabel('Y', fontsize=10)
#plt.axis([0., 1., 0., 1.])
ax.set_zlim3d(-0.1, 0.1)   
plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
plt.show()
#fig.tight_layout()
#plt.savefig(plt.savefig("KH.png", dpi=600))

