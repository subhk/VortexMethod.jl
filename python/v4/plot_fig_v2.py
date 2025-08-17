import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LightSource
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as FF

from scipy.io import loadmat

itr = 98

x = loadmat('nodex%s.mat' %itr)['x']
y = loadmat('nodey%s.mat' %itr)['y']
z = loadmat('nodez%s.mat' %itr)['z']
#uz = loadmat('uz%s.mat' %itr)['uz']
tri = loadmat('tri%s.mat' %itr)['tri']

x1 = np.zeros( x.shape[1] )
y1 = np.zeros( x.shape[1] )
z1 = np.zeros( x.shape[1] )

#uz1 = np.zeros( uz.shape[1] )

for i in range( x.shape[1] ):
	x1[i] = x[0,i] 
	y1[i] = y[0,i]
	z1[i] = z[0,i] 
	
#	uz1[i] = uz[0,i] 
	
( max_x, min_x ) = ( np.max(x1), np.min(x1) )
( max_y, min_y ) = ( np.max(y1), np.min(y1) )
( max_z, min_z ) = ( np.max(z1), np.min(z1) )

#( max_u, min_u ) = ( np.max(uz1), np.min(uz1) )

print('max_x = ', max_x, ' min_x = ', min_x)
print('max_y = ', max_y, ' min_y = ', min_y)
print('max_z = ', max_z, ' min_z = ', min_z)
#print('max_u = ', max_u, ' min_u = ', min_u)

#fig = plt.figure(num=None, figsize=(6, 6), dpi=300, facecolor='w', edgecolor='w')
fig = plt.figure(num=None)
fig.set_size_inches(11.5, 8.5)
ax = fig.add_subplot(111, projection='3d')

#ax = plt.plot(111, projection='3d')

# Create light source object.
ls = LightSource(azdeg=0, altdeg=65)
# Shade data, creating an rgb array.
#rgb = ls.shade(z1, plt.cm.RdYlBu)
 
ax.plot_trisurf( x1, y1, z1,  triangles=tri,  lw=0.2, edgecolor="black",  antialiased=True, color="darkgray", alpha=0.5, shade=True )
#ax.plot_trisurf( x1, y1, z1,  triangles=tri, lw=0., edgecolor="darkgray", antialiased=True,  color="darkgray", alpha=0.6) #, shade=True )
#ax.plot_trisurf( x1, y1, z1,  triangles=tri, lw=0.01, edgecolor="black", color='black', alpha=0.4 )
#ax.plot_trisurf( x1, y1, z1,  triangles=tri,  lw=0., edgecolor="grey",  antialiased=False, color="lightgrey", alpha=0.8)
ax.axis('off')
#plt.title(' ', fontsize=18)
#plt.rcParams.update({'font.size': 10})
#plt.xlabel('X', fontsize=10)
#plt.ylabel('Y', fontsize=10)
plt.axis([0., 1., 0., 1.])
ax.set_zlim3d(-2.*max_z, 2.*max_z)
plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
ax.view_init(30, 20)
#ax.view_init(90, 0)
plt.show()
#fig.tight_layout()
#plt.savefig(plt.savefig("KH.png", dpi=600))


#fig1 = FF.create_trisurf(x1, y1, z1, colormap=['rgb(255, 155, 120)', 'rgb(255, 153, 255)', ], show_colorbar=True, simplices=tri, \
#showbackground=False, gridcolor='rgb(255, 20, 160)', plot_edges=False, aspectratio=dict(x=1, y=1, z=0.75))

#py.iplot(fig1)
