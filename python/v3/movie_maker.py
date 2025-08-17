import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LightSource

from scipy.io import loadmat

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
writer = FFMpegWriter(fps=5, metadata=metadata)

#fig = plt.figure()
#l, = plt.plot([], [], 'k-o')
#
#plt.xlim(-5, 5)
#plt.ylim(-5, 5)
#
#x0, y0 = 0, 0
#
#with writer.saving(fig, "writer_test.mp4", 100):
#	for i in range(100):
#		x0 += 0.1 * np.random.randn()
#		y0 += 0.1 * np.random.randn()
#		l.set_data(x0, y0)
#		writer.grab_frame()


fig = plt.figure()

with writer.saving(fig, "KH_1.mp4", 100):
	for itr in range(0, 300, 5):
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
		
		print('max_x = ', max_x, ' min_x = ', min_x)
		print('max_y = ', max_y, ' min_y = ', min_y)
		
		#ax = plt.plot(111, projection='3d')
		ax = fig.add_subplot(111, projection='3d')
		#ax.plot_trisurf( x1, y1, z1,  triangles=tri,  lw=0., edgecolor="grey",  antialiased=False, color="lightgrey", alpha=1.)
		ax.plot_trisurf( x1, y1, z1,  triangles=tri, lw=0.0, edgecolor="white", color='0.75', alpha=0.3 )
		ax.axis('off')
		#plt.title(' ', fontsize=18)
		#plt.rcParams.update({'font.size': 22})
		#plt.xlabel('X', fontsize=18)
		#plt.ylabel('Y', fontsize=18)
		#plt.axis([0., 1., 0., 1.])
		ax.set_zlim3d(-0.2, 0.2)   
		#plt.show()
		#fig.tight_layout()
		#plt.savefig(plt.savefig("KH%s.png" %itr , dpi=600))
		writer.grab_frame()
	




