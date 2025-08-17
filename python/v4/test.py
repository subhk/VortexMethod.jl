import plotly.plotly as ply
import plotly.figure_factory as FF
#import plotly.graph_objs as go

import numpy as np
from scipy.spatial import Delaunay

u = np.linspace(-np.pi, np.pi, 30)
v = np.linspace(-np.pi, np.pi, 30)
u, v = np.meshgrid(u,v)
u = u.flatten()
v = v.flatten()

x = u
y = u*np.cos(v)
z = u*np.sin(v)

points2D = np.vstack([u,v]).T
tri = Delaunay(points2D)
simplices = tri.simplices

# define a function for the color assignment
def dist_from_x_axis(x, y, z):
	return x

fig1 = FF.create_trisurf(x=x, y=y, z=z, simplices=simplices) #, \
#simplices=simplices, title="Light Cone", showbackground=False, gridcolor='rgb(255, 20, 160)', plot_edges=False, aspectratio=dict(x=1, y=1, z=0.75))
ply.iplot(fig1) #, filename="Light Cone")
