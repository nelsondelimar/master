# --------------------------------------------------------------------------------------------------
# Title: Grav-Mag Codes
# Author: Rodrigo Bijani and Victor Carreira
# Description: Source codes for plotting images.
# --------------------------------------------------------------------------------------------------

# Import Python libraries:
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab as py
from itertools import product, combinations

# read file with the coordinates of all point masses
def points3D(points, theta, phi, xlabel, ylabel, zlabel, model):
    
    # function to plot 3D points in space:
    # inputs: points = list containing the x, y ,z and the density of the points. The color of the point is related to the value of density
    # phi, theta = integers to define the angle and azimuth of the box plot;
    # xlabel, ylabel, zlabel = strings with the label id:  
    # color: color of the points: ('black', 'red', 'green', 'blue', 'yellow')
    # output: figure with all points
       
    
    # transform the list points into array:
    f = np.array(points)
    nf = np.size(f,0)
    
    py.rcParams['figure.figsize'] = (15.0, 10.0) #Redimensiona a figura
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # set the properties of each point:
    dens = f[:,4] # density
    x = f[:,0] # x coordinate (center of the sphere)
    y = f[:,1] # y coordinate (center of the sphere)
    z = f[:,2] # z coordinate (center of the sphere)
    p = ax.scatter(x, y, z, s=80, c = dens, depthshade=True )
    cbar = fig.colorbar(p, aspect=50, fraction = 0.03, orientation="vertical")
    cbar.set_label('density ( $ kg/m^{3}$ )',fontsize=10, rotation = 90)
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_zlabel(zlabel,fontsize=12)
    ax.set_title(model, fontsize=17)
    ax.view_init(theta, phi)
    return plt.show()
####################################################################################################################################

def prism3D(prism, theta, phi, xlabel, ylabel, zlabel ,color, model):
   # plot a 3D prism with a specific color:
   # inputs: model = list with the corners of the prism to be drawn
   # color = string that indicates the color to paint the edges of the prism
   # color can be: ('black', 'red', 'blue', 'yellow', 'green')
   # phi, theta = integers to define the angle and azimuth of the box plot;
   # xlabel, ylabel, zlabel = strings with the label id:  
   # output: the plot of the 3D prism

    fs = 12 # font size 
    py.rcParams['figure.figsize'] = (12.0, 10.0) #Redimensiona a figura
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")

    if color=="black":
        c = "k"
    if color=="red":
        c = "r"
    if color=="blue":
        c = "b"
    if color=="yellow":
        c = "y"
    if color =="green":
        c = "g"

    # get the corners of the prism:    
    r = np.array(prism)
    rx = r[0:2] # x corners
    ry = r[2:4] # y corners
    rz = r[4:6] # z corners
    
   # get ranges for plot the prism:
    x = np.zeros( (2) )
    y = np.zeros( (2) )
    z = np.zeros( (2) )
   # limits over all axis for plotting:
    xmin = r[0]
    ymin = r[2]
    zmin = r[4]
    xmax = r[1]
    ymax = r[3]
    zmax = r[5]
   # 50% extra for plotting the prism:
    x[0] = xmin - (0.5 * np.absolute( xmin) )
    x[1] = xmax + (0.5 * np.absolute( xmax) ) 
    y[0] = ymin - (0.5 * np.absolute( ymin) )
    y[1] = ymax + (0.5 * np.absolute( ymax) )
    z[0] = 0.0 
    z[1] = zmax + (2.0 * np.absolute( zmax ) )
   # print x, y, z

    for s, e in combinations(np.array(list(product(rx,ry,rz))), 2):
        
        if np.sum(np.abs(s-e)) == ry[1]-ry[0]:
            ax.plot3D(*zip(s,e), color=c)
            ax.set_xlim3d(x[0], x[1])
            ax.set_ylim3d(y[0], y[1])
            ax.set_zlim3d(z[0], z[1])
            plt.gca().invert_zaxis()    
      
        if np.sum(np.abs(s-e)) == rx[1]-rx[0]:
            ax.plot3D(*zip(s,e), color=c)
            ax.set_xlim3d(x[0], x[1])
            ax.set_ylim3d(y[0], y[1])
            ax.set_zlim3d(z[0], z[1])
            plt.gca().invert_zaxis()

        if np.sum(np.abs(s-e)) == rz[1]-rz[0]:
            ax.plot3D(*zip(s,e), color=c)
            ax.set_xlim3d(x[0], x[1])
            ax.set_ylim3d(y[0], y[1])
            ax.set_zlim3d(z[0], z[1])
            plt.gca().invert_zaxis()

    # set labelsize 
    plt.tick_params(axis='y', labelsize=fs)
    plt.tick_params(axis='x', labelsize=fs)
    plt.tick_params(axis='z', labelsize=fs)
       
    ax.set_xlabel(xlabel,fontsize=fs)
    ax.set_ylabel(ylabel,fontsize=fs)
    ax.set_zlabel(zlabel,fontsize=fs)
    ax.set_title(model, fontsize=fs + 3 )
    ax.view_init(theta, phi)
        
    return plt.show()

####################################################################################################################################

def rectangle(area, style='-k', linewidth=1, fill=None, alpha=1., label=None):
    """
    Plot a rectangle.

    Parameters:

    * area : list = [x1, x2, y1, y2]
        Borders of the rectangle
    * style : str
        String with the color and line style (as in matplotlib.pyplot.plot)
    * linewidth : float
        Line width
    * fill : str
        A color string used to fill the square. If None, the square is not
        filled
    * alpha : float
        Transparency of the fill (1 >= alpha >= 0). 0 is transparent and 1 is
        opaque
    * label : str
        label associated with the square.
    * xy2ne : True or False
        If True, will exchange the x and y axis so that the x coordinates of
        the polygon are north. Use this when drawing on a map viewed from
        above. If the y-axis of the plot is supposed to be z (depth), then use
        ``xy2ne=False``.

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    x1, x2, y1, y2 = area
   # if xy2ne:
   #     x1, x2, y1, y2 = y1, y2, x1, x2
    xs = [x1, x1, x2, x2, x1]
    ys = [y1, y2, y2, y1, y1]
    kwargs = {'linewidth': linewidth}
    if label is not None:
        kwargs['label'] = label
    plot, = plt.plot(xs, ys, style, **kwargs)
    if fill is not None:
        plt.fill(xs, ys, color=fill, alpha=alpha)
    return plot
    
    
