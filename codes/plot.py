# Title: Plotting
# Author: Nelson Ribeiro Filho / Rodrigo Bijani
import numpy
import warnings
import pylab as py
import scipy.interpolate
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def draw_prism(area, style='--k', linewidth=2, fill=None, alpha=1., label=None,
           xy2ne=False):
    """
    Plot a square.

    Parameters:

    * area : list = [x1, x2, y1, y2]
        Borders of the square
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
    if xy2ne:
        x1, x2, y1, y2 = y1, y2, x1, x2
    xs = [x1, x1, x2, x2, x1]
    ys = [y1, y2, y2, y1, y1]
    kwargs = {'linewidth': linewidth}
    if label is not None:
        kwargs['label'] = label
    plot = pyplot.plot(xs, ys, style, **kwargs)
    if fill is not None:
        pyplot.fill(xs, ys, color=fill, alpha=alpha)
    return plot

def draw_contourf(x, y, v, levels, interp=False, extrapolate=False,
             vmin=None, vmax=None, cmap = pyplot.cm.jet, basemap=None):
    """
    Make a filled contour plot of the data.
    Parameters:
    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * levels : int or list
        Number of contours to use or a list with the contour values.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * vmin, vmax
        Saturation values of the colorbar. If provided, will overwrite what is
        set by *levels*.
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~fatiando.vis.mpl.basemap` for creating basemaps)
    Returns:
    * levels : list
        List with the values of the contour levels
    """
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = gridder.interp(x, y, v, shape, extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap, picker=True)
    if basemap is None:
        ct_data = pyplot.contourf(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contourf(lon, lat, V, levels, **kwargs)
    return ct_data.levels

####################################################################################################################################

def prism3D(fig, prism, color, theta, phi, area, xlabel, ylabel, zlabel ,title):
   # plot a 3D prism with a specific color:
   # fig: the figure box defined outside of the function, usually as follows (copy paste the following 2 lines):
   # py.rcParams['figure.figsize'] = (15.0, 10.0) #Redimensiona a figura
   #fig = plt.figure()
   
   # inputs: model = list with the corners of the prism to be drawn
   # color = string that indicates the color to paint the edges of the prism
   # color can be: ('black', 'red', 'blue', 'yellow', 'green')
   # phi, theta = integers to define the angle and azimuth of the box plot;
   # xlabel, ylabel, zlabel = strings with the label id:  
   # output: the plot of the 3D prism

    ax = fig.add_subplot(111, projection='3d')
    fs = 15
   
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

    
    x0, x1, y0, y1, z0, z1 = area
   
        # get the corners of the prism:    
    r = np.array(prism)
    rx = r[0:2] # x corners
    ry = r[2:4] # y corners
    rz = r[4:6] # z corners
    
    for s, e in combinations(np.array(list(product(rx,ry,rz))), 2):
        
        if np.sum(np.abs(s-e)) == ry[1]-ry[0]:
            ax.plot3D(*zip(s,e), color=c)
            ax.set_xlim3d(x0, x1)
            ax.set_ylim3d(y0, y1)
            ax.set_zlim3d(z0, z1)
            ax.invert_zaxis()    
      
        if np.sum(np.abs(s-e)) == rx[1]-rx[0]:
            ax.plot3D(*zip(s,e), color=c)
            ax.set_xlim3d(x0, x1)
            ax.set_ylim3d(y0, y1)
            ax.set_zlim3d(z0, z1)
            ax.invert_zaxis()

        if np.sum(np.abs(s-e)) == rz[1]-rz[0]:
            ax.plot3D(*zip(s,e), color=c)
            ax.set_xlim3d(x0, x1)
            ax.set_ylim3d(y0, y1)
            ax.set_zlim3d(z0, z1)
            ax.invert_zaxis() 
    
    
    ax.set_xlabel(xlabel, labelpad=25 ,fontsize=fs) # labelpad = distance between the label and the axis
    ax.set_ylabel(ylabel, labelpad=25 ,fontsize=fs)
    ax.set_zlabel(zlabel, labelpad=25,fontsize=fs)
    ax.set_title(title, fontsize=fs + 3 )
    ax.view_init(theta, phi)
      
    # set labelsize 
    pyplot.tick_params(axis='y', labelsize=fs-3)
    pyplot.tick_params(axis='x', labelsize=fs-3)
    pyplot.tick_params(axis='z', labelsize=fs-3)
          
    return plt.show()
