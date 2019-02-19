# Title: Plotting
# Author: Nelson Ribeiro Filho / Rodrigo Bijani

from __future__ import division, absolute_import
import numpy
import warnings
import pylab as py
import scipy.interpolate
from matplotlib import pyplot
from datetime import datetime
from matplotlib import rcParams
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap, addcyclic
from scipy.ndimage.filters import minimum_filter, maximum_filter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from future.builtins import range
import warnings
import numpy
from matplotlib import pyplot, widgets

def draw_prism(area, style = '--k', linewidth = 2., fill = None, alpha = 1., label = None, xy2ne= False):
    '''
    Plot the rectangle-square on a 2D contour map.
    
    Inputs:
    area - numpy list - edges of the square
    '''
    # Define the area
    x1, x2, y1, y2 = area
    # Inverts xy in map
    if xy2ne:
        x1, x2, y1, y2 = y1, y2, x1, x2
    xs = [x1, x1, x2, x2, x1]
    ys = [y1, y2, y2, y1, y1]
    # Define the label
    if label is not None:
        kwargs['label'] = label
    plot = pyplot.plot(xs, ys, style, linewidth)
    # Define if square is fill with color
    if fill is not None:
        pyplot.fill(xs, ys, color=fill, alpha=alpha)
    # Return final output
    return plot

def drawcontour(xdata, ydata, zdata, shape, levels, interp=False, extrapolate=False, color='k',
            label=None, clabel=True, style='solid', linewidth=1.0, basemap=None):
    '''
    Make a contour plot of the data.
    '''
    if style not in ['solid', 'dashed', 'mixed']:
        raise ValueError("Invalid contour style %s" % (style))
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = gridder.interp(x, y, v, shape, extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(colors=color, picker=True)
    if basemap is None:
        ct_data = pyplot.contour(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contour(lon, lat, V, levels, **kwargs)
    if clabel:
        ct_data.clabel(fmt='%g')
    if label is not None:
        ct_data.collections[0].set_label(label)
    if style != 'mixed':
        for c in ct_data.collections:
            c.set_linestyle(style)
    for c in ct_data.collections:
        c.set_linewidth(linewidth)
    return ct_data.levels

def drawcontourf(xdata, ydata, zdata, shape, level, unit, figurename = 'contourmap', 
                 vmin=None, vmax=None, cmap = pyplot.cm.jet):
    '''
    Make a filled contour plot of the data.
    '''
    if xdata.shape != ydata.shape != zdata.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    kwargs = dict(vmin = vmin, vmax = vmax, cmap = cmap, picker = True)
    contourmap = pyplot.contourf(numpy.reshape(x, shape), numpy.reshape(y, shape), 
                                 numpy.reshape(v, shape), levels, **kwargs)
    pyplot.xlim(x.min(), x.max())
    pyplot.ylim(y.min(), y.max())
       
    cb = pyplot.colorbar(orientation = 'vertical')
    cb.set_ticks(numpy.linspace(int(vmin), int(vmax), 10))
    cb.set_label('nT')
    
    pyplot.savefig(figurename, format = png, dpi = 300,  bbox_inches = 'tight')
    pyplot.savefig(figurename, format = pdf, dpi = 300,  bbox_inches = 'tight')
    #pyplot.title(figuretitle)
    return contourmap

def contourf(x, y, v, shape, levels, interp=False, extrapolate=False,
             vmin=None, vmax=None, cmap=pyplot.cm.jet, basemap=None):
    '''
    Make a filled contour plot of the data.
    '''
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")

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

def plota_mapa(projecao, x, y, dado, area, unidade, titulo, titlesize, cores, tamanho,
               delta, perfis=None, estados=None, escala=None, eixos=None):
    '''
    Plota um mapa dos dados "dado", com coordenadas "x" e 
    "y" referidas a uma determinada projecao cartografica 
    "projecao". "unidade" e "titulo" sao, respectivamente,
    a unidade dos dados e o titulo do mapa.
    
    input
    
    projecao: objeto do mpl_toolkits.basemap.Basemap - 
              projecao cartografica.
    x: numpy array 1D - vetor com as coordenadas x
       da projecao.
    y: numpy array 1D - vetor com as coordenadas y
       da projecao.
    dado: numpy array 1D - vetor com os dados a serem
          plotados.
    area: list - lista com os valores minimo e maximo da
          longitude e minimo e maximo da latitude, em
          graus.
    unidade: string - unidade dos dados a serem plotados.
    titulo: string -  titulo do mapa.
    cores: codigo do colormaps_reference.py - esquema de 
           cores do mapa.
    tamanho: tuple - define os tamanhos tx e ty do mapa ao longo,
             repectivamente, dos eixos x e y. Os valores tx e ty
             sao passados entre parenteses e separados por virgula
             (tx, ty).
    delta: float - intervalo, em graus, entre os meridiando e paralelos.
    perfis: numpy array 2D - matriz com as coordenadas x e y
            dos pontos iniciais e finais dos perfis a serem 
            analisados. As coordenadas x e y estao na primeira
            e segunda colunas, respectivamente. As primeiras
            duas linhas contem os dois pontos que formam o 
            primeiro perfil, as proximas duas contem os pontos
            que formam o segundo perfil e assim sucessivamente.
    estados: boolean - se for igual a True, plota o contorno dos estados.
    escala: boolean - se for igual a True, plota a escala do mapa.
    eixos: boolean - se for igual a True, plota os eixos do mapa.
    
    output
    
    mapa: string - codigo de uma matplotlib.figure.Figure.
    '''

    dado_min = numpy.min(dado)
    dado_max = numpy.max(dado)
    
    #Esquema da escala de cores
    #if (dado_min*dado_max < 0.):
    #    ranges = numpy.max(numpy.abs([dado_min, dado_max]))
    #    ranges_0 = 0.
    #else:
    #    ranges = 0.5*(dado_max - dado_min)
    #    ranges_0 = 0.5*(dado_max + dado_min)
    
    longitude_central = 0.5*(area[1] + area[0])
    latitude_central = 0.5*(area[3] + area[2])
    
    x_max = numpy.max(x)*0.001 # valor maximo de x em km
    x_min = numpy.min(x)*0.001 # valor minimo de x em km
    
    if escala == True:
    
        #Valor em km a ser representado na escala
        #Este valor foi estabelecido como aproximadamente 
        #40 porcento da variacao maxima em x
        comprimento_escala = numpy.floor(0.4*(x_max - x_min)/100.)*100.
    
        #Posicao do centro da escala em coordenadas geodesicas
        longitude_escala = area[1] - 0.25*(area[1] - area[0])
        latitude_escala = area[2] + 0.05*(area[3] - area[2])
    
    x_min = numpy.min(x)
    x_max = numpy.max(x)
    y_min = numpy.min(y)
    y_max = numpy.max(y)
    
    pyplot.figure(figsize = tamanho)
    pyplot.title(titulo, fontsize = titlesize, y = 1.0)
    projecao.contourf(x, y, dado, 50, tri = True, cmap = pyplot.get_cmap(cores),
                      vmin = dado.min(), vmax = dado.max())
    #projecao.contourf(x, y, dado, 50, tri = True, cmap = pyplot.get_cmap(cores), 
    #vmin = -ranges + ranges_0, vmax = ranges + ranges_0)
    pyplot.colorbar(orientation = 'vertical', pad = 0.04, aspect = 50, 
                 shrink = 0.7).set_label(unidade, fontsize = 16)
    projecao.drawcoastlines()
    if (numpy.ceil(area[2]) == area[2]):
        parallels = numpy.arange(numpy.ceil(area[2]) + 1., area[3], delta)
    else:
        parallels = numpy.arange(numpy.ceil(area[2]), area[3], delta)
    if (numpy.ceil(area[0]) == area[0]):
        meridians = numpy.arange(numpy.ceil(area[0]) + 1., area[1], delta)
    else:
        meridians = numpy.arange(numpy.ceil(area[0]), area[1], delta)
    if eixos == True:
        projecao.drawparallels(parallels, labels=[1,0,0,0])
        projecao.drawmeridians(meridians, labels=[0,0,0,1])
    else:
        projecao.drawparallels(parallels)
        projecao.drawmeridians(meridians)
    if estados == True:
        projecao.drawstates()
    if perfis != None:
        for i in range(0,perfis.shape[0],2):
            projecao.plot(perfis[i:i+2,0], perfis[i:i+2,1], 'o-k', linewidth=2)
    if escala == True:
        projecao.drawmapscale(longitude_escala, latitude_escala,
                            longitude_central, latitude_central,
                            length=comprimento_escala, barstyle='fancy')
    
    #pyplot.savefig('figure_plota_mapa.png', dpi = 300, bbox_inches='tight')
    #pyplot.savefig('figure_plota_mapa.pdf', dpi = 300, bbox_inches='tight')
    pyplot.show()
    
def basemap(area, projection, resolution='c'):
    """
    Make a basemap to use when plotting with map projections.
    Uses the matplotlib basemap toolkit.
    Parameters:
    * area : list
        ``[west, east, south, north]``, i.e., the area of the data that is
        going to be plotted
    * projection : str
        The name of the projection you want to use. Choose from:
        * 'ortho': Orthographic
        * 'geos': Geostationary
        * 'robin': Robinson
        * 'cass': Cassini
        * 'merc': Mercator
        * 'poly': Polyconic
        * 'lcc': Lambert Conformal
        * 'stere': Stereographic
    * resolution : str
        The resolution for the coastlines. Can be 'c' for crude, 'l' for low,
        'i' for intermediate, 'h' for high
    Returns:
    * basemap : mpl_toolkits.basemap.Basemap
        The basemap
    """
    if projection not in ['ortho', 'aeqd', 'geos', 'robin', 'cass', 'merc',
                          'poly', 'lcc', 'stere']:
        raise ValueError("Unsuported projection '%s'" % (projection))
    global Basemap
    if Basemap is None:
        try:
            from mpl_toolkits.basemap import Basemap
        except ImportError:
            raise
    west, east, south, north = area
    lon_0 = 0.5 * (east + west)
    lat_0 = 0.5 * (north + south)
    if projection == 'ortho':
        bm = Basemap(projection=projection, lon_0=lon_0, lat_0=lat_0,
                     resolution=resolution)
    elif projection == 'geos' or projection == 'robin':
        bm = Basemap(projection=projection, lon_0=lon_0, resolution=resolution)
    elif (projection == 'cass' or
          projection == 'poly'):
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, resolution=resolution)
    elif projection == 'merc':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_ts=lat_0,
                     resolution=resolution)
    elif projection == 'lcc':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, rsphere=(6378137.00, 6356752.3142),
                     lat_1=lat_0, resolution=resolution)
    elif projection == 'stere':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, lat_ts=lat_0, resolution=resolution)
    return bm
    
def savefig(fname, magnification=None):
    """
    Save a snapshot the current Mayavi figure to a file.
    Parameters:
    * fname : str
        The name of the file. The format is deduced from the extension.
    * magnification : int or None
        If not None, then the scaling between the pixels on the screen, and the
        pixels in the file saved.
    """
    _lazy_import_mlab()
    if magnification is None:
        mlab.savefig(fname)
    else:
        mlab.savefig(fname, magnification=magnification)
