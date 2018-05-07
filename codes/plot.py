# Title: Plotting
# Author: Nelson Ribeiro Filho / Rodrigo Bijani

import numpy
import warnings
import pylab as py
import scipy.interpolate
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def draw_prism(area, style = '--k', linewidth = 2, fill = None, alpha = 1., label = None, xy2ne= False):
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

def plota_mapa(projecao, x, y, dado, area, unidade, titulo, cores, tamanho,
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

    dado_min = np.min(dado)
    dado_max = np.max(dado)
    
    #Esquema da escala de cores
    if (dado_min*dado_max < 0.):
        ranges = np.max(np.abs([dado_min, dado_max]))
        ranges_0 = 0.
    else:
        ranges = 0.5*(dado_max - dado_min)
        ranges_0 = 0.5*(dado_max + dado_min)
    
    longitude_central = 0.5*(area[1] + area[0])
    latitude_central = 0.5*(area[3] + area[2])
    
    x_max = np.max(x)*0.001 # valor maximo de x em km
    x_min = np.min(x)*0.001 # valor minimo de x em km
    
    if escala == True:
    
        #Valor em km a ser representado na escala
        #Este valor foi estabelecido como aproximadamente 
        #40 porcento da variacao maxima em x
        comprimento_escala = np.floor(0.4*(x_max - x_min)/100.)*100.
    
        #Posicao do centro da escala em coordenadas geodesicas
        longitude_escala = area[1] - 0.25*(area[1] - area[0])
        latitude_escala = area[2] + 0.05*(area[3] - area[2])
    
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    
    plt.figure(figsize=tamanho)
    plt.title(titulo, fontsize=18, y=1.05)
    projecao.contourf(x, y, dado, 100, tri=True, cmap=plt.get_cmap(cores),
                      vmin = -ranges + ranges_0, vmax = ranges + ranges_0)
    plt.colorbar(orientation='horizontal', pad=0.04, aspect=50, 
                 shrink=0.7).set_label(unidade, fontsize=18)
    projecao.drawcoastlines()
    if (np.ceil(area[2]) == area[2]):
        parallels = np.arange(np.ceil(area[2]) + 1., area[3], delta)
    else:
        parallels = np.arange(np.ceil(area[2]), area[3], delta)
    if (np.ceil(area[0]) == area[0]):
        meridians = np.arange(np.ceil(area[0]) + 1., area[1], delta)
    else:
        meridians = np.arange(np.ceil(area[0]), area[1], delta)
    if eixos == True:
        projecao.drawparallels(parallels, labels=[1,1,0,0])
        projecao.drawmeridians(meridians, labels=[0,0,1,1])
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
    plt.show()
