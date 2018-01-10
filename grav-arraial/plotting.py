import numpy as np
import matplotlib.pyplot as plt

def statistical(data, unit=None):
    
    '''
    It calculates the minimum and maximum values for a simple dataset and also
    its mean values and variations.
    
    Input:
    data - numpy array 1D - data set as a vector
    unit - string - data unit
    
    Outputs:
    datamin - float - 
    dado: numpy array 1D - vetor de dados.
    unidade: string - unidade dos dados.
    
    output
    
    minimo: float - minimo dos dados.
    media: float - media dos dados.
    maximo: float - maximo dos dados.
    variacao: float - diferenca entre o maximo e o
              minimo dos dados.
    '''
    
    assert data.size > 1, 'O vetor de dados deve ter mais de um elemento'
    
    datamin = np.min(data)
    datamax = np.max(data)
    datamed = np.mean(data)
    datavar = datamax - datamin
    
    if (unit != None):
        print 'Minimum:    %5.4f' % datamin, unit
        print 'Maximum:    %5.4f' % datamax, unit
        print 'Mean value: %5.4f' % datamed, unit
        print 'Variation:  %5.4f' % datavar, unit
    else:
        print 'Minimum:    %5.4f' % datamin
        print 'Maximum:    %5.4f' % datamax
        print 'Mean value: %5.4f' % datamed
        print 'Variation:  %5.4f' % datavar
        
    return datamin, datamax, datamed, datavar

def drawstates(ax, shapefile = 'states_brazil/BRA_adm0.shp'):
    shp = m.readshapefile(shapefile, 'states', drawbounds=True)
    for nshape, seg in enumerate(m.states):
        poly = Polygon(seg, facecolor='0.75', edgecolor='k')
        ax.add_patch(poly)
    
def plotting(projection, x, y, data, area, unit, title, colors, mapsize,
             delta, profiles=None, states=None, scale=None, axis=None):
    '''
    Plota um mapa dos datas "data", com coordenadas "x" e 
    "y" referidas a uma determinada projecao cartografica 
    "projecao". "unit" e "title" sao, respectivamente,
    a unit dos datas e o title do mapa.
    
    input
    
    projecao: objeto do mpl_toolkits.basemap.Basemap - 
              projecao cartografica.
    x: numpy array 1D - vetor com as coordenadas x
       da projecao.
    y: numpy array 1D - vetor com as coordenadas y
       da projecao.
    data: numpy array 1D - vetor com os datas a serem
          plotados.
    area: list - lista com os valores minimo e maximo da
          longitude e minimo e maximo da latitude, em
          graus.
    unit: string - unit dos datas a serem plotados.
    title: string -  title do mapa.
    colors: codigo do colormaps_reference.py - esquema de 
           colors do mapa.
    mapsize: tuple - define os mapsizes tx e ty do mapa ao longo,
             repectivamente, dos axis x e y. Os valores tx e ty
             sao passados entre parenteses e separados por virgula
             (tx, ty).
    delta: float - intervalo, em graus, entre os meridiando e paralelos.
    profiles: numpy array 2D - matriz com as coordenadas x e y
            dos pontos iniciais e finais dos profiles a serem 
            analisados. As coordenadas x e y estao na primeira
            e segunda colunas, respectivamente. As primeiras
            duas linhas contem os dois pontos que formam o 
            primeiro perfil, as proximas duas contem os pontos
            que formam o segundo perfil e assim sucessivamente.
    states: boolean - se for igual a True, plota o contorno dos states.
    scale: boolean - se for igual a True, plota a scale do mapa.
    axis: boolean - se for igual a True, plota os axis do mapa.
    
    output
    
    mapa: string - codigo de uma matplotlib.figure.Figure.
    '''

    data_min = np.min(data)
    data_max = np.max(data)
    
    #Esquema da scale de colors
    if (data_min*data_max < 0.):
        ranges = np.max(np.abs([data_min, data_max]))
        ranges_0 = 0.
    else:
        ranges = 0.5*(data_max - data_min)
        ranges_0 = 0.5*(data_max + data_min)
    
    longitude_central = 0.5*(area[1] + area[0])
    latitude_central = 0.5*(area[3] + area[2])
    
    x_max = np.max(x)*0.001 # valor maximo de x em km
    x_min = np.min(x)*0.001 # valor minimo de x em km
    
    if scale == True:
    
        #Valor em km a ser representado na scale
        #Este valor foi estabelecido como aproximadamente 
        #40 porcento da variacao maxima em x
        comprimento_scale = np.floor(0.4*(x_max - x_min)/100.)*100.
    
        #Posicao do centro da scale em coordenadas geodesicas
        longitude_scale = area[1] - 0.25*(area[1] - area[0])
        latitude_scale = area[2] + 0.05*(area[3] - area[2])
    
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    
    plt.figure(figsize=mapsize)
    plt.title(title, fontsize=18, y=1.05)
    projection.contourf(x, y, data, 100, tri=True, cmap=plt.get_cmap(colors),
                        vmin = -ranges + ranges_0, vmax = ranges + ranges_0)
    plt.colorbar(orientation='horizontal', pad=0.04, aspect=50,
                 shrink=0.7).set_label(unit, fontsize=18)
    projection.drawcoastlines()
    if (np.ceil(area[2]) == area[2]):
        parallels = np.arange(np.ceil(area[2]), area[3], delta)
    else:
        parallels = np.arange(np.ceil(area[2]), area[3], delta)
    if (np.ceil(area[0]) == area[0]):
        meridians = np.arange(np.ceil(area[0]), area[1], delta)
    else:
        meridians = np.arange(np.ceil(area[0]), area[1], delta)
    if axis == True:
        projection.drawparallels(parallels, labels=[1,0,0,0])
        projection.drawmeridians(meridians, labels=[0,0,0,1])
    else:
        projection.drawparallels(parallels)
        projection.drawmeridians(meridians)
    if states == True:
        projection.drawstates()
    if profiles != None:
        for i in range(0,profiles.shape[0],2):
            projection.plot(profiles[i:i+2,0], profiles[i:i+2,1], 'o-k', linewidth=2)
    if scale == True:
        projection.drawmapscale(longitude_scale, latitude_scale,
                                longitude_central, latitude_central,
                                length=comprimento_scale, barstyle='fancy')    
    plt.show()
