from matplotlib import pyplot, widgets
from mpl_toolkits.basemap import Basemap
# Quick hack so that the docs can build using the mocks for readthedocs
# Ideal would be to log an error message saying that functions from pyplot
# were not imported
try:
    from matplotlib.pyplot import *
except:
    pass

def map_contour(projecao, x, y, dado, area, unidade, titulo, cores, tamanho,
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
    
    pyplot.figure(figsize=tamanho)
    pyplot.title(titulo, fontsize=18, y=1.05)
    projecao.contourf(x, y, dado, 100, tri=True, cmap=pyplot.get_cmap(cores),
                      vmin = -ranges + ranges_0, vmax = ranges + ranges_0)
    pyplot.colorbar(orientation='horizontal', pad=0.04, aspect=50, 
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
    if perfis.all() != None: #temperfil == True:
        for i in range(0,perfis.shape[0],2):
            projecao.plot(perfis[i:i+2,0], perfis[i:i+2,1], 'o-k', linewidth=2)
    if escala == True:
        projecao.drawmapscale(longitude_escala, latitude_escala,
                            longitude_central, latitude_central,
                            length=comprimento_escala, barstyle='fancy')    
    pyplot.show()
##############################################################################################################################

def map_plot(projecao, area, titulo, tamanho, delta, estados=None, escala=None, eixos=None):
    '''
    Plota um mapa via funcao basemap e desenha um retangulo numa regiao especifica, a ser realccada
    , com coordenadas "x" e "y" referidas a uma determinada projecao cartografica 
    "projecao". "unidade" e "titulo" sao, respectivamente,
    a e o titulo do mapa.
    
    inputfrom mpl_toolkits.basemap import Basemap

    
    projecao: objeto do mpl_toolkits.basemap.Basemap - 
              projecao cartografica.
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
    estados: boolean - se for igual a True, plota o contorno dos estados.
    escala: boolean - se for igual a True, plota a escala do mapa.
    eixos: boolean - se for igual a True, plota os eixos do mapa.
    
    output
    
    mapa: string - codigo de uma matplotlib.figure.Figure.
    '''
       
    if escala == True:
        #Valor em km a ser representado na escala
        #Este valor foi estabelecido como aproximadamente 
        #40 porcento da variacao maxima em x
        x_min = escala[0]
        x_max = escala[1]
        y_min = escala[2]
        y_max = escala[3]
        #x_min, x_max, y_min, y_max = area
        #x_min, y_min = projecao(area[0], area[2])
        #x_max, y_max = projecao(area[1], area[3])
        comprimento_escala = np.floor(0.4*(x_max - x_min)/100.)*100.
        #Posicao do centro da escala em coordenadas geodesicas
        longitude_escala = escala[1] - 0.25*(escala[1] - escala[0])
        latitude_escala = escala[2] + 0.05*(escala[3] - escala[2])
    
    pyplot.figure(figsize=tamanho)
    pyplot.title(titulo, fontsize=18, y=1.05)
    projecao.drawcoastlines()
    #projecao.bluemarble()
    #projecao.etopo()
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
    if escala == True:
        long_central = 0.5*(area[1] + area[0])
        lat_central = 0.5*(area[3] + area[2])
        projecao.drawmapscale(longitude_escala, latitude_escala,
                            long_central, lat_central,
                            length=comprimento_escala, barstyle='fancy')    
    pyplot.show()
