# Importing Python libraries
import numpy
from scipy.stats.stats import pearsonr
from numpy.testing import assert_almost_equal, raises
# Imporing all developed modules
from codes.statistical import cccoef
from codes.grids import regular_grid
from codes.prism import prism_tf, prism_gz
from codes.auxiliars import addnoise, my_atan, my_log
from codes.sphere import sphere_tf, sphere_tfa, sphere_gz
from codes.auxiliars import deg2rad, rad2deg, dircos, regional
# Importing fatiando modules
from fatiando.gravmag import prism, sphere
from fatiando import mesher, utils, gridder

def test_dircos1():
    '''
    This test compares the dircos function created and the dircos function from Fatiando a Terra.
    We use 50 and 50 degrees for inclination and declination values.
    '''
    
    dircos_auxiliars = dircos(50., 50.)
    dircos_fatiando = utils.dircos(50., 50.)
    
    assert_almost_equal(dircos_auxiliars, dircos_fatiando, decimal = 8)

def test_dircos2():
    '''
    This test is pretty similar to the other dircos test. We set two pair of values, one for inclination and the other one for declination. Both values are compare by using the dircos funtion.
    '''
    
    set1 = numpy.random.normal(loc = 45., scale = 40., size = 200)
    set2 = numpy.random.normal(loc = 45., scale = 40., size = 200)
    dircos_auxiliars = dircos(set1, set2)
    dircos_fatiando = utils.dircos(set1, set2)
    assert_almost_equal(dircos_auxiliars, dircos_fatiando, decimal = 8) 
    
def test_deg2rad():
    '''
    This function compares the degrees-to-radian conversion function, once there is a known expected result.
    '''
    listsize = 20
    mylist = numpy.linspace(0., 360., listsize)
    mydeg2rad = deg2rad(mylist)
    true_value = numpy.linspace(0., 2*numpy.pi, listsize)
    
    assert_almost_equal(mydeg2rad, true_value, decimal = 5)

def test_rad2deg():
    '''
    This function compares the radian-to-degrees conversion function, once there is a known expected result.
    '''
    listsize = 13
    mylist = numpy.linspace(0., 2*numpy.pi, listsize)
    
    myrad2deg = rad2deg(mylist)
    true_value = numpy.linspace(0., 360., listsize)
    
    assert_almost_equal(myrad2deg, true_value, decimal = 5)

def test_regional1():
    '''
    In this test, we set a values for the field magnetization intensity. Then we compare the results obtained by the regional calculation function and the function ang2vec from Fatiando a Terra. In thes test we use only integer values for the three parameters.
    '''
    
    intensity = 2.
    inclination = 45.
    declination = 45.
    
    regional_fatiando = utils.ang2vec(intensity, inclination, declination)
    regional_myresult = regional(intensity, inclination, declination)
    
    assert_almost_equal(regional_fatiando, regional_myresult, decimal = 6)

def test_regional2():
    '''
    In this test, we set a values for the field magnetization intensity. Then we compare the results obtained by the regional calculation function and the function ang2vec from Fatiando a Terra. In thes test we use a float values for the three parameters.
    '''
    intensity = 22458.25
    inclination = 33.67
    declination = 60.60
    
    regional_fatiando = utils.ang2vec(intensity, inclination, declination)
    regional_myresult = regional(intensity, inclination, declination)
    
    assert_almost_equal(regional_fatiando, regional_myresult, decimal = 6)
    
def test_totalfield_prism():
    '''
    This test compare the results obtained by both function that calculates the total field anomaly due to a rectangular prism. The model has the same dimensions, magnetization intensity and also the same pair for inclination and declination. We use the function from Fatiando a Terra in order to compare with our function.
    '''
    
    incf = 30.
    decf = 30.
    incs = 60.
    decs = 45.
    magnetization = 3.
    
    # Modelo para o Fatiando
    xp, yp, zp = gridder.regular((-1000, 1000, -1000, 1000), (200, 200), z=-345.)
    model = [
        mesher.Prism(-200., 200., -250., 180., 120., 1000.,
                     {'magnetization': utils.ang2vec(magnetization, incs, decs)})]
    tf = prism.tf(xp, yp, zp, model, incf, decf)
    tf = tf.reshape(200, 200)
    
    # Modelo para minha funcao
    x, y = numpy.meshgrid(numpy.linspace(-1000., 1000., 200), numpy.linspace(-1000., 1000., 200))
    z = -345.*numpy.ones((200, 200))
    mymodel = [-200., 200., -250., 180., 120., 1000., magnetization]
    mytf = prism_tf(y, x, z, mymodel, incf, decf, incs, decs)
    
    assert_almost_equal(tf, mytf, decimal = 5)

def test_totalfield_sphere():
    '''
    This test compare the results obtained by both function that calculates the total field anomaly due to a solid sphere. The model has the same dimensions, magnetization intensity and also the same pair for inclination and declination. We use the function from Fatiando a Terra in order to compare with our function.
    '''
    incf = 30.
    decf = 30.
    incs = 60.
    decs = 45.
    magnetization = 2.345
    
    # Modelo para o Fatiando
    xp, yp, zp = gridder.regular((-1000, 1000, -1000, 1000), (200, 200), z=-345.)
    model = [
        mesher.Sphere(-100., 400., 550., 380.,
                      {'magnetization': utils.ang2vec(magnetization, incs, decs)})]
    tf = sphere.tf(xp, yp, zp, model, incf, decf)
    tf = tf.reshape(200, 200)
    
    # Modelo para minha funcao
    x, y = numpy.meshgrid(numpy.linspace(-1000., 1000., 200), numpy.linspace(-1000., 1000., 200))
    z = -345.*numpy.ones((200, 200))
    mymodel = [-100., 400., 550., 380., magnetization]
    mytf = sphere_tfa(y, x, z, mymodel, incf, decf, incs, decs)
    
    assert_almost_equal(tf, mytf, decimal = 5)

def test_gz_prism():
    '''
    This test compare the results obtained by both function that calculates the vertical gravitational attraction due to a rectangular prism. The model has the same dimensions and same value for the density. We use the function from Fatiando a Terra in order to compare with our function.
    '''

    density = 2600.
    
    # Modelo para o Fatiando
    xp, yp, zp = gridder.regular((-1000, 1000, -1000, 1000), (200, 200), z=-345.)
    model = [
        mesher.Prism(-200., 200., -250., 180., 120., 1000.,{'density': density})]
    gz = prism.gz(xp, yp, zp, model)
    gz = gz.reshape(200, 200)
    
    # Modelo para minha funcao
    x, y = numpy.meshgrid(numpy.linspace(-1000., 1000., 200), numpy.linspace(-1000., 1000., 200))
    z = -345.*numpy.ones((200, 200))
    mymodel = [-200., 200., -250., 180., 120., 1000., density]
    mygz = prism_gz(y, x, z, mymodel)
    
    assert_almost_equal(gz, mygz, decimal = 5)

def test_gz_sphere():
    '''
    This test compare the results obtained by both function that calculates the vertical gravitational attraction due to a solid sphere. The model has the same dimensions and same value for the density. We use the function from Fatiando a Terra in order to compare with our function.
    '''

    density = 1500.
    
    # Modelo para o Fatiando
    xp, yp, zp = gridder.regular((-1000, 1000, -1000, 1000), (200, 200), z=-345.)
    model = [
        mesher.Sphere(-100., 400., 550., 380.,{'density': density})]
    gz = sphere.gz(xp, yp, zp, model)
    gz = gz.reshape(200, 200)
    
    # Modelo para minha funcao
    x, y = numpy.meshgrid(numpy.linspace(-1000., 1000., 200), numpy.linspace(-1000., 1000., 200))
    z = -345.*numpy.ones((200, 200))
    mymodel = [-100., 400., 550., 380., density]
    mygz = sphere_gz(y, x, z, mymodel)
    
    assert_almost_equal(gz, mygz, decimal = 5)

def test_totalfield_true_vs_approx():
    '''
    This test compares the result between the total-field anomaly due to a solid sphere, once it can be calculated with the value for the regional field or without it. The true total-field anomaly is computed with a single value for the Field intensity and a pair for inclination and declination, which is subtracted for the projection in the Field direciton; the approximated total-field anomaly is computed as a projection of all components in x, y and z directions.
    '''
    
    size = 100
    x = numpy.linspace(-500., 500., size)
    y = numpy.linspace(-500., 500., size)
    z = -200.*numpy.ones(size)
    intensity = 22180.75
    magnetization = 1.275
    mymodel = (0., 0., 50., 100., magnetization)
    
    tf_true = sphere_tf(x, y, z, mymodel, intensity, 45., 45.)
    tf_aprx = sphere_tfa(x, y, z, mymodel, 45., 45.)
    
    difference = numpy.around((numpy.abs(tf_aprx - tf_true)), decimals = 1)
    
    assert_almost_equal(tf_true, tf_aprx, decimal = 1)
    assert_almost_equal(numpy.zeros(size), difference, decimal = 1)

def test_mycorrelation_equals_to_1():
    ''' 
    This test compares the cross-correlation function, once the calculated cross-correlation between the same dataset must be equal to 1 (one).
    '''
    data = numpy.random.normal(loc = 50., scale = 2., size = 100)
    
    mycoeff = cccoef(data, data)
    assert_almost_equal(mycoeff, 1., decimal = 3)

def test_mycorrelation_vs_scipy():
    '''
    In this test, we calculate the cross-correlation coefficient of two pair of array with normal distributuion. Then we compare with the result obtained by the Scipy Cross-correlation function. Both result must be the same value, with at least a 3-decimal precision.
    '''
    
    dataset1 = numpy.random.normal(loc = 1., scale = 1., size = 500)
    dataset2 = numpy.random.normal(loc = 2., scale = 1., size = 500)
    
    mycoeff = cccoef(dataset1, dataset2)    
    scipy_c, scipy_diff  = pearsonr(dataset1, dataset2)
    assert_almost_equal(mycoeff, scipy_c, decimal = 3)
    
def test_rotation_matrix1():
    xo, yo, zo = regular_grid((-1000., 1000., -1000., 1000.), (100, 100), -500.)
    xr, yr, zr = rotate3D_xyz(xo, yo, zo, 0.)
    assert_almost_equal(xo, xr, decimal = 3)
    assert_almost_equal(yo, yr, decimal = 3)
    assert_almost_equal(zo, zr, decimal = 3)

def test_rotation_matrix2():
    xo, yo, zo = regular_grid((-1000., 1000., -1000., 1000.), (100, 100), -500.)
    xr, yr, zr = rotate3D_xyz(xo, yo, zo, -90.)
    assert_almost_equal(xo, yr, decimal = 3)
    assert_almost_equal(-yo, xr, decimal = 3)
    assert_almost_equal(zo, zr, decimal = 3)

def test_rotation_matrix3():
    xo, yo, zo = regular_grid((-1000., 1000., -1000., 1000.), (100, 100), -500.)
    xr, yr, zr = rotate3D_xyz(xo, yo, zo, 180.)
    assert_almost_equal(-xo, xr, decimal = 3)
    assert_almost_equal(-yo, yr, decimal = 3)
    assert_almost_equal(zo, zr, decimal = 3)