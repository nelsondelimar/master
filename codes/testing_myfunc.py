import nelsonfunctions as nf
import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import raises
from scipy.linalg import lu

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#
#                        ARQUIVO DE TESTES PARA COMPUTAR AS FUNCOES CRIADAS NO ARQUIVO NELSON_FUNCTIONS
#
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------
#                                              Testes para a funcao de produto escalar
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_dotproduct_a():
    '''
    
    Compara resultado obtido utilizando a funcao criada e o resultado atraves da funcao python np.dot
    
    '''
    
    a = np.array([0, 0, 1])
    b = np.array([3, 2, 1])
    original_value = np.dot(a,b)
    ab = nf.dot1(a, b)
    
    aae(ab, original_value, decimal=15)

def test_dotproduct_b():
    '''
    
    Compara o resultado da funcao criada com o resultado esperado!
    
    '''
    x = np.array([1, 2, 1])
    y = np.array([2, 1, 1])
    original_value = 5
    xy = nf.dot1(x, y)
    
    aae(xy, original_value, decimal=15)

def test_dotproduct_c():
    '''
    
    Dado um INPUT conhecido, faz a verificacao com o arquivo de funcoes para erro de dimesao de vetores!
    
    '''

    vector_x = np.array([1, 3, 5, 6, 8])
    vector_y = np.array([2, 4, 6])
    
    raises (AssertionError, nf.dot1, x=vector_x, y=vector_y)

def test_dotproduct_d():
    '''
    
    Dados dois vetores, x e y, de forma aleatoria, calcula o produto escalar entre eles e compara com a funcao numpy dot. Neste teste, utilizamos 5 casas decimais para os elementos dos vetores.
    
    '''

    n = 4
    x = np.around(np.random.rand(n), decimals = 5)
    y = np.around(np.random.rand(n), decimals = 5)
    
    xy = nf.dot1(x,y)
    
    original_value = np.dot(x,y)
    
    aae(xy, original_value, decimal=15)

def test_dotproduct_e():
    '''
    
    Dados dois vetores, x e y, de forma aleatoria, calcula o produto escalar entre eles e compara com a funcao numpy dot. Neste teste, utilizamos 5 casas decimais para os elementos dos vetores.
    
    '''

    n = 10
    x = np.around(np.random.rand(n), decimals = 5)
    y = np.around(np.random.rand(n), decimals = 5)
    
    xy = nf.dot1(x,y)
    original_value = np.dot(x,y)
    
    aae(xy, original_value, decimal = 15)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                   Testes para o produto de Hadamard
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_hadamard_a():
    '''
    
    Compara o resultado da funcao criada com o resultado esperado do calculo vetorial.
    
    '''
    
    x = np.array([0, 0, 1])
    y = np.array([3, 2, 1])
    original_value = np.array([0, 0, 1])
    xy = nf.hadamard1(x, y)
    
    aae(xy, original_value, decimal=15)

def test_hadamard_b():
    '''
    
    Compara o resultado obtido pela funcao com o resultado original!
    
    '''

    x = np.array([1, 3, 5])
    y = np.array([2, 4, 6])
    original_value = np.multiply(x,y)
    xy1 = nf.hadamard1(x,y)
    xy2 = nf.hadamard1(x,y)

    aae(xy1, original_value, decimal=15)
    aae(xy2, original_value, decimal=15)

def test_hadamard_c():
    '''
    
    Dado um INPUT conhecido, verifica se existe o erro de dimensao de vetores!    
    
    '''

    vector_x = np.array([0, -3,-2, 2, 1])
    vector_y = np.array([1, -4, 2]) 

    raises (AssertionError, nf.hadamard1, x=vector_x, y=vector_y)

def test_hadamard_d():
    '''
    
    Compara o resultado obtido pela funcao com o resultado da funcao original do Python (np.multiply).
    
    '''

    x = np.array([0, 3, 1])
    y = np.array([2, 0, 7])
    original_value = np.multiply(x,y)
    xy = nf.hadamard1(x, y)

    aae(xy, original_value, decimal=15)

def test_hadamard_e():    
    '''
    
    Compara o resultado obtido pela funcao com o resultado da funcao original do Python (np.multiply).
    
    '''

    x = np.array([0, 3, 1])
    y = np.array([2, 0, 7])
    original_value = np.multiply(x,y)
    xy = nf.hadamard2(x, y)

    aae(xy, original_value, decimal=15)

def test_dot_plus_hadamard_1():    
    '''
    
    Dados quatro vetores (x, y, u, v), calcula o produto de Hadamard entre xy e uv e depois calcula o produto escalar entre (xy) e (uv). O teste compara o produto de Hadamard entre (x) e (y) e (u) e (v) o resultado do produto escalar entre xy e uv com um resultado esperado.
    
    '''

    x = np.array([1., 1., 3., 4.])
    y = np.array([2., -2., -1., 0.])
    u = np.array([0., 2., -1., 2.])
    v = np.array([5., 2., 4., 1.])
    
    expected_xy = np.array([2., -2., -3., 0.]) 
    expected_uv = np.array([0., 4., -4., 2.])
    
    x_y1 = nf.hadamard1(x,y)
    x_y2 = nf.hadamard2(x,y)
    
    u_v1 = nf.hadamard1(u,v)
    u_v2 = nf.hadamard2(u,v)
    
    expected_xy_uv = 4
    
    dot1_xy_uv = nf.dot1(x_y1,u_v1)
    dot2_xy_uv = nf.dot1(x_y2,u_v2)
    
    aae(expected_xy, x_y1, decimal=15)
    aae(expected_xy, x_y2, decimal=15)
    aae(expected_uv, u_v1, decimal=15)
    aae(expected_uv, u_v2, decimal=15)
    
    aae(expected_xy_uv, dot1_xy_uv, decimal=15)
    aae(expected_xy_uv, dot2_xy_uv, decimal=15)

def test_dot_plus_hadamard_2():
    '''
    Dados quatro vetores (x, y, u, v), calcula o produto de Hadamard entre xy e uv e depois calcula o produto escalar entre (xy) e (uv). 
    O teste compara o produto de Hadamard entre (x) e (y) e (u) e (v) com a funcao np.multiply e compara o resultado do produto escalar entre xy e uv com a funcao np.dot.
    
    '''
    
    N = 10
    
    x = np.random.rand(N)
    y = np.random.rand(N)
    u = np.random.rand(N)
    v = np.random.rand(N)
    
    expected_xy = np.multiply(x,y)
    expected_uv = np.multiply(u,v)
    
    x_y1 = nf.hadamard1(x,y)
    x_y2 = nf.hadamard2(x,y)
    
    u_v1 = nf.hadamard1(u,v)
    u_v2 = nf.hadamard2(u,v)
    
    expected_xy_uv = np.dot(expected_xy, expected_uv)
    
    dot1_xy_uv = nf.dot1(x_y1,u_v1)
    dot2_xy_uv = nf.dot1(x_y2,u_v2)
    
    aae(expected_xy, x_y1, decimal=15)
    aae(expected_xy, x_y2, decimal=15)
    aae(expected_uv, u_v1, decimal=15)
    aae(expected_uv, u_v2, decimal=15)
    
    aae(expected_xy_uv, dot1_xy_uv, decimal=15)
    aae(expected_xy_uv, dot2_xy_uv, decimal=15)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                       Testes para o produto externo
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_outer_a():
    '''
    
    Compara o resultado das funcoes Outer1, Outer2 e Outer3 com o resultado da funcao Python.
    
    '''

    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3, 4, 5])
    
    original_value = np.outer(x, y)
    
    xy_1 = nf.outer1(x, y)
    xy_2 = nf.outer2(x, y)
    xy_3 = nf.outer3(x, y)

    aae(xy_1, original_value, decimal=10)
    aae(xy_2, original_value, decimal=10)
    aae(xy_3, original_value, decimal=10)

def test_outer_b():    
    '''
    
    Compara o resultado obtido atraves da funcao Outer1 com o resultado esperado!
    
    '''
    
    x = np.array([1, 1, 1])
    y = np.array([2, 2, 2, 2, 2])
    
    original_value = np.array([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]])
    
    xy = nf.outer1(x, y)

    aae(xy, original_value, decimal=15)

def test_outer_c():
    '''
    Compara o resultado obtido atraves da funcao Outer2 com o resultado esperado.
    
    '''
    
    x = np.array([1, 1, 1])
    y = np.array([2, 2, 2, 2, 2])
    
    original_value = np.array([[2, 2, 2, 2, 2],
                               [2, 2, 2, 2, 2],
                               [2, 2, 2, 2, 2]])
    
    xy = nf.outer2(x, y)

    aae(xy, original_value, decimal=12)

def test_outer_d():    
    '''
    
    Compara o resultado obtido atraves da funcao Outer3 com o resultado esperado!
    
    '''
    
    x = np.array([1, 1, 1])
    y = np.array([2, 2, 2, 2, 2])
    
    original_value = np.array([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]])
    
    xy = nf.outer3(x, y)

    aae(xy, original_value, decimal=12)

def test_outer_e():
    '''
    
    Verifica se a matriz de resultado do produto externo XoY e igual a matriz transposta de resultado da operacao de produto externo YoX!
    
    '''
    
    x = np.array([1, 1])
    y = np.array([2, 2, 2])
    
    result_xy = ([[2, 2, 2],[2, 2, 2]])
    result_yx = ([[2, 2],[2, 2],[2, 2]])
        
    xy = nf.outer2(x,y)
    yx = nf.outer2(y,x)
    np.allclose(xy, yx.T)

def test_outer_f():    
    '''
    
    Verifica se a matriz de resultado do produto externo XoY e igual a matriz transposta de resultado da operacao de produto externo YoX!
    
    '''
    
    x = np.array([1, 1])
    y = np.array([2, 2, 2])
    
    result_xy = ([[2, 2, 2],[2, 2, 2]])
    result_yx = ([[2, 2],[2, 2],[2, 2]])
    
    xy = nf.outer2(x,y)
    yx = nf.outer2(y,x)
    np.allclose(xy, yx.T)

def test_outer_f():
    '''
    
    Verifica se a matriz de resultado do produto externo XoY e igual a matriz transposta de resultado da operacao de produto externo YoX!
    
    '''
    
    x = np.array([1, 1])
    y = np.array([2, 2, 2])
    
    result_xy = ([[2, 2, 2],[2, 2, 2]])
    result_yx = ([[2, 2],[2, 2],[2, 2]])
    
    xy = nf.outer3(x,y)
    yx = nf.outer3(y,x)
    np.allclose(xy, yx.T)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                               Teste para o calculo de Matriz e Vetor
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_mat_vec1():
    '''
    
    Compara o resultado obtido atraves da primeira funcao com o resultado esperado.
    
    '''

    A = np.array([[1, 0, 1], [1, 2, 1]])
    x = np.array([0, 1, 0])
    original_value = [0, 2]
    final_value = nf.matrix_vector1(A, x)

    aae(final_value, original_value, decimal=15)
    
def test_mat_vec2():
    '''
    
    Compara o resultado obtido atraves da segunda funcao com o resultado esperado
    
    '''

    A = np.array([[1, 0, 1], [1, 2, 1]])
    x = np.array([0, 1, 0])
    original_value = [0, 2]
    final_value = nf.matrix_vector2(A, x)

    aae(final_value, original_value, decimal=15)
    
def test_mat_vec3():
    '''
    
    Compara o resultado obtido atraves da terceira funcao com o resultado esperado
    
    '''

    A = np.array([[1, 0, 1], [1, 2, 1]])
    x = np.array([0, 1, 0])
    original_value = [0, 2]
    final_value = nf.matrix_vector3(A, x)

    aae(final_value, original_value, decimal=15)

def test_mat_vec4():
    '''
    
    Compara o resultado obtido atraves da primeira funcao com o resultado esperado atraves da funcao numpy.dot
    
    '''
    
    A = np.array([[1, 0, 1], [1, 2, 1]])
    x = np.array([0, 1, 0])
    original_value = np.dot(A, x)
    final_value = nf.matrix_vector1(A, x)

    aae(final_value, original_value, decimal=15)

def test_mat_vec5():
    '''
    
    Compara o resultado obtido atraves da segunda funcao com o resultado esperado atraves da funcao numpy.dot
    
    '''
    
    A = np.array([[1, 0, 1], [1, 2, 1]])
    x = np.array([0, 1, 0])
    original_value = np.dot(A, x)
    final_value = nf.matrix_vector2(A, x)

    aae(final_value, original_value, decimal=15)
    
def test_mat_vec6():
    '''
    
    Compara o resultado obtido atraves da segunda funcao com o resultado esperado atraves da funcao numpy.dot
    
    '''
    
    A = np.array([[1, 0, 1], [1, 2, 1]])
    x = np.array([0, 1, 0])
    original_value = np.dot(A, x)
    final_value = nf.matrix_vector3(A, x)

    aae(final_value, original_value, decimal=15)
    
# ----------------------------------------------------------------------------------------------------------------------------------------# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                    Teste para o calculo de derivadas
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_derivative_seno():
    '''
    
    Compara o resultado produzido pela funcao de primeira derivada. Utilizamos a derivada da funcao seno, que equivale a funcao cosseno e comparamos com o valor resultante para a funcao cosseno existente no biblioteca numpy.
        
    '''
    
    # Input
    dx = 2.*np.pi/10000
    argument = np.arange(0., 2.*np.pi, dx)
    # Funcao a ser derivada
    seno = np.sin(argument)
    # Derivada da funcao acima
    cosseno = np.cos(argument)
    cosseno[0] = 0.
    cosseno[-1] = 0.
    
    cos_calc = nf.derivative(seno,dx)
    
    aae(cosseno, cos_calc, decimal = 3)

def test_derivative_cosseno():
    '''
    
    Compara o resultado produzido pela funcao de primeira derivada. Utilizamos a derivada da funcao cosseno, que equivale a funcao seno e comparamos com o valor resultante para a funcao seno existente no biblioteca numpy.
        
    '''
    
    # Input
    dx = 2.*np.pi/10000
    argument = np.arange(0., 2.*np.pi, dx)
    # Funcao a ser derivada
    cosseno = np.cos(argument)
    # Derivada da funcao
    seno = (-1)*np.sin(argument)
    seno[0] = 0.
    seno[-1] = 0.
    
    sen_calc = nf.derivative(cosseno,dx)
    
    aae(seno, sen_calc, decimal = 3)

def test_derivative_soma_sen_cos():
    '''
    
    Compara o resultado produzido pela funcao de primeira derivada. Utilizamos a derivada da funcao (seno + cos), que equivale a funcao (cos - seno) e comparamos com o valor resultante.
        
    '''
    
    # Input
    dx = 2.*np.pi/10000
    argument = np.arange(0., 2.*np.pi, dx)
    # Funcao a ser derivada
    func = np.sin(argument) + np.cos(argument)
    # Derivada da funcao
    dfunc = np.cos(argument) - np.sin(argument)
    dfunc[0] = 0.
    dfunc[-1] = 0.
    
    dfunc_calc = nf.derivative(func,dx)
    
    aae(dfunc, dfunc_calc, decimal = 3)

def test_derivative_diferenca_sen_cos():
    '''
    
    Compara o resultado produzido pela funcao de primeira derivada. Utilizamos a derivada da funcao (sen - cos), que equivale a funcao (seno + cos) e comparamos com o valor resultante.
        
    '''
    
    # Input
    dx = 2.*np.pi/10000
    argument = np.arange(0., 2.*np.pi, dx)
    # Funcao a ser derivada
    func = np.sin(argument) - np.cos(argument)
    # Derivada da funcao
    dfunc = np.cos(argument) + np.sin(argument)
    dfunc[0] = 0.
    dfunc[-1] = 0.
    
    dfunc_calc = nf.derivative(func,dx)
    
    aae(dfunc, dfunc_calc, decimal = 3)

def test_derivative_prod_sen_cos():
    '''
    
    Compara o resultado produzido pela funcao de primeira derivada. Utilizamos a derivada da funcao (sen*cos), que equivale a funcao (cos^2 - sen^2) e comparamos com o valor resultante.
        
    '''
    
    # Input
    dx = 2.*np.pi/10000
    argument = np.arange(0., 2.*np.pi, dx)
    # Funcao a ser derivada
    func = np.sin(argument)*np.cos(argument)
    # Derivada da funcao
    dfunc = (np.cos(argument))**2 - (np.sin(argument))**2
    dfunc[0] = 0.
    dfunc[-1] = 0.
    
    dfunc_calc = nf.derivative(func,dx)
    
    aae(dfunc, dfunc_calc, decimal = 5)

def test_derivative_quadratic():
    '''
    
    Compara o resultado produzido pela funcao de primeira derivada. Utilizamos a derivada da funcao quadratica, que equivale a funcao linear do primeiro grau.
    Funcao: 5x^2 - 10x + 10. Derivada: 10x - 10.
        
    '''
    
    # Input
    dx = 2.*np.pi/10000
    argument = np.arange(0., 2.*np.pi, dx)
    # Funcao a ser derivada
    func = 5.*(argument**2) - 10.*(argument) + 10.
    # Derivada da funcao
    dfunc = 10*(argument) - 10.
    dfunc[0] = 0.
    dfunc[-1] = 0.
    
    dfunc_calc = nf.derivative(func,dx)
    
    aae(dfunc, dfunc_calc, decimal = 5)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                   Teste para o calculo de media movel
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_sma1_window():
    '''
    Verifica se o tamanho da janela escolhido para o calculo da media movel e maior do que o numero de elementos que existem no vetor. O teste e valido para a funcao SMA1.
    
    '''
    
    vec_test = np.arange(7)
    ws_test = 8
    raises(AssertionError, nf.sma_function1, vector=vec_test, window=ws_test)

def test_sma2_window():
    '''
    
    Verifica se o tamanho da janela escolhido para o calculo da media movel e maior do que o numero de elementos que existem no vetor. O teste e valido para a funcao SMA2.
    
    '''
    
    vec_test = np.arange(7)
    ws_test = 8
    raises(AssertionError, nf.sma_function2, vector = vec_test, window = ws_test)
    
def test_sma3_window():
    '''
    
    Verifica se o tamanho da janela escolhido para o calculo da media movel e maior do que o numero de elementos que existem no vetor. O teste e valido para a funcao SMA3.
    
    '''
    
    vec_test = np.arange(7)
    ws_test = 8
    raises(AssertionError, nf.sma_function3, vector = vec_test, window = ws_test)

def test2_sma_result_ones():
    '''
    
    Compara o resultado da funcao SMA1D com as funcoes criadas no arquivo de funcoes nelson_functions.
    
    '''
    ws_test = 3
    y_test = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    expected = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    result = nf.sma1d(y_test, ws_test)
    result1 = nf.sma_function1(y_test, ws_test)
    result2 = nf.sma_function2(y_test, ws_test)
    result3 = nf.sma_function3(y_test, ws_test)
    aae(result, result1, decimal = 15)
    aae(result, result2, decimal = 15)
    aae(result, result3, decimal = 15)

def test2_sma_result_random():
    '''
    
    Compara o resultado da funcao sma1D com as funcoes criadas no arquivo de funcoes nelson_functions.
    
    '''
    
    number = 15
    ws_test = 3
    y_test = 10*np.around(np.random.rand(number))
    result = nf.sma1d(y_test, ws_test)
    result1 = nf.sma_function1(y_test, ws_test)
    result2 = nf.sma_function2(y_test, ws_test)
    result3 = nf.sma_function3(y_test, ws_test)
    aae(result, result1, decimal = 15)
    aae(result, result2, decimal = 15)
    aae(result, result3, decimal = 15)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                            Teste para o calculo de Matriz e Matriz
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_matrix_matrix1():
    '''
    
    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_1 e o resultado esperado.
    
    '''
    
    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.array([[2, 1], [0, 1], [1,1]])
    final_result = nf.mat_mat1(A,B)
    aae(final_result, expected, decimal=15)

def test_matrix_matrix1_dot():
    '''

    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_1 e o resultado esperado atraves da funcao de produto escalar na biblioteca numpy.

    '''
    
    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.dot(A,B)
    final_result = nf.mat_mat1(A,B)
    aae(final_result, expected, decimal=15)

def test_matrix__matrix2():
    '''

    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_2_nelson e o resultado esperado.
    
    '''
    
    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.array([[2, 1], [0, 1], [1,1]])
    final_result = nf.mat_mat2_nelson(A,B)
    aae(final_result, expected, decimal=15)

def test_matrix_matrix2_dot():
    '''
    
    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_2_nelson e o resultado esperado atraves da funcao de produto escalar na biblioteca numpy.
    
    '''
    
    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.dot(A,B)
    final_result = nf.mat_mat2_nelson(A,B)
    aae(final_result, expected, decimal=15)

def test_matrix__matrix3():
    '''

    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_3_nelson e o resultado esperado.
    
    '''
    
    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.array([[2, 1], [0, 1], [1,1]])
    final_result = nf.mat_mat3_nelson(A,B)
    aae(final_result, expected, decimal=15)

def test_matrix_matrix3_dot():
    '''
    
    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_3_nelson e o resultado esperado atraves da funcao de produto escalar na biblioteca numpy.
    
    '''
    
    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.dot(A,B)
    final_result = nf.mat_mat3_nelson(A,B)
    aae(final_result, expected, decimal=15)
    
def test_matrix_matrix4():
    '''
    
    Compara o resultado da operacao matriz_matriz atraves da funcao mat_mat_4_nelson e o resultado esperado.
    
    '''
    
    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.array([[2, 1], [0, 1], [1,1]])
    final_result1 = nf.mat_mat4_nelson1(A,B)
    final_result2 = nf.mat_mat4_nelson2(A,B)
    final_result3 = nf.mat_mat4_nelson3(A,B)
    aae(final_result1, expected, decimal=15)
    aae(final_result2, expected, decimal=15)
    aae(final_result3, expected, decimal=15)

def test_matrix_matrix4_dot():
    '''
    
    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_4_nelson e o resultado esperado atraves da funcao de produto escalar na biblioteca numpy.
    
    '''

    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.dot(A,B)
    final_result = nf.mat_mat4_nelson1(A,B)
    aae(final_result, expected, decimal=15)
    
def test_matrix_matrix5():
    '''
    
    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_5_nelson e o resultado esperado.
    
    '''
    
    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.array([[2, 1], [0, 1], [1,1]])
    final_result1 = nf.mat_mat5_nelson1(A,B)
    final_result2 = nf.mat_mat5_nelson2(A,B)
    final_result3 = nf.mat_mat5_nelson3(A,B)
    aae(final_result1, expected, decimal=15)
    aae(final_result2, expected, decimal=15)
    aae(final_result3, expected, decimal=15)

def test_matrix_matrix5_dot():
    '''
    
    Compara o resultado da operacao matriz-matriz atraves da funcao mat_mat_5_nelson e o resultado esperado atraves da funcao de produto escalar na biblioteca numpy.
    
    '''

    A = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    expected = np.dot(A,B)
    final_result1 = nf.mat_mat5_nelson1(A,B)
    final_result2 = nf.mat_mat5_nelson2(A,B)
    final_result3 = nf.mat_mat5_nelson3(A,B)
    aae(final_result1, expected, decimal=15)
    aae(final_result2, expected, decimal=15)
    aae(final_result3, expected, decimal=15)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                   Testes para a Matriz de Rotacao
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_rotation_matrix1():
    '''
    
    Teste (1) para verificar se as matrizes Rx, Ry e Rz sao ortogonais. Para isso, calculamos a matriz transporta e a matriz inversa. Se as respectivas matrizes inversa e transposta sao iguais, entao a matriz e ortogonal.

    '''
    
    # Escolha do angulo (em graus)
    theta = 60
    # Calculo das matrizes de rotacao Rx, Ry, Rz
    Rx, Ry, Rz = nf.rotation_matrix(theta)
    
    # Para as direcoes nos eixos X, Y e Z, respectivamente, temos:
    aae(np.linalg.inv(Rx), Rx.T, decimal=5)
    aae(np.linalg.inv(Ry), Ry.T, decimal=5)
    aae(np.linalg.inv(Rz), Rz.T, decimal=5)

def test_rotation_matrix2():
    '''
    
    Teste (2) para verificar se as matrizes Rx, Ry e Rz sao ortogonais. Para isso, calculamos o produto entre a matriz R e a matriz transposta. Se a multiplicacao resulta na matriz identidade, as matrizes sao ortogonais.

    '''
    
    # Escolha do angulo (em graus)
    theta = 45
    # Calculo das matrizes de rotacao Rx, Ry, Rz
    Rx, Ry, Rz = nf.rotation_matrix(theta)
    
    # Calculo do produto entre a cada matriz e sua transposta
    result1 = np.dot(Rx,Rx.T)
    result2 = np.dot(Ry,Ry.T)
    result3 = np.dot(Rz,Rz.T)
    
    # Para as direcoes nos eixos X, Y e Z, respectivamente, temos:
    aae(result1, result2, decimal=5)
    aae(result1, result3, decimal=5)
    aae(result2, result3, decimal=5)
    aae(result1, np.identity(3), decimal=5)
    aae(result2, np.identity(3), decimal=5)
    aae(result3, np.identity(3), decimal=5)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                          Testes criados para o algoritmos 3 e 5, 8 e 10
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test1_alg3_mat():
    '''
    
    Este teste compara o resultado obtido atraves da multiplicacao entre uma matriz triangular superior e um vetor x com um resultado conhecido.
    
    '''
    
    mat_u = np.array([[1., 1., 1.],
                      [0., 1., 2.],
                      [0., 0., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.array([4., 5., 4.])
    calculated = nf.upper_matvec_3(mat_u,vetor)
    
    aae(expected, calculated, decimal = 15)
    
def test2_alg3_mat():
    '''
    
    Este teste compara o resultado obtido atraves da multiplicacao entre uma matriz triangular superior e um vetor x com um resultado obtido atraves da funcao numpy.dot.
    
    '''
    
    mat_u = np.array([[1., 1., 1.],
                      [0., 1., 2.],
                      [0., 0., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.dot(mat_u,vetor)
    calculated = nf.upper_matvec_3(mat_u,vetor)
    
    aae(expected, calculated, decimal = 15)

def test3_alg5_mat():
    '''
    
    Este teste compara o resultado obtido atraves da multiplicacao entre uma matriz triangular superior e um vetor x com um resultado obtido atraves da funcao numpy.dot.
    
    '''

    mat_u = np.array([[1., 1., 1.],
                      [0., 1., 2.],
                      [0., 0., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.array([4., 5., 4.])
    calculated = nf.upper_matvec_5(mat_u,vetor)
    
    aae(expected, calculated, decimal = 15)

def test4_alg5_mat():
    '''
    
    Este teste compara o resultado obtido atraves da multiplicacao entre uma matriz triangular superior e um vetor x com um resultado conhecido.
    
    '''
    
    mat_u = np.array([[1., 1., 1.],
                      [0., 1., 2.],
                      [0., 0., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.dot(mat_u,vetor)
    calculated = nf.upper_matvec_5(mat_u,vetor)
    
    aae(expected, calculated, decimal = 15)
    
def test5_alg3_alg5_mat():
    '''
    
    Este teste compara os resultado obtido atraves da multiplicacao entre uma matriz triangular superior e um vetor x, utilizando os dois algoritmos citados (3 e 5).
    
    '''
    
    mat_u = np.array([[1., 1., 1.],
                      [0., 1., 2.],
                      [0., 0., 2.]])
    vetor = np.array([1., 1., 2.])
    
    result_alg3 = nf.upper_matvec_3(mat_u,vetor)
    result_alg5 = nf.upper_matvec_5(mat_u,vetor)
    
    aae(result_alg3, result_alg5, decimal = 15)

def test1_alg8_mat():
    '''
    
    Este teste compara o resultado obtido atraves da multiplicacao entre uma matriz triangular inferior e um vetor x com um resultado conhecido.
    
    '''
    
    mat_l = np.array([[1., 0., 0.],
                      [2., 1., 0.],
                      [1., 1., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.array([1., 3., 6.])
    calculated = nf.lower_matvec_8(mat_l,vetor)
    
    aae(expected, calculated, decimal = 15)
    
def test2_alg8_mat():
    '''
    
    Este teste compara o resultado obtido atraves da multiplicacao entre uma matriz triangular inferior e um vetor x com um resultado obtido atraves da funcao numpy.dot.
    
    '''
    
    mat_l = np.array([[1., 0., 0.],
                      [2., 1., 0.],
                      [1., 1., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.dot(mat_l,vetor)
    calculated = nf.lower_matvec_8(mat_l,vetor)
    
    aae(expected, calculated, decimal = 15)

def test3_alg10_mat():
    '''
    
    Este teste compara o resultado obtido atraves da multiplicacao entre uma matriz triangular inferior e um vetor x com um resultado conhecido.
    
    '''

    mat_l = np.array([[1., 0., 0.],
                      [2., 1., 0.],
                      [1., 1., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.array([1., 3., 6.])
    calculated = nf.lower_matvec_10(mat_l,vetor)
    
    aae(expected, calculated, decimal = 15)

def test4_alg10_mat():
    '''
    
    Este teste compara o resultado obtido atraves da multiplicacao entre uma matriz triangular inferior e um vetor x com um resultado obtido atraves da funcao numpy.dot.

    '''
    
    mat_l = np.array([[1., 0., 0.],
                      [2., 1., 0.],
                      [1., 1., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.dot(mat_l,vetor)
    calculated = nf.lower_matvec_10(mat_l,vetor)
    
    aae(expected, calculated, decimal = 15)
    
def test5_alg8_alg10_mat():
    '''
    
    Este teste compara os resultado obtido atraves da multiplicacao entre uma matriz triangular superior e um vetor x, utilizando os dois algoritmos citados (8 e 10).
    
    '''
    
    mat_l = np.array([[1., 0., 0.],
                      [2., 1., 0.],
                      [1., 1., 2.]])
    vetor = np.array([1., 1., 2.])
    
    expected = np.dot(mat_l,vetor)
    
    result_alg8 = nf.lower_matvec_8(mat_l,vetor)
    result_alg10 = nf.lower_matvec_10(mat_l,vetor)
    
    
    aae(result_alg8, result_alg10, decimal = 15)

# ----------------------------------------------------------------------------------------------------------------------------------------# ----------------------------------------------------------------------------------------------------------------------------------------
#                                    Testes para armazenamento de U e L por linhas e colunas
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test1_storage_upper_elements():
    '''
    
    Dada uma matriz triangular superior de dimensao N, verifica se o numero de elementos do vetor armazenado a partir do esquema de linhas e colunas e igual a N*(N+1)/2.
    
    '''
    
    A = np.array([[1., 2., 3., 4., 5.],
                  [6., 7., 8., 9., 10.],
                  [10., 9., 8., 7., 6.],
                  [5., 4., 3., 2., 1.],
                  [1., 0., 1., 0., -1]])
    
    U = np.triu(A)
    ur = nf.up_storage_rows(U)
    uc = nf.up_storage_columns(U)
    
    N = U.shape[0]
    ur_elem = N*(N+1)/2
    uc_elem = N*(N+1)/2
    
    aae(ur.size, ur_elem, decimal = 15)
    aae(uc.size, uc_elem, decimal = 15)

def test1_storage_lower_elements():
    '''
    
    Dada uma matriz triangular superior de dimensao N, verifica se o numero de elementos do vetor armazenado a partir do esquema de linhas e colunas e igual a N*(N+1)/2.
    
    '''
    
    A = np.array([[1., 2., 3., 4., 5.],
                  [6., 7., 8., 9., 10.],
                  [10., 9., 8., 7., 6.],
                  [5., 4., 3., 2., 1.],
                  [1., 0., 1., 0., -1]])
    
    L = np.tril(A)
    lr = nf.low_storage_rows(L)
    lc = nf.low_storage_columns(L)
    
    N = L.shape[0]
    lr_elem = N*(N+1)/2
    lc_elem = N*(N+1)/2
    
    aae(lr.size, lr_elem, decimal = 15)
    aae(lc.size, lc_elem, decimal = 15)
    
def test2_storage_upper():
    '''
    
    Dada uma matriz triangular superior, compara o resultado da funcao de armazenamento por linhas e colunas com um resultado esperado.
    
    '''
    
    mat = np.array([[1., 2., 3.],
                    [0., 4., 5.],
                    [0., 0., 6.]])
    
    vector_row = np.array([1., 2., 3., 4., 5., 6.])
    vector_col = np.array([1., 2., 4., 3., 5., 6.])
    
    urow = nf.up_storage_rows(mat)
    ucol = nf.up_storage_columns(mat)
    
    aae(vector_row, urow, decimal = 15)
    aae(vector_col, ucol, decimal = 15)

def test2_storage_lower():
    '''
    
    Dada uma matriz triangular inferior, compara o resultado da funcao de armazenamento por linhas e colunas com um resultado esperado.
    
    '''
    
    mat = np.array([[1., 0., 0.],
                    [2., 4., 0.],
                    [3., 5., 6.]])
    
    vector_row = np.array([1., 2., 4., 3., 5., 6.])
    vector_col = np.array([1., 2., 3., 4., 5., 6.])
    
    lrow = nf.low_storage_rows(mat)
    lcol = nf.low_storage_columns(mat)
    
    aae(vector_row, lrow, decimal = 15)
    aae(vector_col, lcol, decimal = 15)

def test3_storage_symmetric():
    '''
    
    Dada uma matriz simetrica, o teste compara: (1) se o armazenamento da matriz superior por linhas e igual ao armazenamento da matriz inferior por colunas; (2) se o armazenamento da matriz superior por colunas e matriz inferior por linhas.
    
    '''
    
    mat = np.array([[1., 2., 3.],
                    [2., 2., 5.],
                    [3., 5., 3.]])
        
    urow = nf.up_storage_rows(mat)
    ucol = nf.up_storage_columns(mat)
    lrow = nf.low_storage_rows(mat)
    lcol = nf.low_storage_columns(mat)
    
    aae(urow, lcol, decimal = 15)
    aae(ucol, lrow, decimal = 15)    

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
# Testes criados para compara o resultado das multiplicacoes entre uma matriz triangular (U e L) com um vetor x qualquer, usando os algoritmos 3, 5, 8 e 10. Esses algoritmos calculam o produto entre um vetor u ou l com outro vetor x, sendo os vetores ur e uc os vetores obtidos a partir do armazenamento da matriz U por linhas e colunas, respectivamente, enquanto os vetores lr e lc sao obtidos a partir do arzamenamento da matriz L por linhas e colunas, respectivamente. 
# Os testes irao comparar: (1) para a matriz triangular superior: uc*x e ur*x, uc*x e dot(U,x), ur*x e dot(U,x); (2) para a matriz triangular inferior: lc*x e lr*x, lc*x e dot(L,x), lr*c e dot(L,x).
# ----------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

def test_storage_calculation_3e5():
    '''
    
    Dada uma matriz triangular superior U, este teste compara o resultado da multiplicao entre os vetores ur e uc com o vetor x, sendo ur e uc os vetores armazenados a partir do esquema de linhas e colunas, respectivamente. Neste teste, comparamos com um resultado conhecido.
    
    '''
    
    mat = np.array([[1.4, 2.9, 3.3],
                    [0.1, 9.9, 2.4],
                    [2.3, 3.4, 5.7]])
    vetor = np.array([1.5, 2.6, 3.7])
    
    U = np.triu(mat)
    vet_ur = nf.up_storage_rows(U)
    vet_uc = nf.up_storage_columns(U)
    
    expected = np.array([21.85, 34.62, 21.09])
    calc_alg3 = nf.storage_alg3(vet_ur,vetor)
    calc_alg5 = nf.storage_alg5(vet_uc,vetor)
    
    aae(calc_alg3, calc_alg5, decimal = 10)
    aae(calc_alg3, expected, decimal = 10)
    aae(calc_alg5, expected, decimal = 10)
    
def test_storage_calculation_3_dot():
    '''
    
    Dada uma matriz triangular superior U, este teste compara o resultado da multiplicao entre os vetores ur com o vetor x, sendo ur o vetor armazenado a partir do esquema de linhas, respectivamente. Neste teste, comparamos com um resultado obtido a partir de Numpy Dot.
    
    '''
    
    mat = np.array([[1.4, 2.9, 3.3],
                    [0.1, 9.9, 2.4],
                    [2.3, 3.4, 5.7]])
    vetor = np.array([1.5, 2.6, 3.7])
    
    U = np.triu(mat)
    vet_ur = nf.up_storage_rows(U)
        
    expected = np.dot(U,vetor)
    calc_alg3 = nf.storage_alg3(vet_ur,vetor)
        
    aae(calc_alg3, expected, decimal = 10)
        
def test_storage_calculation_5_dot():
    '''
    
    Dada uma matriz triangular superior U, este teste compara o resultado da multiplicao entre os vetores uc com o vetor x, sendo uc o vetor armazenado a partir do esquema de colunas, respectivamente. Neste teste, comparamos com um resultado obtido a partir de Numpy Dot.
    
    '''
    
    mat = np.array([[1.4, 2.9, 3.3],
                    [0.1, 9.9, 2.4],
                    [2.3, 3.4, 5.7]])
    vetor = np.array([1.5, 2.6, 3.7])
    
    U = np.triu(mat)
    vet_uc = nf.up_storage_columns(U)
        
    expected = np.dot(U,vetor)
    calc_alg5 = nf.storage_alg5(vet_uc,vetor)
        
    aae(calc_alg5, expected, decimal = 10)


def test_storage_calculation_8e10():
    '''
    
    Dada uma matriz triangular inferior L, este teste compara o resultado da multiplicao entre os vetores lr e lc com o vetor x, sendo lr e lc os vetores armazenados a partir do esquema de linhas e colunas, respectivamente. Neste teste, comparamos com um resultado conhecido.
    
    '''
    
    mat = np.array([[1.4, 2.9, 3.3],
                    [0.1, 9.9, 2.4],
                    [2.3, 3.4, 5.7]])
    vetor = np.array([1.5, 2.6, 3.7])
    
    L = np.tril(mat)
    vet_lr = nf.low_storage_rows(L)
    vet_lc = nf.low_storage_columns(L)
    
    expected = np.array([2.1, 25.89, 33.38])
    calc_alg8 = nf.storage_alg8(vet_lr,vetor)
    calc_alg10 = nf.storage_alg10(vet_lc,vetor)
    
    aae(calc_alg8, calc_alg10, decimal = 10)
    aae(calc_alg8, expected, decimal = 10)
    aae(calc_alg10, expected, decimal = 10)

def test_storage_calculation_8_dot():
    '''
    
    Dada uma matriz triangular inferior L, este teste compara o resultado da multiplicao entre o vetor lr com o vetor x, sendo lr o vetor armazenado a partir do esquema de linhas, respectivamente. Neste teste, comparamos com um resultado obtido por Numpy Dot.
    
    '''
    
    mat = np.array([[1.4, 2.9, 3.3],
                    [0.1, 9.9, 2.4],
                    [2.3, 3.4, 5.7]])
    vetor = np.array([1.5, 2.6, 3.7])
    
    L = np.tril(mat)
    vet_lr = nf.low_storage_rows(L)
    #vet_lc = nf.low_storage_columns(L)
    
    expected = np.dot(L,vetor)
    calc_alg8 = nf.storage_alg8(vet_lr,vetor)
    #calc_alg10 = nf.storage_alg5(vet_lc,vetor)
    
    aae(calc_alg8, expected, decimal = 10)
    #aae(calc_alg10, expected, decimal = 10)

def test_storage_calculation_10_dot():
    '''
    
    Dada uma matriz triangular inferior L, este teste compara o resultado da multiplicao entre o vetor lc com o vetor x, sendo lc o vetor armazenado a partir do esquema de colunas, respectivamente. Neste teste, comparamos com um resultado obtido por Numpy Dot.
    
    '''
    
    mat = np.array([[1.4, 2.9, 3.3],
                    [0.1, 9.9, 2.4],
                    [2.3, 3.4, 5.7]])
    vetor = np.array([1.5, 2.6, 3.7])
    
    L = np.tril(mat)
    #vet_lr = nf.low_storage_rows(L)
    vet_lc = nf.low_storage_columns(L)
    
    expected = np.dot(L,vetor)
    #calc_alg8 = nf.storage_alg3(vet_lr,vetor)
    calc_alg10 = nf.storage_alg10(vet_lc,vetor)
    
    #aae(calc_alg8, expected, decimal = 10)
    aae(calc_alg10, expected, decimal = 10)

def testfinal_storage_calculation_alg3_vs_alg5():
    '''
    
    Este teste compara o resultado obtido entre ur*x e uc*x, a partir de uma matriz aleatoria e um vetor aleatorio, com 5 casas decimais.
    
    '''
    
    matriz = np.around(15*np.random.rand(10,10), decimals = 5)
    vetor = np.around(15*np.random.rand(10), decimals = 5)
    
    vet_ur = nf.up_storage_rows(np.triu(matriz))
    vet_uc = nf.up_storage_columns(np.triu(matriz))
    
    expected = np.dot(np.triu(matriz),vetor)
    calc_alg3 = nf.storage_alg3(vet_ur,vetor)
    calc_alg5 = nf.storage_alg5(vet_uc,vetor)
    
    aae(calc_alg3, calc_alg5, decimal = 10)
    aae(calc_alg3, expected, decimal = 10)
    aae(calc_alg5, expected, decimal = 10)

def testfinal_storage_calculation_alg8_vs_alg10():
    '''
    
    Este teste compara o resultado obtido entre lr*x e lc*x, a partir de uma matriz aleatoria e um vetor aleatorio, com 5 casas decimais.
    
    '''
    
    matriz = np.around(12*np.random.rand(10,10), decimals = 5)
    vetor = np.around(18*np.random.rand(10), decimals = 5)
    
    vet_lr = nf.low_storage_rows(np.tril(matriz))
    vet_lc = nf.low_storage_columns(np.tril(matriz))
    
    expected = np.dot(np.tril(matriz),vetor)
    calc_alg8 = nf.storage_alg8(vet_lr,vetor)
    calc_alg10 = nf.storage_alg10(vet_lc,vetor)
    
    aae(calc_alg8, calc_alg10, decimal = 10)
    aae(calc_alg8, expected, decimal = 10)
    aae(calc_alg10, expected, decimal = 10)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                   Teste para matriz diagonal
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_diagonal1():
    '''

    Compara o resultado da multiplicao entre o vetor d e a matriz B obtido com a funcao diagonal1 com o resultado esperado.

    '''
    
    d = np.array([1., 2., 3.])
    B = np.array([[1., 0., 1.],[1., 1., 1.,],[0., 1., 1.]])
    expected = np.array([[1., 0., 1.],[2., 2., 2.],[0., 3., 3.]])
    result = nf.diagonal1(d,B)
    aae(result, expected, decimal=15)

#-------------------------------------------------------------------------------------------------------------------------

def test_diagonal2():
    '''
    
    Compara o resultado da multiplicao entre a matriz D e a matriz B com o resultado utilizando a funcao diagonal1 do arquivo de funcoes.
    
    '''
    
    d = np.array([1., 2., 3.])
    D = np.diag(d)
    B = np.array([[1., 0., 1.],[1., 1., 1.,],[0., 1., 1.]])
    expected = np.dot(D,B)
    
    final_result = nf.diagonal1(d,B)
    
    result1 = nf.mat_mat1(D,B)
    result2 = nf.mat_mat2_nelson(D,B)
    result3 = nf.mat_mat3_nelson(D,B)
    result4 = nf.mat_mat4_nelson1(D,B)
    result5 = nf.mat_mat4_nelson2(D,B)
    result6 = nf.mat_mat4_nelson3(D,B)
    result7 = nf.mat_mat5_nelson1(D,B)
    result8 = nf.mat_mat5_nelson2(D,B)
    result9 = nf.mat_mat5_nelson3(D,B)
    
    aae(final_result, expected, decimal=15)
    aae(result1, expected, decimal=15)
    aae(result2, expected, decimal=15)
    aae(result3, expected, decimal=15)
    aae(result4, expected, decimal=15)
    aae(result5, expected, decimal=15)
    aae(result6, expected, decimal=15)
    aae(result7, expected, decimal=15)
    aae(result8, expected, decimal=15)
    aae(result9, expected, decimal=15)

def test_diagonal3():
    '''

    Compara o resultado da multiplicao entre o vetor d e a matriz B obtido com a funcao diagonal2 com o resultado esperado.

    '''
    
    d = np.array([1., 2., 3.])
    B = np.array([[1., 0., 1.],[1., 1., 1.,],[0., 1., 1.]])
    expected = np.array([[1., 0., 3.],[1., 2., 3.],[0., 2., 3.]])
    result = nf.diagonal2(d,B)
    aae(result, expected, decimal=15)
    
def test_diagonal4():
    '''

    Compara o resultado da multiplicao entre a matriz B e a matriz D com o resultado utilizando a funcao diagonal2 do arquivo de funcoes.
    
    '''
    
    d = np.array([1., 2., 3.])
    D = np.diag(d)
    B = np.array([[1., 0., 1.],[1., 1., 1.,],[0., 1., 1.]])
    expected = np.dot(B,D)
    
    final_result = nf.diagonal2(d,B)
    
    result1 = nf.mat_mat1(B,D)
    result2 = nf.mat_mat2_nelson(B,D)
    result3 = nf.mat_mat3_nelson(B,D)
    result4 = nf.mat_mat4_nelson1(B,D)
    result5 = nf.mat_mat4_nelson2(B,D)
    result6 = nf.mat_mat4_nelson3(B,D)
    result7 = nf.mat_mat5_nelson1(B,D)
    result8 = nf.mat_mat5_nelson2(B,D)
    result9 = nf.mat_mat5_nelson3(B,D)
    
    aae(final_result, expected, decimal=15)
    aae(result1, expected, decimal=15)
    aae(result2, expected, decimal=15)
    aae(result3, expected, decimal=15)
    aae(result4, expected, decimal=15)
    aae(result5, expected, decimal=15)
    aae(result6, expected, decimal=15)
    aae(result7, expected, decimal=15)
    aae(result8, expected, decimal=15)
    aae(result9, expected, decimal=15)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                       Testes para produto de algoritmos 3,5 e 8,10 com um vetor
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test1_algoritmo3():
    '''

    Compara o resultado do produto entre um vetor (armazenado a partir de uma matriz triangular superior por linhas) e um vetor x dado com um resultado conhecido.
    
    '''
    
    mat = np.array([[1., 0., 1.],
                    [0., 1., 2.],
                    [0., 0., 2]])
    x = np.array([0., 1., 2.])
    
    expected = np.array([2., 5., 4.])
    mat_row = nf.up_storage_rows(mat)
    result = nf.storage_alg3(mat_row,x)
    
    aae(expected, result, decimal = 15)

def test2_algoritmo3():
    '''

    Compara o resultado do produto entre um vetor (armazenado a partir de uma matriz triangular superior por linhas) e um vetor x dado com um resultado experado, calculado com a funcao numpy.dot.
    
    '''
    
    mat = np.triu(np.random.rand(10,10))
    x = np.random.rand(10)
    
    expected = np.dot(mat,x)
    mat_row = nf.up_storage_rows(mat)
    result = nf.storage_alg3(mat_row,x)
    
    aae(expected, result, decimal = 15)

def test1_algoritmo8():
    '''
    
    Compara o resultado do produto entre um vetor (armazenado a partir de uma matriz triangular inferior por linhas) e um vetor x dado com um resultado conhecido.
    
    '''
    
    mat = np.array([[1., 0., 0.],
                    [2., 1., 0.],
                    [1., 2., 1]])
    x = np.array([-1., 1., 2.])
    
    expected = np.array([-1., -1., 3.])
    mat_row = nf.low_storage_rows(mat)
    result = nf.storage_alg8(mat_row,x)
    
    aae(expected, result, decimal = 15)

def test2_algoritmo3():
    '''
    
    Compara o resultado do produto entre um vetor (armazenado a partir de uma matriz triangular inferior por linhas) e um vetor x dado com um resultado experado, calculado com a funcao numpy.dot.
    
    '''
    
    mat = np.tril(np.random.rand(10,10))
    x = np.random.rand(10)
    
    expected = np.dot(mat,x)
    mat_row = nf.low_storage_rows(mat)
    result = nf.storage_alg8(mat_row,x)
    
    aae(expected, result, decimal = 15)
    
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                    Testes para matriz simetrica
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_symmetric1():
    '''
    Compara o resultado obtido para o calculo do vetor s_vec e o vetor x, utilizando a funcao do arquivo de funcoes, com o resultado esperado, a partir de uma matriz simetrica dada.
    
    '''
    
    # Input conhecido para matriz S, vetor x e resultado final
    matS = np.array([[2., 4., 3.],[4., 6., 5.],[3., 5., 4.]])
    vet_x = np.array([2., 4., 2.])
    expected = np.array([26., 42., 34.])
    # Realiza o armazenamento por linhas:
    vet_s = nf.up_storage_rows(matS)
    # Calcula o resultado a partir da funcao criada
    calculated = nf.symmetric(vet_s,vet_x)
    
    aae(expected, calculated, decimal = 5)
    
def test_symmetric2():
    '''

    Compara o resultado obtido para o calculo do vetor s_vec e o vetor x, criados de forma aleatoria, com as funcoes do numpy.dot e a funcao de matriz_vetor criada no arquivo de funcoes.
    
    '''
    
    # Cria o input aleatorio para uma dimensao N qualquer.
    N = 6
    # Inputs: 
    # (1) matriz S
    s = np.around(10*np.random.rand(1,N), decimals=2)
    S_mat = (s + s.T) - np.diag(s.diagonal())
    # (2) vetor x
    x = np.around(10*np.random.rand(N), decimals=2)
    # Calcula o produto entre S e x utilizando numpy.dot
    final_result = np.dot(S_mat,x)
    # Calcula o produto entre S e x utilizando as funcoes de mat_vec
    result1 = nf.matrix_vector1(S_mat,x)
    result2 = nf.matrix_vector2(S_mat,x)
    # Armazena a matriz S como vetor
    s_vetor = nf.up_storage_rows(S_mat)
    result_calc = nf.symmetric(s_vetor,x)
    
    aae(final_result, result1, decimal = 3)
    aae(final_result, result2, decimal = 3)
    aae(final_result, result_calc, decimal = 3)
    aae(result1, result_calc, decimal = 3)
    aae(result2, result_calc, decimal = 3)
    
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                               Teste para o exercicio de perfilagem sismica
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_seismic_profile():
    '''

    Calcula o resultado do vetor de tempos atraves da funcao numpy dot e compara com o resultado obtido atraves da funcao escrita no arquivo de funcoes. Para a comparacao, computamos o valor do vetor de tempos na funcao criada e estimamos o novo vetor vagarosidade. Por fim, comparamos os vetores vagarosidade real e calculada.
    
    '''
    
    # Valor de Z inicial
    z_zero = 1.15
    # Valor de diferenca de profundidade
    deltaz = 2.53
    
    # Matriz M disposta em:
    # coluna 1 - z_inicial
    # coluna 2:7 - deltaz a partir da segunda linha
    M = np.array([[z_zero, 0, 0, 0, 0, 0, 0],
                  [z_zero, deltaz, 0, 0, 0, 0, 0],
                  [z_zero, deltaz, deltaz, 0, 0, 0, 0],
                  [z_zero, deltaz, deltaz, deltaz, 0, 0, 0],
                  [z_zero, deltaz, deltaz, deltaz, deltaz, 0, 0],
                  [z_zero, deltaz, deltaz, deltaz, deltaz, deltaz, 0],
                  [z_zero, deltaz, deltaz, deltaz, deltaz, deltaz, deltaz]])
    
    # Valor do vetor vagarosidade
    s_real = np.array([1., 1.5, 3., 3.3, 3.6, 4.16, 5.21])
    
    # Calculo do vetor de tempos
    t = np.zeros(s_real.size)
    t_real = np.dot(M,s_real)
    
    # Calculo do vetor vagarosidade a partir dos valores z0 e deltaz, e do vetor de tempos
    # Este calculo e feito utilizando a funcao criada no arquivo de funcoes
    s_calc = nf.seismic_profile(z_zero, deltaz, t_real)
    
    # Comparacao de resultados:
    aae(s_real, s_calc, decimal = 15)

#-------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                    Testes para solucao de sistemas lineares com U e L
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test1_system_upper():
    '''
    
    Dado uma matriz triangular superior U e um vetor resultante y, este teste faz a comparacao do resultado obtido com o resultado conhecido.
    
    '''
    
    matsup = np.array([[2., 1., 1., 1.],
                       [0., 4., 2., 2.],
                       [0., 0., 2., 1.],
                       [0., 0., 0., 3.]])
    y = np.array([11., 12., 6., 6.])
    
    expected = np.array([3., 1., 2., 2.])
    calculated = nf.system_upper(matsup, y)
    
    aae(expected, calculated, decimal = 10)    

def test2_system_upper():
    '''
    
    Dado uma matriz triangular superior U e um vetor resultante y, este teste faz a comparacao do resultado obtido pela funcao Linalg Solve.
    
    '''
        
    mat = np.array([[2., 3., 4.],
                    [-1., 2., 2.],
                    [2., 2., 1.]])
    matsup = np.triu(mat)
    
    y = np.array([4., 3., 1.])
        
    expected = np.linalg.solve(matsup,y)
    calculated = nf.system_upper(matsup, y)
    
    aae(expected, calculated, decimal = 10)

def test1_system_lower():
    '''
    
    Dado uma matriz triangular superior U e um vetor resultante y, este teste faz a comparacao do resultado obtido com o resultado conhecido.
    
    '''
    
    matinf = np.array([[1., 0., 0., 0.],
                       [3., 2., 0., 0.],
                       [2., 1., 4., 0.],
                       [3., 2., 1., 2.]])
    y = np.array([2., 2., 6., 5.])
    
    expected = np.array([2., -2., 1., 1])
    calculated = nf.system_lower(matinf, y)
    
    aae(expected, calculated, decimal = 10)    

def test2_system_lower():
    '''
    
    Dado uma matriz triangular inferior L e um vetor resultante y, este teste faz a comparacao do resultado obtido pela funcao Linalg Solve.
    
    '''
        
    mat = np.array([[2., 3., 4.],
                    [-1., 2., 2.],
                    [2., 2., 1.]])
    matinf = np.tril(mat)
    
    y = np.array([4., 3., 1.])
    
    expected = np.linalg.solve(matinf,y)
    calculated = nf.system_lower(matinf, y)
    
    aae(expected, calculated, decimal = 10)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                Testes para Eliminacao de Gauss
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test1_gauss_elimination():
    '''
    
    Dado uma matriz A qualquer e um vetor V, este teste compara o sistema triangular obtido a partir da eliminacao de Gauss com um resultado esperado, levando em consideracao a matriz A e o vetor V.
    
    '''
    
    mat = np.array([[4., 2., 1., 2.],[2., 2., 5., 3.],[3., 1., 1., 4.],[3., 5., 1., 5.]])
    vet = np.array([1., 2., 1., 3.])
    
    expected_mat = np.array ([[4., 2., 1., 2.],
                              [0., 3.500, 0.250, 3.500],
                              [0., 0., 4.428571, 1.],
                              [0., 0., 0., 2.93548]])
    expected_vet = np.array([1., 2.25, 0.85714286, 0.516129])
    
    mat_modified, vet_modified = nf.gauss_elim(mat, vet)
    
    mat_calculated = np.triu(mat_modified)
    vet_calculated = vet_modified
    
    aae(expected_mat, mat_calculated, decimal = 3)
    aae(expected_vet, vet_calculated, decimal = 3)

def test2_gauss_elimination():
    '''
    
    Dada uma matriz A e um vetor X, calculamos o produto Ax = Y e computamos o valor de Y. A partir dele, aplicamos a eliminacao de Gauss para a matriz A e o vetor Y gerado. Dai, resolvemos o sistema triangular superior a partir da nova matriz A modificada e do novo vetor Y modificado.
    Este teste compara se o vetor X calculado a partir da solucao do sistema e igual ao vetor X real utilizado para computar o vetor Y inicialmente.
    
    '''
    
    # Dado: matriz A e vetor X
    matriz = np.array([[5., 2., -4., 1.],
                       [-4., 5., -2., 1.],
                       [2., 4., -4., -4.],
                       [-3., 5., 4., 1.]])
    x_real = np.array([2., 3., 1., -2.])
    
    # Computa o vetor Y
    vector = np.dot(matriz,x_real)
    
    # Aplica a eliminacao de Gauss
    # Obtem matriz A e vetor Y (modificados)
    new_mat, new_vec = nf.gauss_elim(matriz, vector)
    
    # Calcula o vetor solucao do sistema linera A(mod) X = Y(mod)
    x_solve = nf.system_upper(new_mat,new_vec)
    
    # Verifica se o vetor X calculado e igual ao vetor X real
    aae(x_real, x_solve, decimal = 15)

def test1_permutacao():
    '''
    
    Dada uma matriz A qualquer, este teste compara o resultado da funcao de permutacao com um resultado esperado, para um valor de K conhecido.
    
    '''
    
    # Cria a matriz A
    A = np.array([[5., 2., 4., 1.],
                  [4., 3., 2., 1.],
                  [2., 7., 1., 4.],
                  [6., 3., 5., 1.]])
    
    K0 = 0
    K1 = 1
    K2 = 2
    K3 = 3
        
    # Primeira permutacao - Todas as linhas da primeira coluna
    expected_mat_k0 = np.array([[6., 3., 5., 1.],
                                [4., 3., 2., 1.],
                                [2., 7., 1., 4.],
                                [5., 2., 4., 1.]])
    expected_p0 = np.array([3, 1, 2, 0])
    
    p0, calc_mat_k0 = nf.permut(A,K0)
    
    # Segunda permutacao - Todas as linhas (a partir da L2) da segunda coluna
    expected_mat_k1 = np.array([[5., 2., 4., 1.],
                                [2., 7., 1., 4.],
                                [4., 3., 2., 1.],
                                [6., 3., 5., 1.]])
    expected_p1 = np.array([0, 2, 1, 3])
    
    p1, calc_mat_k1 = nf.permut(A,K1)
    
    # Terceira permutacao - Todas as linhas (a partir da L3) da terceira coluna
    expected_mat_k2 = np.array([[5., 2., 4., 1.],
                                [4., 3., 2., 1.],
                                [6., 3., 5., 1.],
                                [2., 7., 1., 4.]])
    expected_p2 = np.array([0, 1, 3, 2])
    
    p2, calc_mat_k2 = nf.permut(A,K2)
    
    # Quarta permutacao - Ultimo elemento (retorna para a matriz original)
    expected_mat_k3 = np.array([[5., 2., 4., 1.],
                                [4., 3., 2., 1.],
                                [2., 7., 1., 4.],
                                [6., 3., 5., 1.]])
    expected_p3 = np.array([0, 1, 2, 3])
    
    p3, calc_mat_k3 = nf.permut(A,K3)
    
    # Verifica os resultados para a matriz de permutacao
    aae(expected_mat_k0, calc_mat_k0, decimal = 15)
    aae(expected_mat_k1, calc_mat_k1, decimal = 15)
    aae(expected_mat_k2, calc_mat_k2, decimal = 15)
    aae(expected_mat_k3, calc_mat_k3, decimal = 15)
    
    # Verifica o resutlado para os vetores de indice de linhas
    aae(expected_p0, p0)
    aae(expected_p1, p1)
    aae(expected_p2, p2)
    aae(expected_p3, p3)
    
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                   Teste para matriz inversa
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test_matriz_inversa0():
    '''
    
    Dada uma matriz qualquer quadrada de dimensao NxN, este teste calcula a matriz inversa e compara se o produto A*A(inv) e igual a matriz identidade de dimensao N.
    
    '''
    
    # Define a matriz
    mat = np.array([[1., 1., 1., 1.],
                    [1., 4., 4., 6.],
                    [1., 2., 4., 9.],
                    [4., 4., 2., 1.]])
    
    #Calcula a matriz inversa
    inv = nf.inversa(mat)
    
    # Computa a matriz identidade por:
    N = mat.shape[0]
    # (1) Cria a matriz identidade de dimensao N
    ID = np.identity(N)
    # (2) Computa A * A(inv)
    A_vs_Ainv = np.dot(mat, inv)
    # (2) Computa A * A(inv)
    Ainv_vs_A = np.dot(inv,mat)
    
    # Verifica o resultado calculado
    aae(ID, A_vs_Ainv, decimal = 15)
    aae(ID, Ainv_vs_A, decimal = 15)
    aae(Ainv_vs_A, A_vs_Ainv, decimal = 15)

def test_matriz_inversa0_myfunctions():
    '''
    
    Dada uma matriz qualquer quadrada de dimensao NxN, este teste calcula a matriz inversa e compara se o produto A*A(inv) (obtido a partir das funcoes de calculo entre matrizes por produto externo) e igual a matriz identidade de dimensao N.
    '''
    
    # Define a matriz
    mat = np.array([[1., 0., 1., 2.],
                    [2., 1., 0., 3.],
                    [3., 1., 1., 4.],
                    [4., 2., 1., 1.]])
    
    #Calcula a matriz inversa
    inv = nf.inversa(mat)
    
    # Computa a matriz identidade
    N = mat.shape[0]
    ID = np.identity(N)
    # Computa o produto A * A(inv)
    A_vs_Ainv1 = nf.mat_mat5_nelson1(mat, inv)
    A_vs_Ainv2 = nf.mat_mat5_nelson2(mat, inv)
    A_vs_Ainv3 = nf.mat_mat5_nelson3(mat, inv)
    
    # Verifica o resultado calculado
    aae(ID, A_vs_Ainv1, decimal = 15)
    aae(ID, A_vs_Ainv2, decimal = 15)
    aae(ID, A_vs_Ainv3, decimal = 15)
    aae(A_vs_Ainv1, A_vs_Ainv2, decimal = 15)
    aae(A_vs_Ainv1, A_vs_Ainv3, decimal = 15)
    aae(A_vs_Ainv2, A_vs_Ainv3, decimal = 15)
    
def test_matriz_inversa1():
    '''
    
    Dada uma matriz A qualquer de dimensao NxN, compara o resultado da matriz inversa calculada com o resultado obtido pela funcao Linalg Inv.
    
    '''
    
    # Define a matriz
    matrizA = np.array([[1., 1., 1.],
                        [2., 2., 3.],
                        [1., 2., 1.]])
    # Calcula a inversa partir da funcao
    inv_calc = nf.inversa(matrizA)
    
    # Calcula a inversa utilizando Linalg
    inv_real = np.linalg.inv(matrizA)
    
    # Compara os resultados
    aae(inv_calc, inv_real, decimal = 15)


def test_matriz_inversa2():
    '''
    
    Dada uma matriz A quadrada, compara a matriz inversa calculada com um resultado conhecido.
    
    '''
    
    # Define a matriz quadrada
    mat = np.array([[1., 1.],[2., 3.]])
    # Define o resultado conhecido
    inv = np.array([[3., -1.],[-2., 1.]])
    # Calcula a inversa
    calc = nf.inversa(mat)
    
    aae(inv, calc, decimal = 15)

def test_matriz_inversa3():
    '''
    
    Dada uma matriz A qualquer, o teste verifica se a inversa da matriz inversa calculada e igual a matriz original.
    
    '''
    
    # Dada a matriz A
    matriz_A = np.array([[1., 0., 1., 2.],
                         [0., 1., 0., 0.],
                         [0., 1., 1., 0.],
                         [1., 2., 1., 1.]])
    
    # Dada a inversa conhecida
    inv = np.array([[-1., -3., -1.,  2.],
                    [ 0.,  1.,  0.,  0.],
                    [ 0., -1.,  1.,  0.],
                    [ 1.,  2.,  0., -1.]])
    
    # Dada a inversa calculada
    inv_calc = nf.inversa(matriz_A)
    
    # Calcula novamente a matriz inversa para:
    # (1) Inversa conhecida
    inv_inv_real = nf.inversa(inv) 
    # (2) Inversa calculada
    inv_inv_calc = nf.inversa(inv_calc)
    
    # Verifica os resultados
    # (a) Calculo da inversa
    # (b) Segunda inversa para a inversa real
    # (c) Segunda inversa para a inversa calculada
    aae(inv, inv_calc, decimal = 5)
    aae(inv_inv_real, matriz_A, decimal = 5)
    aae(inv_inv_calc, matriz_A, decimal = 5)

def test_matriz_inversa4():
    '''
    
    Dada uma matriz A qualquer, este teste calcula a matriz inversa de A e a matriz inversa da transposta de A, e compara se a INV(A.T) e igual a INV(A).T.
    
    '''
    
    # Define a matriz A
    matrizA = np.array([[1., 9., 0., 2.],
                        [3., 3., 2., 1.],
                        [9., 9., 1., 4.],
                        [5., 1., 5., 6.]])
    # Define a transposta de A
    transpostaA = matrizA.T
    
    # Calcula a inversa da matriz A
    # Calcula a inversa da transposta
    inversa_A = nf.inversa(matrizA) 
    inversa_AT = nf.inversa(transpostaA)
    
    # Compara os resultados
    aae(inversa_A, inversa_AT.T, decimal = 15)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                                    Testes para Decomposicao LU
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test1_lu():
    '''
    
    Dada uma matriz A qualquer quadrada, este tete compara a decomposicao LU calculada pelas funcoes criadas com as matrizes L e U conhecidas.
    
    '''
    
    # Define a matriz A
    A = np.array([[2., 4., 2.],
                  [3., 4., 1.],
                  [5., 2., 3.]])
    
    # Matrizes L e U conhecidas
    L_real = np.array([[1.0, 0., 0.],
                       [1.5, 1., 0.],
                       [2.5, 4., 1.]])
    U_real = np.array([[2., 4., 2.],
                       [0., -2., -2.],
                       [0., 0., 6.]])
    
    # Calcula as novas matrizes L e U    
    L_calc, U_calc = nf.gauss_lu(A)
    
    # Verifica os resultados
    aae(L_real, L_calc, decimal = 15)
    aae(U_real, U_calc, decimal = 15)

def teste2_lu():
    '''
    
    Dada uma matriz A quadrada, de dimensao NxN, primeiramente este teste computa a matriz A modificada a partir da funcao de eliminacao Gaussiana criada no arquivo de funcoes. Posteriormente, computa as matriz L e U e verifica se o produto L*U e igual a matriz A original.
    
    '''
    
    # Cria a matriz A
    A = np.array([[1., 2., 1., -1.],
                  [-2., 1., 1., 2.],
                  [1., 2., 0., -3.],
                  [3., 3., -1., 2.]])
    
    # Computa as matrizes L e U
    L, U = nf.gauss_lu(A)
    
    # Computa o produto entre utilizando
    # (1) Numpy Dot de L,U
    # (2) Funcoes de produto de matrizes
    
    LU_real = np.dot(L,U)
    LU1 = nf.mat_mat5_nelson1(L,U)
    LU2 = nf.mat_mat5_nelson2(L,U)
    LU3 = nf.mat_mat5_nelson3(L,U)
    
    # Verifica os resultados A = L*U
    aae(A, LU_real, decimal = 15)
    aae(A, LU1, decimal = 15)
    aae(A, LU2, decimal = 15)
    aae(A, LU3, decimal = 15)

def test3_lu_solve_system():    
    '''
    
    Dada uma matriz A qualquer, este teste compara o resultado do sistema linear A*x = y com a solucao obtida ataves da Decomposicao LU, uma vez que LU*x = y. Primeiro denomina-se U*x = w, dai resolve-se o sistema triangular inferior para L*w = y. Posteriormente, com o valor de w, resolve-se o sistema triangular superior para U*x = w.
    
    '''
    
    # Cria a matriz A
    A = np.array([[2., 4., 1.],
                  [3., 2., 2.],
                  [1., 4., 3.]])
    # Cria o vetor y
    y = np.array([1., 2., -1.])
    
    # Solucao Real:
    x_real = np.linalg.solve(A,y)
    
    # Cria as matriz L e U
    L, U = nf.gauss_lu(A)
    
    # Resolvendo o sistema para Lw = y
    w = nf.system_lower(L,y)
    # Resolvendo o sistema para Ux = w
    x_calc = nf.system_upper(U,w)
    
    aae(x_real, x_calc, decimal = 15)

def test4_make_lu():
    ''' 
    
    Dada uma matriz quadrada, este teste modifica a matriz a partir da Eliminacao de Gauss e verifica se a matriz LU criada e verdadeira, uma vez que o produto L*U deve ser igual a matriz A original.
    
    '''
    
    # Cria a matriz quadrdada
    A = np.array([[1., 2., 2.],
                  [5., 5., 1.],
                  [1., 1., 9.]])
    
    # Aplica eliminacao de Gauss
    C = nf.lu_simple(A)
    
    # Decompoem L e U
    L, U = nf.lu_maker(C)
    
    LU_real = np.dot(L,U)
    LU1 = nf.mat_mat5_nelson1(L,U)
    LU2 = nf.mat_mat5_nelson2(L,U)
    LU3 = nf.mat_mat5_nelson3(L,U)
    
    # Verifica os resultados A = L*U
    aae(A, LU_real, decimal = 15)
    aae(A, LU1, decimal = 15)
    aae(A, LU2, decimal = 15)
    aae(A, LU3, decimal = 15)

def test4_make_lu_2():
    ''' 
    Dada uma matriz quadrada, este teste modifica a matriz a partir da Eliminacao de Gauss e verifica se a matriz LU criada e igual as matriz L e U computadas a partir do Scipy Linalg.
    
    '''
    
    # Cria a matriz quadrdada
    A = np.array([[5., -1., 2.],
                  [3., -4., 1.],
                  [2., 6., -2.]])
    
    # Aplica eliminacao de Gauss
    C = nf.lu_simple(A)
    
    # Decompoem L e U
    # (1) Atraves da funcao criada
    L_calc, U_calc = nf.lu_maker(C)
    # (2) Atraves da funcao Scipy    
    P, L_real, U_real = lu(A)
    
    # Calcula o produto LU para as matrizes calculadas
    LU_calc = np.dot(L_calc, U_calc)
    # Calcula o produto LU para a matriz de Scipy
    LU_real = np.dot(L_real, U_real)
    # Calcula o produto de P vs LU
    # Uma vez que PLU = A
    P_vs_LU = np.dot(P,LU_real)

    # Verifica se as matrizes L e U sao verdadeiras
    aae(LU_calc, A, decimal = 15)
    aae(P_vs_LU, A, decimal = 15)

def test5_lu_pivoting():
    '''
    
    Dada uma matriz A, este teste compara se a matriz LU gerada e igual a matriz A original, utilizando a funcao criada de decomposicao LU com pivotamento.
    
    '''
    
    # Cria a matriz A
    A = np.array([[1., 2., 3., 3.],
                  [5., 3., 2., 1.],
                  [9., 8., 7., 7.],
                  [3., 3., 5., 1.]])
    
    # Computa a matriz C e P
    P, C = nf.lu_pivoting(A)
    
    # Cria as matrizes L e U
    L, U = nf.lu_maker(C)
    
    # Computa a matriz A permutada
    PA = np.dot(P,A)
    LU_real = np.dot(L,U)
    LU1 = nf.mat_mat5_nelson1(L,U)
    LU2 = nf.mat_mat5_nelson2(L,U)
    LU3 = nf.mat_mat5_nelson3(L,U)
    
    # Verifica os resultados A = L*U
    aae(PA, LU_real, decimal = 15)
    aae(PA, LU1, decimal = 15)
    aae(PA, LU2, decimal = 15)
    aae(PA, LU3, decimal = 15)

def test5_lu_pivoting_solve():
    '''
    
    Dada uma matriz A e um vetor Y, este teste compara o resultado da funcao de decomposicao LU com solucao para um sistema linear (definido por A*x = Y --> LU*x = P*y ) com o resultado obtido atraves da funcao Numpy.
    
    '''
    
    # Define a matriz A e o vetor Y
    A = np.array([[4., 5., 5.],
                  [3., 6., 5.],
                  [5., 6., 2.]])
    Y = np.array([3., 9., 11.])
    
    # Computa a solucao real
    X_Real = np.linalg.solve(A,Y)
    
    # Computa a solucao a partir da funcao criada
    X_Calc = nf.lu_pivoting_solve(A,Y)
    
    # Compara os resultados
    aae(X_Real, X_Calc, decimal = 15)

def test6_lu_vs_scipy():
    '''
    
    Dada uma matriz A quadrada de dimensao N, este teste compara o resultado obtido a partir da funcao criada no arquivo de funcoes com o resultado obtido a partir da funcao Scipy Linalg (LU), para as matrizes de permutacao (P), triangular inferior (L) e superior (P).
    
    Ps. A matriz de permutacao obtida a partir da funcao criada e igual a matriz de permutacao transposta obtida a partir da funcao Linalg.
    
    '''
    
    # Computa a matriz A e o vetor Y
    A = np.array([[9., 4., -3., -2.],
                  [3., -2., -1., 6.],
                  [-5., 5., 5., -1.],
                  [10., 1., 2., 10]])
    
    # Computa P, L e U a partir da funcao criada
    myP, myC = nf.lu_pivoting(A)
    myL, myU = nf.lu_maker(myC)
    
    P_real, L_real, U_real = lu(A)
    
    aae(myP, P_real.T, decimal = 15)
    aae(myL, L_real, decimal = 15)
    aae(myU, U_real, decimal = 15)
    
    
def test7_lu_vs_scipy_solve_system():
    '''
    
    Dada uma matriz A quadrada de dimensao N, este teste verifica se a solucao do sistema linear encontrado a partir das funcoes de Decomposicao LU criadas no arquivo de funcoes e igual a solucao encontrar para um sistema lienar utilizando as matriz P, L e U criadas a partir da funcao Scipy Linalg.
    
    '''
    
    # Computa a matriz A e o vetor Y
    A = np.array([[9., 4., -3., -2.],
                  [3., -2., -1., 6.],
                  [-5., 5., 5., -1.],
                  [10., 1., 2., 10]])
    Y = np.array([-2., 2., 2., 1.])
    
    # Calcula a verdadeira solucao do sistema
    Solution = np.linalg.solve(A,Y)
    
    # Computa a solucao a partir da funcao criada
    X_Calc = nf.lu_pivoting_solve(A,Y)
    
    # Computa P, L e U a partir da funcao Scipy Linalg
    P_real, L_real, U_real = lu(A)
    
    # Seja o sistema Ax = y, de modo que A pode ser escrito como LU. Entao, LUx = Py.
    # Como P obtido por Scipy Linalg(LU) e igual a P transposto obtido a partir da funcao criada,
    # o vetor Y deve ficar pos multiplicado. Logo: LUx = yP.
    # Passo 1: resolver Lw = yP
    YP = np.dot(Y,P_real)
    W = nf.system_lower(L_real, YP)
    # Passo 2: resolver Ux = W
    X_Calc2 = nf.system_upper(U_real, W)
    
    aae(Solution, X_Calc, decimal = 15)
    aae(Solution, X_Calc2, decimal = 15)
    aae(X_Calc, X_Calc2, decimal = 15)


# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                               Testes criados para a decomposicao LDT
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test1_ldlt():
    '''

    Dada uma matriz S simetria, este teste verifica se o produto dos elementos da decomposicao LDL(T) e igual a matriz S original, desde que S admita decomposicao LU.

    '''
    
    # Cria a matriz S
    S = np.array([[2., 5., 4., 3.],
                  [5., 8., 6., 2.],
                  [4., 6., 4., 5.],
                  [3., 2., 5., 2.]])
    
    # Computa a matriz L e o vetor d
    # Cria a matriz diagonal a partir de D
    L, d = nf.ldlt_decomposition(S)
    D = np.diag(d)
    
    # Computa o resultado LDL(T) em dois passos:
    # i) Calcula L*D
    LD = np.dot(L,D)
    # ii) Calcula LD*L(T)
    LDLT = np.dot(LD, L.T)
    
    # Verifica a condicao de igualdade
    aae(S, LDLT, decimal = 15)
    
def test2_ldlt():
    '''
    
    Dada uma matriz S simetrica, este teste verifica se a matriz L criada pela decomposicao LDL(T) e igual a matriz L criada pela decomposicao LU, uma vez que S admite decomposicao LU. Neste teste, utilizamos as duas funcoes para criar as matrizes L e U.
    
    '''
    
    # Cria a matriz S
    matS = np.array([[5., 4., 7., 7., 4.],
                     [4., 3., 6., 6., 3.],
                     [7., 6., 2., 2., 6.],
                     [7., 6., 2., 4., 6.],
                     [4., 3., 6., 6., 3.]])
    
    # Computa a matriz L via LDL(T)
    L_from_LDLT, vecD = nf.ldlt_decomposition(matS)
    
    # Computa a matriz L via LU
    L_from_LU, U = nf.gauss_lu(matS)
    
    # Verifica se as matrizes L sao iguais
    aae(L_from_LDLT, L_from_LU, decimal = 15)
    
    
def test3_ldlt_solve():
    '''
    
    Dada uma matriz S simetrica, este aplica a funcao para solucao de sistema linear a partir da decomposicao LDL(T) e verifica se a solucao calculada e igual a solucao real obtida pela funcao Numpy Linalg Solve.
    
    '''
    
    # Cria a matriz S e o vetor Y
    S = np.array([[4., 12., -16.],
                  [12., 37., -43.],
                  [-16., -43., 98.]])
    Y = np.array([2., 1., 5.])
    
    # Calcula a solucao real
    X_Real = np.linalg.solve(S,Y)
    
    # Calcula a solucao por LDL(T)
    X_Calc = nf.ldlt_decomposition_solve(S,Y)
    
    #Verifica os resultados calculados
    aae(X_Real, X_Calc, decimal = 5)

def test3_ldlt_inversa():
    '''
    
    Dada uma matriz S simetrica, este teste computa a matriz inversa de S a partir da decomposicao LDLt e verifica o resultado calculado com o resutlado da funcao Linalg.
    
    '''
    
    # Matriz S
    S = np.array([[4., 12., -16.],
                  [12., 37., -43.],
                  [-16., -43., 98.]])
    
    # Calcula a matriz inversa verdadeira
    invS_real = np.linalg.inv(S)
    
    # Calcula a matriz inversa via LDLt
    invS_calc = nf.inversa_ldlt(S)
    
    # Verifica se as duas matrizes sao iguais
    aae(invS_real, invS_calc, decimal = 5)

def test4_ldlt_inversa():
    '''
    
    Dada uma matriz S simetrica, este teste computa a matriz inversa de S a partir da decomposicao LDLt e verifica o resultado calculado com o resutlado da funcao de matriz inversa criada no arquivo de funcoes.
    
    '''
    
    # Matriz S
    S = np.array([[7., 3., 4., 6.],
                  [3., -1., 1., 2.],
                  [4., 1., 3., 3.],
                  [6., 2., 3., 5.]])
    
    # Calcula a matriz inversa via LDLt
    invS_calc1 = nf.inversa_ldlt(S)
    
    # Calcula a matriz inversa via funcao
    invS_calc2 = nf.inversa(S)    
    
    # Verifica se as duas matrizes sao iguais
    aae(invS_calc1, invS_calc2, decimal = 5)

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#                                        Testes criados para a decomposiceos de Cholesky
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def test1_cholesky():
    '''
    
    Dada uma matriz S simetrica e positiva definida, este teste compara se a matriz resultante obtida atraves da Decomposicao de Cholesky obedece a propriedade de G*G(t) = S.
    Neste caso, utilizamos a matirz identidade.
    
    '''
    
    # Cria a matriz identidade
    ID = np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    
    # Calculo da matriz resultante
    G_calc = nf.cholesky(ID)
    
    # Calculo do produto real
    ID_real = np.dot(G_calc, G_calc.T)
    
    # Calculo do produto atraves das funcoes de multiplicacao de matrizes
    ID1_calc = nf.mat_mat5_nelson1(G_calc,G_calc.T)
    ID2_calc = nf.mat_mat5_nelson2(G_calc,G_calc.T)
    ID3_calc = nf.mat_mat5_nelson3(G_calc,G_calc.T)
    
    # Verifica os resultados
    aae(ID, ID_real, decimal = 5)
    aae(ID, ID1_calc, decimal = 5)
    aae(ID, ID2_calc, decimal = 5)
    aae(ID, ID3_calc, decimal = 5)

def test2_cholesky():
    '''
    
    Dada uma matriz S simetrica e positiva definida, este teste compara se a matriz resultante obtida atraves da funcao de Decomposicao de Cholesky e igual a matriz resultante obtida atraves da funcao Linalg Cholesky, uma vez que L e unica.
    
    '''
    
    # Cria a matriz simetrica
    S = np.array([[1.55, 0.45, 1.11, 1.09],
                  [0.45, 0.30, 0.62, 0.58],
                  [1.11, 0.62, 2.01, 1.44],
                  [1.09, 0.58, 1.44, 1.26]])

    # Computa os resultados via funcao Linag_Cholesky e via funcao Cholesky
    L_real = np.linalg.cholesky(S)
    L_calc = nf.cholesky(S)
    
    aae(L_real, L_calc, decimal = 10)

def test3_cholesky_inversa():
    '''
    
    Dada uma matriz S simetrica, este teste verifica se a matriz inversa calculada o codigo de decomposicao de Cholesky e igual a matriz inversa calculada pela funcao Linalg Inv.
    
    '''
    
    # Cria a matriz simetrica
    S = np.array([[1.5, 0.4, 1.1, 1.0],
                  [0.4, 0.3, 0.6, 0.5],
                  [1.1, 0.6, 2.0, 1.4],
                  [1.0, 0.5, 1.4, 1.2]])
    
    # Computa as matrizes inversas
    inv_real = np.linalg.inv(S)
    inv_calc = nf.inversa_cholesky(S)
    
    # Verifica os resultados
    aae(inv_real, inv_calc, decimal = 10)

def test4_cholesky_inversa():
    '''
    
    Dada uma matriz S simetrica, este teste verifica se a matriz inversa calculada o codigo de decomposicao de Cholesky obdece a propriedade de S*S(inv) = I.
    
    '''
    
    # Cria a matriz simetrica
    S = np.array([[1., 1., 1., 1.],
                  [1., 2., 1., 2.],
                  [1., 1., 3., 2.],
                  [1., 2., 2., 4.]])
    
    # Computa a matriz inversa
    S_inv = nf.inversa_cholesky(S)
    
    # Computa o produto verdadeiro entre matrizes
    S_original = np.dot(S, S_inv)
    #Computa o produto entre as matrizes via funcoes de produto de matrizes
    S_initial1 = nf.mat_mat5_nelson1(S,S_inv)
    S_initial2 = nf.mat_mat5_nelson2(S,S_inv)
    S_initial3 = nf.mat_mat5_nelson3(S,S_inv)
    
    # Cria a matriz identidade
    N = S.shape[0]
    matID = np.identity(N)
    
    # Verifica os resultados
    aae(matID, S_original, decimal = 10)
    aae(matID, S_initial1, decimal = 10)
    aae(matID, S_initial2, decimal = 10)
    aae(matID, S_initial3, decimal = 10)
    
def test5_cholesky_solve1():
    '''
    
    Dada uma matriz S simetrica e um vetor Y, este teste calcula a solucao do sistema linear atraves do codigo de Decomposicao de Cholesky, implementado no arquivo de funcoes e compara com a solucao da funcao Linalg Solve. Neste primeiro caso, utilizamos a matriz Identidade.
    
    '''
    
    # Cria a matriz simetrica
    S = np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]])
    
    # Cria o vetor Y
    Y = np.array([1., 1., 1., 1.])
    
    # Calcula da solucao verdadeira
    x_real = np.linalg.solve(S,Y)
    # Calculo da solucao via Cholesky
    x_calc = nf.cholesky_solve(S,Y)
    
    aae(x_real, x_calc, decimal = 10)

def test6_cholesky_solve2():
    '''
    
    Dada uma matriz S simetrica e um vetor Y, este teste calcula a solucao do sistema linear atraves do codigo de Decomposicao de Cholesky, implementado no arquivo de funcoes e compara com a solucao da funcao Linalg Solve.
    
    '''
    
    # Cria a matriz simetrica
    S = np.array([[1., 1., 1., 1.],
                  [1., 2., 1., 2.],
                  [1., 1., 3., 2.],
                  [1., 2., 2., 4.]])
    
    # Cria o vetor Y
    Y = np.array([1., 2., 2., 1.])
    
    # Calcula da solucao verdadeira
    x_real = np.linalg.solve(S,Y)
    # Calculo da solucao via Cholesky
    x_calc = nf.cholesky_solve(S,Y)
    
    aae(x_real, x_calc, decimal = 10)

# ----------------------------------------------------------------------------------------------------------------------------------------
def test1_determinante():
    '''
    
    Dada uma matriz A qualquer, este teste realiza o codigo criado para computar o determinante das submatrizes de uma matriz e compara com um resultado esperado.
    Neste primeiro caso, A matriz corersponde a matriz identidade.
    
    '''
    
    # Cria a matriz A - matriz identidade
    A = np.identity(5)
    
    # Resultado esperado
    det_expected = np.ones(5)
    
    # Calcula o resultado a partir da funcao
    det_calculated = nf.determinante(A)
    
    # Verificacao
    aae(det_expected, det_calculated, decimal = 15)
    
def test2_determinante():
    '''
    
    Dada uma matriz A qualquer, este teste realiza o codigo criado para computar o determinante das submatrizes de uma matriz e compara com um resultado esperado.
    Neste primeiro caso, A matriz corersponde a uma matriz de zeros.
    
    '''
    
    # Cria a matriz A - matriz identidade
    A = np.zeros([7,7])
    
    # Resultado esperado
    det_expected = np.zeros(7)
    
    # Calcula o resultado a partir da funcao
    det_calculated = nf.determinante(A)
    
    # Verificacao
    aae(det_expected, det_calculated, decimal = 15)

def test3_determinante():
    '''
    
    Dada uma matriz A qualquer, este teste realiza o codigo criado para computar o determinante das submatrizes de uma matriz e compara com um resultado esperado.
    Neste primeiro caso, A matriz corersponde a uma matriz cheia de dimensao NxN.
    
    '''
    
    # Cria a matriz A - matriz identidade
    A = np.array([[2., 1., 1.],
                  [1., 3., 2.],
                  [1., 2., 4.]])
    
    # Calculo de determinantes das submatrizes
    det1 = 2.
    det2 = 5.
    det3 = 13.
    
    # Resultado esperado
    det_expected = np.array([det1, det2, det3])
    
    # Calcula o resultado a partir da funcao
    det_calculated = nf.determinante(A)
    
    # Verificacao
    aae(det_expected, det_calculated, decimal = 15)

def test4_determinante():
    '''
    
    Verifica se a matriz dada e uma matriz simetrica, de fato. Caso A nao seja simetrica, mas seja quadrada, indica o erro.
    
    '''
    
    # Cria a matriz A - matriz identidade
    A = np.array([[3., 1., 1.],
                  [1., 3., 1.],
                  [1., 2., 1. ]])
    
    # Verificacao
    raises (AssertionError, nf.determinante, symmetric_matrix = A)

def test5_determinante():
    '''
    
    Verifica se a matriz dada e uma matriz quadrada, de fato. Caso A nao seja quadrada, indica o erro.
    
    '''
    
    # Cria a matriz A - matriz identidade
    A = np.array([[3., 1., 1.],
                  [1., 3., 1.]])
    
    # Verificacao
    raises (AssertionError, nf.determinante, symmetric_matrix = A)
# ----------------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
