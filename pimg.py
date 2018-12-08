'''
    ndarray - imagem
    string - nome de arquivo
    os.listdir(“.”) - lista os arquivos do diretório “.”, ou seja, do diretório atual.
    os.chdir(“..”) - Acessa o diretório “..”, ou seja, o pai do diretório atual.
    os.getcwd() - Mostra o nome do diretório atual.

    referencias:
    # http://www.galirows.com.br/meublog/opencv-python/opencv2-python27/capitulo1-basico/mostrar-imagem-opencv-python/
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.html
    # https://matplotlib.org/users/image_tutorial.html
    # http://www.scipy-lectures.org/advanced/image_processing/
    # https://python-guide-pt-br.readthedocs.io/pt_BR/latest/scenarios/imaging.html
    # https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html
    # http://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/

'''
'''
    Aluno: Raul Andrade
    Disciplina: Processamento de Imagem
'''

# 1
import matplotlib.pyplot as plt
import matplotlib.image as mpimp
from matplotlib import colors
import numpy as np
from numpy import ma
import os

def imread(arq):  # 2
    ''' mpimp.imread() retorna um tipo ndarray. '''
    img = mpimp.imread(arq, 'uint8')
    return img

def nchannels(img):  # 3
    ''' retorna o número de canais da imagem de entrada. '''
    try:
        return img.shape[2]
    except IndexError:
        return 1

def size(img):  # 4
    ''' shape[0] = ALTURA(height) e shape[1] = LARGURA(width). '''
    return [img.shape[0], img.shape[1]]


def rgb2gray(img):  # 5
    ''' recebe uma imagem RGB e retorna outra imagem 
    igual à imagem de entrada convertida para escala de cinza. '''
    try:
        return np.dot(img, [0.299, 0.587, 0.114])
    except ValueError:
        return img

def imreadgray(arq):  # 6
    ''' recebe um nome de arquivo e retorna 
    a imagem lida em escala de cinza. '''
    img = mpimp.imread(arq, 'uint8')
    if nchannels(img) > 1:
        return rgb2gray(img).astype('uint8')
    return img.astype('uint8')
    
def imshow(img):  # 7
    ''' recebe uma imagem como parâmetro e a exibe. '''
    image = plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.show(image)

def thresh(img, alpha):
    ''' apaga o pixel onde é menor que o limiar 
    e acende o pixel onde é maior q o limiar. '''
    # a numpy permite fazer esse tipo de operacao
    # cada pixel da imagem for >= ao alpha vira true caso contrario é false
    # entao true * 255 = 255 e false * 255 = 0
    return ((img >= alpha) * 255)

def negative(img):  # 9
    ''' recebe uma imagem e retorna sua negativa. ''' 
    # é necessario usar o astype('uint8') pois com -1 ficou fora do uint8
    # dai usando essa funcao ele faz o mapeamento de volta pro uint8
    return (img * (-1)).astype('uint8')

def contrast(f, r, m):  # 10
    ''' Recebe uma imagem f, real r e um real m
        Retorna uma imagem g = r(f - m) + m. '''
    # pegando o tamanho
    N, M = size(f)
    # fazendo a copia da matrix
    newImg = np.copy(f) 
    if nchannels(f) > 1: 
        for i in np.arange(N):
            for j in np.arange(M):
                #fazendo o calculo do contrast pra rgb
                # lembrando que pixels menores que 0 sao mapeados para 0
                # e pixels maiores que 255 sao mapeados para 255
                newImg[i,j,0] = min(max(r * (newImg[i,j,0] - m) + m, 0), 255) # r
                newImg[i,j,1] = min(max(r * (newImg[i,j,1] - m) + m, 0), 255) # g
                newImg[i,j,2] = min(max(r * (newImg[i,j,2] - m) + m, 0), 255) # b
    else: #imagem em escala de cinza
        for i in np.arange(N):
            for j in np.arange(M):
                newImg[i,j] = min(max(r * (newImg[i,j] - m) + m, 0), 255)

    return newImg

def hist(img):  # 11
    ''' retorna uma matriz coluna onde cada posição 
    contém o número de pixels com cada intensidade de cinza. '''
    # pegando o tamanho
    N, M = size(img) 
    # verificando se é rgb
    if nchannels(img) > 1: 
        # declarando vetores com zeros
        r = np.zeros(256)
        g = np.zeros(256)
        b = np.zeros(256)
        for i in np.arange(N):
            for j in np.arange(M):
                #fazendo o calculo do histograma pra rgb
                r[img[i,j, 0]] += 1 
                g[img[i,j, 1]] += 1
                b[img[i,j, 2]] += 1
        return np.array([r, g, b])
    else:
        c = np.zeros(256)
        for i in np.arange(N):
            for j in np.arange(M):
                c[img[i,j]] += 1
        return np.array(c)

def showhist(img, bin=1):  # 12 e 13
    ''' mostra um gráfico de barras para o histograma da imagem '''
    
    histo = hist(img)
    width = 0.25
    n_bin = 256//bin
    plt.figure(figsize=(20,10))
    plt.xlabel('pixels')
    plt.ylabel('count')
    plt.title('Histograma')

    if nchannels(img) > 1:
        # calculando novo histograma para o novo bin
        newHisto = np.zeros((3, n_bin+1))
        #RGB
        for i in np.arange(0, 256, bin):
            newHisto[0, i//bin] = sum(histo[0, i:i+bin])
            newHisto[1, i//bin] = sum(histo[1, i:i+bin])
            newHisto[2, i//bin] = sum(histo[2, i:i+bin])

        # plotando o histograma RGB
        b1 = np.arange(n_bin+1)
        b2 = [x + width for x in b1]
        b3 = [x + width for x in b2] 
        plt.bar(b1, newHisto[0], width, alpha= 0.7, color='red')
        plt.bar(b2, newHisto[1], width, alpha= 0.7, color='green')
        plt.bar(b3, newHisto[2], width, alpha= 0.7, color='blue')
        plt.show()
    
    else:
        # calculando novo histograma para o novo bin
        newHisto = np.zeros(n_bin+1)
        for i in np.arange(0, 256, bin):
            newHisto[i//bin] = sum(histo[i:i+bin])

        # plotando o histograma escala de cinza
        b1 = np.arange(n_bin+1)
        plt.bar(b1, newHisto, width, alpha= 0.7, color='gray')
        plt.show()

def histeq(img):  # 14
    ''' calcula a equalização do histograma da imagem 
    de entrada e retorna a imagem resultante. '''
    # calculando histograma
    histo = hist(img)
    # calculando a fdp
    fdp = np.cumsum(histo) / np.sum(histo)
    # equalizacao
    e = (fdp[img] * 255).astype('uint8')   
    return e

def convolve(img, mask): # 15
    ''' Retorna a convolução da imagem de entrada pela máscara. '''
    # fazendo a copia da matrix
    newImg = np.copy(img)
    # tamanho da matrix
    N, M = size(img)
    # convertendo as posicoes da matrix pra uma lista
    Z = _matrix2List(mask)
    if nchannels(img) > 1:
        for i in np.arange(N):
            for j in np.arange(M):
                #fazendo o convolve pra cada pixel
                newImg[i, j, 0] = sum( img[ _clamp(i + z [0], N), _clamp(j + z[1], M), 0 ] * mask[z] for z in Z ) # r
                newImg[i, j, 1] = sum( img[ _clamp(i + z [0], N), _clamp(j + z[1], M), 1 ] * mask[z] for z in Z ) # g
                newImg[i, j, 2] = sum( img[ _clamp(i + z [0], N), _clamp(j + z[1], M), 2 ] * mask[z] for z in Z ) # b 
        return newImg
    else: 
        # para imagem em escala de cinza
        #newImg = [ [ sum( img[ _clamp(i + z [0], N), _clamp(j + z[1], M)] * mask[z] for z in Z ) 
        #     for j in range(M) ] for i in range(N) ]
        print(Z)
        for i in np.arange(N):
          for j in np.arange(M):
            newImg[i, j] = sum( img[ _clamp(i + z [0], N), _clamp(j + z[1], M)] * mask[z] for z in Z )
        return newImg
   


def maskBLur(): # 16
    '''retorna a máscara de blur'''
    return np.array([[1,2,1],[2,4,2],[1,2,1]]) * 1/16

def blur(img): # 17
    '''convolve a imagem de entrada pela máscara retornada pela função maskBlur'''
    return convolve(img, maskBLur())

def seSquare3():
    ''' retorna o elemento estruturante binário. '''
    return np.array([[1,1,1],[1,1,1],[1,1,1]])

def seCross3():
    ''' retorna o elemento estruturante binário. '''
    return np.array([[0,1,0],[1,1,1],[0,1,0]])

def _matrix2List(matrix):
    ''' funcao auxiliar que transforma matrix em lista
        cada posicao da lista é uma tupla da posicão x y da matrix. '''
    # pegando o tamanho da matrix
    N, M = size(matrix)
    # gerando a lista
    S = [ (i-1, j-1) for i in range(N) for j in range(M) if matrix[i,j] != 0 ]
    return S

def _clamp(x, l):
    ''' funcao auxiliar que usa o valor do pixel mais 
        próximo pertencente à borda. '''
     # clamp ensinado em sala de aula
    return min(max(x, 0), l-1)

def _minOrMax(img, N, M, Z, i, j, func):
    ''' funcao auxiliar da erode e dilate 
        pega o minimo ou maximo utilizando a mascara 
        e caso seja necessario faz o clamp. '''
    #RGB
    if nchannels(img) > 1: 
        return [ func( img[ _clamp(i + z [0], N), _clamp(j + z[1], M), 0] for z in Z ), 
                 func( img[ _clamp(i + z [0], N), _clamp(j + z[1], M), 1] for z in Z ),
                 func( img[ _clamp(i + z [0], N), _clamp(j + z[1], M), 2] for z in Z ) ]
    #GRAY
    else:    
        return func( img[ _clamp(i + z [0], N), _clamp(j + z[1], M)] for z in Z )

def erode(img, S):
    ''' Retorna uma imagem onde cada pixel (i, j) da saída é igual ao menor valor presente 
        no conjunto de pixels definido pelo elemento estruturante centrado no pixel (i, j) da entrada.  '''
    # pegando tamanho
    N, M = size(img)
    # transformando a mascara em lista
    Z = _matrix2List(S)
    # usando Compreensão de lista
    newImg = np.array([ [ _minOrMax(img, N, M, Z, i, j, min) for j in range(M) ] for i in range(N) ])

    return newImg.astype('uint8')
    
def dilate(img, S):
    ''' Retorna uma imagem onde cada pixel (i, j) da saída é igual ao maior valor presente 
        no conjunto de pixels definido pelo elemento estruturante centrado no pixel (i, j) da entrada. '''
    # pegando tamanho
    N, M = size(img)
    # transformando a mascara em lista
    Z = _matrix2List(S)
    # usando Compreensão de lista
    newImg = np.array([ [ _minOrMax(img, N, M, Z, i, j, max) for j in range(M) ] for i in range(N) ])
    return newImg.astype('uint8')

######################### TESTES
listdir = os.listdir('.')
print(listdir)
grey = listdir[0]
small = listdir[1]
color = listdir[2]

k = imread('felix.jpg').astype('uint8')
x = rgb2gray(k).astype('uint8')

