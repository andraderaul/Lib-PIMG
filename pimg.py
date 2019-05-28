########################################
#
# Nome: Gabriel Augusto
# Matricula: 201500307148
# E­mail: gabriel.silva@dcomp.ufs.br
#
# Nome: Raul Andrade
# Matricula: 201500307353
# E­mail: raul.andrade@dcomp.ufs.br
#
########################################

# Q.01
import matplotlib.pyplot as plt
import matplotlib.image as mpimp
import numpy as np
import os

# Q.02
def imread(arq):
    return mpimp.imread(arq, 'uint8')
# Q.03
def nchannels(img):
    try:
        return img.shape[2]
    except IndexError:
        return 1
# Q.04
def size(img):
    return [img.shape[0], img.shape[1]]
# Q.05
def rgb2gray(img):  # 5
    try:
        gray_img = np.dot(img, [0.299, 0.587, 0.114]).astype('uint8')
        return gray_img
    except ValueError:
        return img
# Q.06
def imreadgray(arq):
    img = mpimp.imread(arq, 'uint8')
    if nchannels(img) > 1:
        return rgb2gray(img).astype('uint8')
    return img.astype('uint8')
# Q.07    
def imshow(img):  # 7
    if nchannels(img) > 1:
        plt.imshow(img, interpolation='nearest', cmap=None)
    else :    
        plt.imshow(img, interpolation='nearest', cmap='gray')

    plt.show()
# Q.08
def thresh(img, alpha):
    t = ((img >= alpha) * 255)
    return t
# Q.09
def negative(img):
    n = (255 - img)
    return n
# Q.10
def contrast(f, r, m):
    g = np.copy(f) #np.ndarray(size(f), dtype= np.uint8)
    channels = nchannels(f)
    for i in range(len(f)):
      for j in range(len(f[i])):
        if channels > 1:  #caso imagem não esteja em escala de cinza
          for k in range(channels):
            g[i,j, k] = max(0, min(r*(f[i,j,k] - m) + m, 255))  #clamp
        else:
          g[i,j] = max(0, min(r*(f[i,j] - m) + m, 255))  #clamp

    return g.astype(np.uint8)
# Q.11
def hist(img):  # 11
    N, M = size(img) 
    if nchannels(img) > 1: 
        r = np.zeros(256)
        g = np.zeros(256)
        b = np.zeros(256)
        for i in np.arange(N):
            for j in np.arange(M):
                r[img[i,j, 0]] += 1 
                g[img[i,j, 1]] += 1
                b[img[i,j, 2]] += 1
        return np.array([r, g, b])
    else:
        c = np.zeros(256)
        for i in np.arange(N):
            for j in np.arange(M):
                c[img[i,j]] += 1
        return np.array([c])
# Q.12 e Q.13
def showhist(histo, bin=1):
    ''' mostra um gráfico de barras para o histograma da imagem '''
    width = 0.25
    n_bin = 256//bin
    plt.figure(figsize=(20,10))
    plt.xlabel('pixels')
    plt.ylabel('count')
    plt.title('Histograma')
    print(len(histo))
    if len(histo) > 1:
        newHisto = np.zeros((3, n_bin+1))
        for i in np.arange(0, 256, bin):
            newHisto[0, i//bin] = sum(histo[0, i:i+bin])
            newHisto[1, i//bin] = sum(histo[1, i:i+bin])
            newHisto[2, i//bin] = sum(histo[2, i:i+bin])
        b1 = np.arange(n_bin+1)
        b2 = [x + width for x in b1]
        b3 = [x + width for x in b2] 
        plt.bar(b1, newHisto[0], width, alpha= 0.7, color='red', align='center')
        plt.bar(b2, newHisto[1], width, alpha= 0.7, color='green', align='center')
        plt.bar(b3, newHisto[2], width, alpha= 0.7, color='blue', align='center')
        plt.show()
    
    else:
        newHisto = np.zeros((1, n_bin+1))
        for i in np.arange(0, 256, bin):
            newHisto[0, i//bin] = sum(histo[0, i:i+bin])
        b1 = np.arange(n_bin+1)
        plt.bar(b1, newHisto[0], width, alpha= 0.7, color='gray', align='center')
        plt.show()
# Q.14
def histeq(img):
    histo = hist(img)
    fdp = np.cumsum(histo) / np.sum(histo)
    e = (fdp[img] * 255).astype('uint8')   
    return e
# FUNC AUXILIAR    
def clamp(x, y, img):
    dim = size(img)
    xl = max(0, min(x, dim[0]-1))
    yl = max(0, min(y, dim[1]-1))
    return xl,yl

# Q.15
def convolve(f, mask):
    g = np.copy(f) * 0
    a = (len(mask) - 1)//2
    b = (len(mask[0]) - 1)//2

    for i in range(len(f)):
        for j in range(len(f[i])):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    x,y = clamp(i+s, j+t, f)
                    g[i,j] = g[i,j] + (mask[a+s,b+t] * f[x,y])

    return g.astype(np.uint8)
# Q.16
def maskBlur():
  	return np.dot([[1,2,1], [2,4,2], [1,2,1]], 1/16)
# Q.17
def blur(img):
  	blur_img = convolve(img, maskBlur())
  	return blur_img
# Q.18
def seSquare3():
    return np.ones((3,3), np.uint8)
# Q.19
def seCross3():
    return np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8)
# Q.20
def erode( f, elem):
    size1 = size(f)
    size1.append(nchannels(f))
    g = np.ones(size1, dtype= np.uint8) * 255
    a = (len(elem) - 1)//2
    b = (len(elem[0]) - 1)//2
    for i in range(len(f)):
      for j in range(len(f[i])):
        for s in range(-a, (a+1) if a > 0 else 1):
          for t in range(-b, (b+1) if b > 0 else 1):
            if elem[a+s, b+t]:
              x,y = clamp(i+s, j+t, f)
              if nchannels(f) == 1:
                g[i,j] = min(f[x,y], g[i,j])
              else:
                g[i,j] = np.fmin(f[x,y], g[i,j])               
    return g
# Q.21
def dilate(f, elem):
    size1 = size(f)
    size1.append(nchannels(f))
    g = np.zeros(size1, dtype= np.uint8)
    a = (len(elem) - 1)//2
    b = (len(elem[0]) - 1)//2

    for i in range(len(f)):
      for j in range(len(f[i])):
        for s in range(-a, (a+1) if a > 0 else 1):
          for t in range(-b, (b+1) if b > 0 else 1):
            if elem[a+s, b+t]:
              x,y = clamp(i+s, j+t, f)
              if nchannels(f) == 1:
                g[i,j] = max(f[x,y], g[i,j])
              else:
                g[i,j] = np.fmax(f[x,y], g[i,j])               
    return g
