# Objetivo:
# - Filtros paso bajo y paso alto


import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fft as fft


def crear_circulo_centrado(shape, radio):  # Asumo dimensiones impares
    filas = np.matrix(list(range(shape[0]))).T - shape[0] // 2
    cols = np.matrix(list(range(shape[1]))) - shape[1] // 2
    indices_dentro = np.power(filas, 2) + np.power(cols, 2) < radio ** 2
    circulo = np.zeros(shape)
    circulo[indices_dentro] = 1
    return circulo


FREQ_CORTE_EN_PIXELES = 40

imagen = ski.io.imread("images/boat.511.tiff")
imagen = ski.util.img_as_float(imagen)
FTimagen = fft.fft2(imagen)
FTimagen_centrada = fft.fftshift(FTimagen)

# Filtro paso bajo
mascara_centrada_bajo = crear_circulo_centrado(imagen.shape, FREQ_CORTE_EN_PIXELES)
FT_imagen_filtradaBajo = FTimagen_centrada * mascara_centrada_bajo
imagen_filtrada_bajo = fft.ifft2(fft.ifftshift(FT_imagen_filtradaBajo))
imagen_filtrada_bajo_real = np.real(imagen_filtrada_bajo)
imagen_filtrada_bajo_imag = np.imag(imagen_filtrada_bajo)
if not np.allclose(imagen_filtrada_bajo_imag, np.zeros(imagen.shape)):
    print("Warning. Algo no está yendo bien!!")

# Filtro paso alto
mascara_centrada_alto = 1 - mascara_centrada_bajo
FT_imagen_filtradaAlto = FTimagen_centrada * mascara_centrada_alto
imagen_filtrada_alto = fft.ifft2(fft.ifftshift(FT_imagen_filtradaAlto))
imagen_filtrada_alto_real = np.real(imagen_filtrada_alto)
imagen_filtrada_alto_imag = np.imag(imagen_filtrada_alto)
if not np.allclose(imagen_filtrada_alto_imag, np.zeros(imagen.shape)):
    print("Warning. Algo no está yendo bien!!")

# Visulización de resultados
fig, axs = plt.subplots(2, 4, layout="constrained")
axs[0, 0].imshow(imagen, cmap=plt.cm.gray)
axs[0, 1].imshow(mascara_centrada_bajo, cmap=plt.cm.gray)
axs[0, 2].imshow(np.log(np.absolute(FT_imagen_filtradaBajo) + 1), cmap=plt.cm.gray)
axs[0, 3].imshow(imagen_filtrada_bajo_real, cmap=plt.cm.gray)

axs[1, 0].imshow(np.log(np.absolute(FTimagen_centrada) + 1), cmap=plt.cm.gray)
axs[1, 1].imshow(mascara_centrada_alto, cmap=plt.cm.gray)
axs[1, 2].imshow(np.log(np.absolute(FT_imagen_filtradaAlto) + 1), cmap=plt.cm.gray)
axs[1, 3].imshow(imagen_filtrada_alto_real, cmap=plt.cm.gray)  # Prueba a mostrar el valor absoluto

for a in axs.ravel():
    a.set_axis_off()
plt.show()
