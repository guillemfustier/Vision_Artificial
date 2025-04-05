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

def hacer_filtro_pasa_bajo(imagen, radio):
    FTimagen = fft.fft2(imagen)
    FTimagen_centrada = fft.fftshift(FTimagen)
    # Filtro paso bajo
    mascara_centrada_bajo = crear_circulo_centrado(imagen.shape, radio)
    FT_imagen_filtradaBajo = FTimagen_centrada * mascara_centrada_bajo
    imagen_filtrada_bajo = fft.ifft2(fft.ifftshift(FT_imagen_filtradaBajo))
    imagen_filtrada_bajo_real = np.real(imagen_filtrada_bajo)
    imagen_filtrada_bajo_imag = np.imag(imagen_filtrada_bajo)
    if not np.allclose(imagen_filtrada_bajo_imag, np.zeros(imagen.shape)):
        print("Warning. Algo no está yendo bien!!")
    return [mascara_centrada_bajo, FT_imagen_filtradaBajo, imagen_filtrada_bajo_real]
    

def hacer_filtro_pasa_alto(imagen, radio):
    FTimagen = fft.fft2(imagen)
    FTimagen_centrada = fft.fftshift(FTimagen)
    # Filtro paso alto
    mascara_centrada_alto = 1 - crear_circulo_centrado(imagen.shape, radio)
    FT_imagen_filtradaAlto = FTimagen_centrada * mascara_centrada_alto
    imagen_filtrada_alto = fft.ifft2(fft.ifftshift(FT_imagen_filtradaAlto))
    imagen_filtrada_alto_real = np.real(imagen_filtrada_alto)
    imagen_filtrada_alto_imag = np.imag(imagen_filtrada_alto)
    if not np.allclose(imagen_filtrada_alto_imag, np.zeros(imagen.shape)):
        print("Warning. Algo no está yendo bien!!")
    return [mascara_centrada_alto, FT_imagen_filtradaAlto, imagen_filtrada_alto_real]

FREQ_CORTE_EN_PIXELES = 20

imagen_einstein = ski.io.imread("Prácticas/practica5/images/Einstein.png")
imagen_einstein = ski.util.img_as_float(imagen_einstein)

imagen_marilyn = ski.io.imread("Prácticas/practica5/images/Marilyn.png")
imagen_marilyn = ski.util.img_as_float(imagen_marilyn)

# Einstein
einstein_pasa_bajo = hacer_filtro_pasa_bajo(imagen_einstein, FREQ_CORTE_EN_PIXELES)
einstein_mascara_centrada_bajo = einstein_pasa_bajo[0]
einstein_FT_imagen_filtradaBajo = einstein_pasa_bajo[1]
einstein_imagen_filtrada_bajo_real = einstein_pasa_bajo[2]

einstein_pasa_alto = hacer_filtro_pasa_alto(imagen_einstein, FREQ_CORTE_EN_PIXELES)
einstein_mascara_centrada_alto = einstein_pasa_alto[0]
einstein_FT_imagen_filtradaAlto = einstein_pasa_alto[1]
einstein_imagen_filtrada_alto_real = einstein_pasa_alto[2]

# Marilyn
marilyn_pasa_bajo = hacer_filtro_pasa_bajo(imagen_marilyn, FREQ_CORTE_EN_PIXELES)
marilyn_mascara_centrada_bajo = marilyn_pasa_bajo[0]
marilyn_FT_imagen_filtradaBajo = marilyn_pasa_bajo[1]
marilyn_imagen_filtrada_bajo_real = marilyn_pasa_bajo[2]

marilyn_pasa_alto = hacer_filtro_pasa_alto(imagen_marilyn, FREQ_CORTE_EN_PIXELES)
marilyn_mascara_centrada_alto = marilyn_pasa_alto[0]
marilyn_FT_imagen_filtradaAlto = marilyn_pasa_alto[1]
marilyn_imagen_filtrada_alto_real = marilyn_pasa_alto[2]

# Visulización de resultados
fig, axs = plt.subplots(4, 4, layout="constrained")
axs[0, 0].imshow(imagen_einstein, cmap=plt.cm.gray)
axs[0, 1].imshow(einstein_mascara_centrada_bajo, cmap=plt.cm.gray)
axs[0, 2].imshow(np.log(np.absolute(einstein_FT_imagen_filtradaBajo) + 1), cmap=plt.cm.gray)
axs[0, 3].imshow(einstein_imagen_filtrada_bajo_real, cmap=plt.cm.gray)

axs[1, 0].imshow(imagen_marilyn, cmap=plt.cm.gray)
axs[1, 1].imshow(marilyn_mascara_centrada_bajo, cmap=plt.cm.gray)
axs[1, 2].imshow(np.log(np.absolute(marilyn_FT_imagen_filtradaBajo) + 1), cmap=plt.cm.gray)
axs[1, 3].imshow(marilyn_imagen_filtrada_bajo_real, cmap=plt.cm.gray)

axs[2, 0].imshow(imagen_einstein, cmap=plt.cm.gray)
axs[2, 1].imshow(einstein_mascara_centrada_alto, cmap=plt.cm.gray)
axs[2, 2].imshow(np.log(np.absolute(einstein_FT_imagen_filtradaAlto) + 1), cmap=plt.cm.gray)
axs[2, 3].imshow(einstein_imagen_filtrada_alto_real, cmap=plt.cm.gray)

axs[3, 0].imshow(imagen_marilyn, cmap=plt.cm.gray)
axs[3, 1].imshow(marilyn_mascara_centrada_alto, cmap=plt.cm.gray)
axs[3, 2].imshow(np.log(np.absolute(marilyn_FT_imagen_filtradaAlto) + 3), cmap=plt.cm.gray)
axs[3, 3].imshow(marilyn_imagen_filtrada_alto_real, cmap=plt.cm.gray)

for a in axs.ravel():
    a.set_axis_off()
plt.show()

# Suma de las dos imágenes filtradas
combinacion = einstein_imagen_filtrada_bajo_real + marilyn_imagen_filtrada_alto_real

fig = plt.figure()
plt.imshow(combinacion, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# Suma de las frecuencias y luego inversa de la suma
combinacion2 = einstein_FT_imagen_filtradaBajo * einstein_mascara_centrada_bajo + marilyn_FT_imagen_filtradaAlto * marilyn_mascara_centrada_alto
imagen_combinacion2 = fft.ifft2(fft.ifftshift(combinacion2))
imagen_combinacion2_real = np.real(imagen_combinacion2)
imagen_combinacion2_imag = np.imag(imagen_combinacion2)

fig = plt.figure()
plt.imshow(imagen_combinacion2_real, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

if np.allclose(imagen_combinacion2_real, combinacion):
    print("Es lo mismo sumar las frecuencias y luego hacer la inversa que hacer la suma de las imágenes transformadas")
else:
    print("No es lo mismo sumar las frecuencias y luego hacer la inversa que las imágenes transformadas")