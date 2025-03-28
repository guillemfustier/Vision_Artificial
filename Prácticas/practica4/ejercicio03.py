# Objetivo:
# -Convolve (en 1D y en 2D)

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

imagen = ski.io.imread("Prácticas/practica4/images/boat.512.tiff")
imagen = ski.util.img_as_float(imagen)

def tiempo_filtro_media_2D(imagen, MASK_SIZE):
    ### FILTRO MEDIA 2D
    filtro_media_1D = np.ones(MASK_SIZE)
    filtro_media_1D /= np.sum(filtro_media_1D)

    filtroH = filtro_media_1D.reshape(1, MASK_SIZE)
    filtroV = filtro_media_1D.reshape(MASK_SIZE, 1)

    matriz = filtroV @ filtroH
    
    inicio2D = time.time()
    res_convol2D = scipy.ndimage.convolve(imagen, matriz)
    fin2D = time.time()

    return fin2D - inicio2D

def tiempo_filtro_media_1D(imagen, MASK_SIZE):
    ### FILTRO MEDIA 1D
    filtro_media_1D = np.ones(MASK_SIZE)
    filtro_media_1D /= np.sum(filtro_media_1D)

    filtroH = filtro_media_1D.reshape(1, MASK_SIZE)
    filtroV = filtro_media_1D.reshape(MASK_SIZE, 1)

    resH = scipy.ndimage.convolve(imagen, filtroH)
    resMedia = scipy.ndimage.convolve(resH, filtroV)

    inicio1D = time.time()
    resH = scipy.ndimage.convolve(imagen, filtroH)
    res1D = scipy.ndimage.convolve(resH, filtroV)
    fin1D = time.time()

    return fin1D - inicio1D

mask_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
tiempos2D = []
tiempos1D = []
for mask_size in mask_sizes:
    tiempo2D = tiempo_filtro_media_2D(imagen, mask_size)
    tiempo1D = tiempo_filtro_media_1D(imagen, mask_size)
    tiempos2D.append(tiempo2D)
    tiempos1D.append(tiempo1D)

promedio2D = np.mean(tiempos2D)
promedio1D = np.mean(tiempos1D)
print(f"Promedio de tiempo 2D: {promedio2D:0.9f}")
print(f"Promedio de tiempo 1D: {promedio1D:0.9f}")
print(f"Factor 2D/1D: {promedio2D / promedio1D:0.2f}")

# Ploteamos todas las gráficas
fig, axs = plt.subplots(1, 2, layout="constrained")
axs[0].imshow(imagen, cmap='gray')
axs[0].set_title("Imagen Original")

axs[1].plot(mask_sizes, tiempos2D, label="2D")
axs[1].plot(mask_sizes, tiempos1D, label="1D")
axs[1].legend()

for a in axs:
    a.set_axis_off()
plt.show()
