# Objetivo:
# -Convolve (en 1D y en 2D)

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fft as fft
import time

imagen = ski.io.imread("Prácticas/practica5/images/boat.512.tiff")
imagen = ski.util.img_as_float(imagen)

def tiempo_filtro_media_convol(imagen, MASK_SIZE):
    tiempos = []
    for i in range(10):
        # Convolución en el espacio
        mascara = np.ones((MASK_SIZE, MASK_SIZE))  # Máscara de NxN toda con 1
        mascara /= np.sum(mascara)  # Máscara normalizada
        res_convol = scipy.ndimage.convolve(imagen, mascara, mode="wrap")

        # Ampliamos la máscara con ceros para que tenga el mismo tamaño que la imagen
        mascara_centrada = np.zeros(imagen.shape)
        fila_i = imagen.shape[0] // 2 - MASK_SIZE // 2
        col_i = imagen.shape[1] // 2 - MASK_SIZE // 2
        mascara_centrada[fila_i:fila_i + MASK_SIZE, col_i:col_i + MASK_SIZE] = mascara

        inicioConvol = time.time()
        # Pasamos imagen y máscara a las frecuencias
        FTimagen = fft.fft2(imagen)
        mascara_en_origen = fft.ifftshift(mascara_centrada)
        FTmascara = fft.fft2(mascara_en_origen)

        # Convolución en la frecuancia
        FTimagen_filtrada = FTimagen * FTmascara  # Producto punto a punto

        # Recuperamos resultado en el espacio
        res_filtro_FT = fft.ifft2(FTimagen_filtrada)
        res_filtro_real = np.real(res_filtro_FT)
        res_filtro_imag = np.imag(res_filtro_FT)
        if not np.allclose(res_filtro_imag, np.zeros(imagen.shape)):
            print("Warning. Algo no está yendo bien!!!")
        finConvol = time.time()

        tiempos.append(finConvol - inicioConvol)
    return np.mean(tiempos)

def tiempo_filtro_media_1D(imagen, MASK_SIZE):
    tiempos = []
    for i in range(10):
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

        tiempos.append(fin1D - inicio1D)
    return np.mean(tiempos)

mask_sizes = [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103]
tiemposConvol = []
tiempos1D = []
for mask_size in mask_sizes:
    tiempo2D = tiempo_filtro_media_convol(imagen, mask_size)
    tiempo1D = tiempo_filtro_media_1D(imagen, mask_size)
    tiemposConvol.append(tiempo2D)
    tiempos1D.append(tiempo1D)

promedio2D = np.mean(tiemposConvol)
promedio1D = np.mean(tiempos1D)
print(f"Promedio de tiempo 2D: {promedio2D:0.9f}")
print(f"Promedio de tiempo 1D: {promedio1D:0.9f}")
print(f"Factor 2D/1D: {promedio2D / promedio1D:0.2f}")

# Ploteamos todas las gráficas
fig, axs = plt.subplots(1, 2, layout="constrained")
axs[0].imshow(imagen, cmap='gray')
axs[0].set_title("Imagen Original")

axs[1].plot(mask_sizes, tiemposConvol, label="Teorema de la convolución")
axs[1].plot(mask_sizes, tiempos1D, label="1D")
axs[1].legend()

plt.show()
