# Objetivo:
# - Teorema de la convolución


import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fft as fft

MASK_SIZE = 31
MASK_SIZE_2 = 5
STD_DEV = 5

imagen = ski.io.imread("Prácticas/practica5/images/boat.511.tiff")
imagen = ski.util.img_as_float(imagen)

### MÁSCARAS GAUSSIANAS ###

# Hacemos una máscara de convolución de un filtro gaussiano
vector = scipy.signal.windows.gaussian(MASK_SIZE, STD_DEV)
vector /= np.sum(vector)  # Máscara normalizada
vectorH = vector.reshape(1, MASK_SIZE)
vectorV = vector.reshape(MASK_SIZE, 1)

mascara = vectorV @ vectorH
res_convol = scipy.ndimage.convolve(imagen, mascara, mode="wrap")
# Ampliamos la máscara con ceros para que tenga el mismo tamaño que la imagen
mascara_centrada = np.zeros(imagen.shape)
fila_i = imagen.shape[0] // 2 - MASK_SIZE // 2
col_i = imagen.shape[1] // 2 - MASK_SIZE // 2
mascara_centrada[fila_i:fila_i + MASK_SIZE, col_i:col_i + MASK_SIZE] = mascara

# Gaussiana 5x5 , std=5
vector = scipy.signal.windows.gaussian(MASK_SIZE_2, STD_DEV)
vector /= np.sum(vector)  # Máscara normalizada
vectorH = vector.reshape(1, MASK_SIZE_2)
vectorV = vector.reshape(MASK_SIZE_2, 1)

mascara_2 = vectorV @ vectorH
res_convol = scipy.ndimage.convolve(imagen, mascara_2, mode="wrap")
# Ampliamos la máscara con ceros para que tenga el mismo tamaño que la imagen
mascara_2_centrada = np.zeros(imagen.shape)
fila_i = imagen.shape[0] // 2 - MASK_SIZE_2 // 2
col_i = imagen.shape[1] // 2 - MASK_SIZE_2 // 2
mascara_2_centrada[fila_i:fila_i + MASK_SIZE_2, col_i:col_i + MASK_SIZE_2] = mascara_2

### SACAMOS LAS FILAS ###
fila_255_mascara = mascara_centrada[255, :]
fila_255_mascara_2 = mascara_2_centrada[255, :]
segmento_mascara = fila_255_mascara[240:270]
segmento_mascara_2 = fila_255_mascara_2[240:270]

# Graficar los segmentos
plt.plot(segmento_mascara, color='red', label='Mascara 31x31 (Ejercicio 1)')
plt.plot(segmento_mascara_2, color='blue', label='Mascara 5x5 (Ejercicio 2)')

# Añadir etiquetas y leyenda
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Comparación de las filas centrales de las máscaras gaussianas')
plt.legend()

# Mostrar la gráfica
plt.show()

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(imagen, cmap=plt.cm.gray)
axs[0].set_title("Imagen original")
axs[1].imshow(mascara_centrada, cmap=plt.cm.gray)
axs[1].set_title("Gaussiana 31x31, std=5")
axs[2].imshow(mascara_2_centrada, cmap=plt.cm.gray)
axs[2].set_title("Gaussiana 5x5, std=5")

for a in axs.ravel():
    a.set_axis_off()
plt.show()