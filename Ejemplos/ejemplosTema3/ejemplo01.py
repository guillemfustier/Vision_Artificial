# Objetivos
# - Reescalar una imagen
# - Entender los distintos tipos de interpolación

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

FACTOR = 8

img_original = ski.io.imread("images/ojo_azul.png")

img_ojo = ski.transform.rescale(img_original, (1 / FACTOR, 1 / FACTOR, 1), order=5)

img_final_0 = ski.transform.rescale(img_ojo, (FACTOR, FACTOR, 1), order=0)  # Vecino más próximo
img_final_1 = ski.transform.rescale(img_ojo, (FACTOR, FACTOR, 1), order=1)  # Bilineal
img_final_2 = ski.transform.rescale(img_ojo, (FACTOR, FACTOR, 1), order=2)
img_final_3 = ski.transform.rescale(img_ojo, (FACTOR, FACTOR, 1), order=3)  # Bicúbica
img_final_4 = ski.transform.rescale(img_ojo, (FACTOR, FACTOR, 1), order=4)
img_final_5 = ski.transform.rescale(img_ojo, (FACTOR, FACTOR, 1), order=5)

img_ojo_centrado = ski.util.img_as_ubyte(np.zeros(img_original.shape)) + 255
fila_c = img_ojo_centrado.shape[0] // 2
col_c = img_ojo_centrado.shape[1] // 2
filas_peque2 = img_ojo.shape[0] // 2
cols_peque2 = img_ojo.shape[1] // 2
img_ojo_centrado[fila_c - filas_peque2:fila_c + filas_peque2,
                 col_c - cols_peque2:col_c + cols_peque2, :] = ski.util.img_as_ubyte(img_ojo)

fig, axs = plt.subplots(3, 3, layout="constrained")
axs[0, 0].imshow(img_original)
axs[1, 0].imshow(img_ojo_centrado)

axs[0, 1].imshow(img_final_0)
axs[0, 1].set_title("Orden 0")
axs[1, 1].imshow(img_final_2)
axs[1, 1].set_title("Orden 2")
axs[2, 1].imshow(img_final_4)
axs[2, 1].set_title("Orden 4")

axs[0, 2].imshow(img_final_1)
axs[0, 2].set_title("Orden 1")
axs[1, 2].imshow(img_final_3)
axs[1, 2].set_title("Orden 3")
axs[2, 2].imshow(img_final_5)
axs[2, 2].set_title("Orden 5")

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()

