# Objetivos:
# - Calcular histogramas de imágenes en color

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_color = ski.io.imread("images/lena256.ppm")
h_color, centros_color = ski.exposure.histogram(img_color, channel_axis=-1)

plano_rojo = img_color[:, :, 0]
histo_rojo = h_color[0, :]

plano_verde = img_color[:, :, 1]
histo_verde = h_color[1, :]

plano_azul = img_color[:, :, 2]
histo_azul = h_color[2, :]

fig, axs = plt.subplots(4, 2, layout="constrained")

axs[0, 0].imshow(img_color)
axs[0, 0].set_title("Color")

axs[1, 0].imshow(plano_rojo, cmap='gray')
axs[1, 0].set_title("Rojo")
axs[1, 1].bar(centros_color, histo_rojo, 1.1)

axs[2, 0].imshow(plano_verde, cmap='gray')
axs[2, 0].set_title("Verde")
axs[2, 1].bar(centros_color, histo_verde, 1.1)

axs[3, 0].imshow(plano_azul, cmap='gray')
axs[3, 0].set_title("Azul")
axs[3, 1].bar(centros_color, histo_azul, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])
axs[0, 1].set_axis_off()

plt.show()
