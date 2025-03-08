# Objetivos:
# - Entender el plano de transparencia

import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

imagenOriginal = ski.io.imread("images/flecha_transparente.png")
print(f"Forma de la imagen: {imagenOriginal.shape}")

planoT = imagenOriginal[:, :, 3]

copia = imagenOriginal.copy()
copia[:, 0:80, 3] = 0
copia[:, 80:160, 3] = 128
copia[:, 160:, 3] = 255

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(imagenOriginal)
axs[1].imshow(planoT, cmap=plt.cm.gray)
axs[2].imshow(copia)

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()
