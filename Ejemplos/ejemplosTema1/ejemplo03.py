# Objetivos
# - Copia de imágenes.
# - Acceso/Copia de un área determinada

import skimage as ski
import matplotlib.pyplot as plt

imagenItalia = ski.io.imread("images/banderaItalia.jpg")
imagenIrlanda = ski.io.imread("images/banderaIrlanda.jpg")

nueva = imagenIrlanda.copy()
nueva[0:imagenItalia.shape[0], 0:imagenItalia.shape[1], :] = imagenItalia

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(imagenItalia)
axs[1].imshow(imagenIrlanda)
axs[2].imshow(nueva)
for ax in axs:
    ax.set_axis_off()
plt.show()
