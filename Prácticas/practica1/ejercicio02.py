# Objetivos
# - Copia de imágenes.
# - Acceso/Copia de un área determinada

import skimage as ski
import matplotlib.pyplot as plt

imagenItalia = ski.io.imread("images/banderaItalia.jpg")
imagenIrlanda = ski.io.imread("images/banderaIrlanda.jpg")

nueva = imagenIrlanda.copy()
nueva_horiz = nueva.shape[0]//2 - imagenItalia.shape[1]//2
nueva_vert = nueva.shape[1]//2 - imagenItalia.shape[0]//2
nueva[nueva_horiz:imagenItalia.shape[1]+nueva_horiz, nueva_vert:imagenItalia.shape[0]+nueva_vert, :] = imagenItalia.transpose(1,0,2)


plt.imshow(nueva)
plt.show()
