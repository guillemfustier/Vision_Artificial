# Objetivos:
# - lectura de una imagen, atributos ndim, shape, size y métodos max y min.
# - Acceso a los planos RGB de una imagen en color
# - Visualización simultánea de imágenes en color y en niveles de gris (cmap)

import skimage as ski
import matplotlib.pyplot as plt

imagen_en_color = ski.io.imread("images/banderaItalia.jpg")
"""
Depende de como sea la imagen, nos dará un output de "dimensiones" diferente:
2 DIMENSIONES: Gris  -> matriz[filas, cols]
3 DIMENSIONES: Color -> matriz[filas, cols, bandacolor]  // bandacolor = 3 (RGB) ó 4 (nose)
4 DIMENSIONES: Vídeos o animaciones -> matriz[frame, filas, cols, bandacolor]
"""

print(f'Número de dimensiones: {imagen_en_color.ndim}')         # número de dimensiones
print(f'Tamaño de la matriz (shape): {imagen_en_color.shape}')  # (filas, cols, bandacolor)
print(f'Tamaño de la matriz (size): {imagen_en_color.size}')    # tamaño en bytes que ocupa
print(f'Valor máximo: {imagen_en_color.max()} y valor mínimo: {imagen_en_color.min()}')

# (:) todos los elementos de filas, (:) todos los elementos de cols
plano_rojo = imagen_en_color[:, :, 0]   # (0) bandacolor = R
plano_verde = imagen_en_color[:, :, 1]  # (1) bandacolor = G
plano_azul = imagen_en_color[:, :, 2]   # (2) bandacolor = B

fig, axs = plt.subplots(2, 3, layout="constrained")
axs[0, 1].imshow(imagen_en_color)
axs[1, 0].imshow(plano_rojo, cmap=plt.cm.gray)
axs[1, 1].imshow(plano_verde, cmap=plt.cm.gray)
axs[1, 2].imshow(plano_azul, cmap=plt.cm.gray)

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()
