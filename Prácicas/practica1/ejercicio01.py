"""
Sobre los diferentes canales RGB, para la bandera que he elegido, espero
encontrar:
R -> Franjas horizontales muy blancas en la bandera, las que corresponden al
color rojo. Dentro del escudo también espero encontrar muy blanco la cruz, las
4 franjas de la senyera y las 3 franjas de la parte blaugrana, el balón será
claro también pero no blanco puro.
G -> No espero encontrar nada de un color blanco, como mucho tonos grisáceos
que se corresponden a que otros colores tienen alguna componente verde, pero no
hay nada verde enteramente
B -> El fondo será muy blanco y algunas franjas horizontales de la bandera
también, las que son azules.

Creo que el escudo se verá en todos los canales de color, porque no es ningún
color puro RGB, al ser una combinación siempre se podrá diferenciar.
"""

# Objetivos:
# - lectura de una imagen, atributos ndim, shape, size y métodos max y min.
# - Acceso a los planos RGB de una imagen en color
# - Visualización simultánea de imágenes en color y en niveles de gris (cmap)

import skimage as ski
import matplotlib.pyplot as plt

imagen_en_color = ski.io.imread("images/banderaBarça.jpg")

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
