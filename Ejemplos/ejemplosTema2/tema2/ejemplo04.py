# Objetivos
# - Calcular planos PCA de una imagen en color

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/lena256.ppm")

img_original = ski.util.img_as_float(img_original)
forma_inicial = img_original.shape

imagen_por_pixeles = np.reshape(img_original, (forma_inicial[0] * forma_inicial[1], forma_inicial[2])).transpose()
# Matriz de tamaño 3 x número de píxeles

cov_matrix = np.cov(imagen_por_pixeles)
U, S, V = np.linalg.svd(cov_matrix)

y = U.transpose() @ imagen_por_pixeles
y = y.transpose().reshape(forma_inicial)

pca = (y - y.min()) / (y.max() - y.min())  # Reajustar al rango [0.0-1.0]

print("Autovalores:", S)
fig, axs = plt.subplots(3, 3, layout="constrained")
axs[0, 1].imshow(img_original)
axs[0, 1].set_title("Original")

axs[1, 0].imshow(img_original[:, :, 0], cmap='gray')
axs[1, 0].set_title("Rojo")
axs[1, 1].imshow(img_original[:, :, 1], cmap='gray')
axs[1, 1].set_title("Verde")
axs[1, 2].imshow(img_original[:, :, 2], cmap='gray')
axs[1, 2].set_title("Azul")

axs[2, 0].imshow(1 - pca[:, :, 0], cmap='gray')
axs[2, 0].set_title("PCA 1")
axs[2, 1].imshow(1 - pca[:, :, 1], cmap='gray')
axs[2, 1].set_title("PCA 2")
axs[2, 2].imshow(1 - pca[:, :, 2], cmap='gray')
axs[2, 2].set_title("PCA 3")

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()
