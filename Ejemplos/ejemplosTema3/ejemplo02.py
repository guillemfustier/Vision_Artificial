# Objetivos:
# - Rotar una imagen (parámetros centro, resize e interpolación) y trasladar

import skimage as ski
import matplotlib.pyplot as plt

img_original = ski.io.imread("images/lena256.pgm")

img_girada = ski.transform.rotate(img_original, 30, center=(0, 0), resize=True, order=3)
transf_despl = ski.transform.EuclideanTransform(translation=(100, 50))
img_final = ski.transform.warp(img_girada, transf_despl.inverse, output_shape=(512, 512))


fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[0].set_title("Original")
axs[1].imshow(img_girada, cmap=plt.cm.gray)
axs[1].set_title("Rotada")
axs[2].imshow(img_final, cmap=plt.cm.gray)
axs[2].set_title("Rotada y desplazada")


axs[0].set_axis_off()
plt.show()
