# Objetivos:
# - Invertir, aclarar y oscurecer imágenes
# - Calcular y visualizar histogramas

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/girl.pgm")
h_orig, c_orig = ski.exposure.histogram(img_original)

img_invertida = 255 - img_original
h_inv, c_inv = ski.exposure.histogram(img_invertida)

img_real = ski.util.img_as_float(img_original) * 255

img_oscura = img_real ** 2 / 255
img_oscura = ski.util.img_as_ubyte(img_oscura / 255)
h_osc, c_osc = ski.exposure.histogram(img_oscura)

img_clara = np.sqrt(255 * img_real)
img_clara = ski.util.img_as_ubyte(img_clara / 255)
h_clara, c_clara = ski.exposure.histogram(img_clara)

fig, axs = plt.subplots(2, 4, layout="constrained")

axs[0, 0].imshow(img_original, cmap='gray')
axs[0, 0].set_title("Original")
axs[0, 1].bar(c_orig, h_orig, 1.1)

axs[0, 2].imshow(img_invertida, cmap='gray')
axs[0, 2].set_title("Invertida")
axs[0, 3].bar(c_inv, h_inv, 1.1)

axs[1, 0].imshow(img_oscura, cmap='gray')
axs[1, 0].set_title("Oscurecida")
axs[1, 1].bar(c_osc, h_osc, 1.1)

axs[1, 2].imshow(img_clara, cmap='gray')
axs[1, 2].set_title("Aclarada")
axs[1, 3].bar(c_clara, h_clara, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
