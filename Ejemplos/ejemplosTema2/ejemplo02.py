# Objetivos:
# - Ecualizar histograma
# - Ecualización adaptativa

import skimage as ski
import matplotlib.pyplot as plt

img_original = ski.io.imread("images/girl.pgm")
h_orig, c_orig = ski.exposure.histogram(img_original)

img_eq = ski.exposure.equalize_hist(img_original)
img_eq = ski.util.img_as_ubyte(img_eq)
h_eq, c_eq = ski.exposure.histogram(img_eq)

img_eq_adapt = ski.exposure.equalize_adapthist(img_original, clip_limit=1.0)
img_eq_adapt = ski.util.img_as_ubyte(img_eq_adapt)
h_eq_adapt, c_eq_adapt = ski.exposure.histogram(img_eq_adapt)

img_eq_CLAHE = ski.exposure.equalize_adapthist(img_original)
img_eq_CLAHE = ski.util.img_as_ubyte(img_eq_CLAHE)
h_eq_CLAHE, c_eq_CLAHE = ski.exposure.histogram(img_eq_CLAHE)

fig, axs = plt.subplots(4, 2, layout="constrained")

axs[0, 0].imshow(img_original, cmap='gray')
axs[0, 0].set_title("Original")
axs[0, 1].bar(c_orig, h_orig, 1.1)

axs[1, 0].imshow(img_eq, cmap='gray')
axs[1, 0].set_title("Ecualizada")
axs[1, 1].bar(c_eq, h_eq, 1.1)

axs[2, 0].imshow(img_eq_adapt, cmap='gray')
axs[2, 0].set_title("AHE")
axs[2, 1].bar(c_eq_adapt, h_eq_adapt, 1.1)

axs[3, 0].imshow(img_eq_CLAHE, cmap='gray')
axs[3, 0].set_title("CLAHE")
axs[3, 1].bar(c_eq_CLAHE, h_eq_CLAHE, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
