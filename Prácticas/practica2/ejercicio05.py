import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import match_histograms

img_original_girl = ski.io.imread("Prácticas/practica2/images/girl.pgm")
img_original_nieve = ski.io.imread("Prácticas/practica2/images/nieve.pgm")

h_orig_girl, c_orig_girl = ski.exposure.histogram(img_original_girl)
h_orig_nieve, c_orig_nieve = ski.exposure.histogram(img_original_nieve)

img_real_girl = ski.util.img_as_float(img_original_girl) * 255
img_real_nieve = ski.util.img_as_float(img_original_nieve) * 255

matched_girl_to_nieve = match_histograms(img_original_girl, img_original_nieve)
h_matched_girl_to_nieve, c_matched_girl_to_nieve = ski.exposure.histogram(matched_girl_to_nieve)

matched_nieve_to_girl = match_histograms(img_original_nieve, img_original_girl)
h_matched_nieve_to_girl, c_matched_nieve_to_girl = ski.exposure.histogram(matched_nieve_to_girl)

# === SHOW PLOTS ===

fig, axs = plt.subplots(4, 2, layout="constrained")

axs[0, 0].imshow(img_original_girl, cmap='gray')
axs[0, 0].set_title("Girl Original")
axs[0, 1].bar(c_orig_girl, h_orig_girl, 1.1)

axs[1, 0].imshow(img_original_nieve, cmap='gray')
axs[1, 0].set_title("Nieve Original")
axs[1, 1].bar(c_orig_nieve, h_orig_nieve, 1.1)

axs[2, 0].imshow(matched_girl_to_nieve, cmap='gray')
axs[2, 0].set_title("Matched Girl to Nieve")
axs[2, 1].bar(c_matched_girl_to_nieve, h_matched_girl_to_nieve, 1.1)

axs[3, 0].imshow(matched_nieve_to_girl, cmap='gray')
axs[3, 0].set_title("Matched Nieve to Girl")
axs[3, 1].bar(c_matched_nieve_to_girl, h_matched_nieve_to_girl, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
