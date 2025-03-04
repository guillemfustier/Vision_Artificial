import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

# Cargar y procesar la imagen 'medica.pgm'
img_original_medica = ski.io.imread("images/medica.pgm")
h_orig_medica, c_orig_medica = ski.exposure.histogram(img_original_medica)

img_real_medica = ski.util.img_as_float(img_original_medica) * 255

# ACLARADA
img_clara_medica = np.sqrt(255 * img_real_medica)
img_clara_medica = ski.util.img_as_ubyte(img_clara_medica / 255)
h_clara_medica, c_clara_medica = ski.exposure.histogram(img_clara_medica)

# ECUALIZADA
img_eq_medica = ski.exposure.equalize_hist(img_original_medica)
img_eq_medica = ski.util.img_as_ubyte(img_eq_medica)
h_eq_medica, c_eq_medica = ski.exposure.histogram(img_eq_medica)

# CLAHE
img_eq_CLAHE_medica = ski.exposure.equalize_adapthist(img_original_medica)
img_eq_CLAHE_medica = ski.util.img_as_ubyte(img_eq_CLAHE_medica)
h_eq_CLAHE_medica, c_eq_CLAHE_medica = ski.exposure.histogram(img_eq_CLAHE_medica)

# Cargar y procesar la imagen 'fachada_iglesia.pgm'
img_original_fachada = ski.io.imread("images/fachada_iglesia.pgm")
h_orig_fachada, c_orig_fachada = ski.exposure.histogram(img_original_fachada)

img_real_fachada = ski.util.img_as_float(img_original_fachada) * 255

# ACLARADA
img_clara_fachada = np.sqrt(255 * img_real_fachada)
img_clara_fachada = ski.util.img_as_ubyte(img_clara_fachada / 255)
h_clara_fachada, c_clara_fachada = ski.exposure.histogram(img_clara_fachada)

# ECUALIZADA
img_eq_fachada = ski.exposure.equalize_hist(img_original_fachada)
img_eq_fachada = ski.util.img_as_ubyte(img_eq_fachada)
h_eq_fachada, c_eq_fachada = ski.exposure.histogram(img_eq_fachada)

# CLAHE
img_eq_CLAHE_fachada = ski.exposure.equalize_adapthist(img_original_fachada)
img_eq_CLAHE_fachada = ski.util.img_as_ubyte(img_eq_CLAHE_fachada)
h_eq_CLAHE_fachada, c_eq_CLAHE_fachada = ski.exposure.histogram(img_eq_CLAHE_fachada)

# Visualización
fig, axs = plt.subplots(2, 8, layout="constrained")

# Mostrar para la imagen 'medica.pgm'
axs[0, 0].imshow(img_original_medica, cmap='gray')
axs[0, 0].set_title("Original (Medica)")
axs[0, 1].bar(c_orig_medica, h_orig_medica, 1.1)

axs[0, 2].imshow(img_clara_medica, cmap='gray')
axs[0, 2].set_title("Aclarada (Medica)")
axs[0, 3].bar(c_clara_medica, h_clara_medica, 1.1)

axs[0, 4].imshow(img_eq_medica, cmap='gray')
axs[0, 4].set_title("Ecualizada (Medica)")
axs[0, 5].bar(c_eq_medica, h_eq_medica, 1.1)

axs[0, 6].imshow(img_eq_CLAHE_medica, cmap='gray')
axs[0, 6].set_title("CLAHE (Medica)")
axs[0, 7].bar(c_eq_CLAHE_medica, h_eq_CLAHE_medica, 1.1)

# Mostrar para la imagen 'fachada_iglesia.pgm'
axs[1, 0].imshow(img_original_fachada, cmap='gray')
axs[1, 0].set_title("Original (Fachada)")
axs[1, 1].bar(c_orig_fachada, h_orig_fachada, 1.1)

axs[1, 2].imshow(img_clara_fachada, cmap='gray')
axs[1, 2].set_title("Aclarada (Fachada)")
axs[1, 3].bar(c_clara_fachada, h_clara_fachada, 1.1)

axs[1, 4].imshow(img_eq_fachada, cmap='gray')
axs[1, 4].set_title("Ecualizada (Fachada)")
axs[1, 5].bar(c_eq_fachada, h_eq_fachada, 1.1)

axs[1, 6].imshow(img_eq_CLAHE_fachada, cmap='gray')
axs[1, 6].set_title("CLAHE (Fachada)")
axs[1, 7].bar(c_eq_CLAHE_fachada, h_eq_CLAHE_fachada, 1.1)

# Desactivar ejes innecesarios
axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
