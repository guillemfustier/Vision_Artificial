import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

# Función para reconvertir a RGB
def convertir_a_rgb(img_hsv, canal_h):
    img_modificada = np.copy(img_hsv)
    img_modificada[:, :, 2] = canal_h / 255
    img_rgb = hsv2rgb(img_modificada)
    return ski.util.img_as_ubyte(img_rgb)

img_original = ski.io.imread("images/calle.png")
img_hsv = rgb2hsv(img_original)
img_rgb = hsv2rgb(img_hsv)

h_orig, c_orig = ski.exposure.histogram(img_rgb)

img_real = ski.util.img_as_float(img_original) * 255

h_channel = img_hsv[:, :, 0] * 255
s_channel = img_hsv[:, :, 1] * 255
v_channel = img_hsv[:, :, 2] * 255  # Normalizado

# === ACLARAR ===
v_aclarar = np.sqrt(255 * v_channel)
img_clara = convertir_a_rgb(img_hsv, v_aclarar)

h_clara_1, c_clara_1 = ski.exposure.histogram(img_clara)

# === ACLARAR 2 ===
v_aclarar_2 = np.sqrt(255 * v_aclarar)
img_clara_2 = convertir_a_rgb(img_hsv, v_aclarar_2)

h_clara_2, c_clara_2 = ski.exposure.histogram(img_clara_2)

# === ECUALIZADA ===
v_eq = ski.exposure.equalize_hist(v_channel / 255) * 255
img_eq = convertir_a_rgb(img_hsv, v_eq)

h_eq, c_eq = ski.exposure.histogram(img_eq)

# === CLAHE ===
v_clahe = ski.exposure.equalize_adapthist(v_channel / 255, clip_limit=0.1) * 255
img_eq_CLAHE = convertir_a_rgb(img_hsv, v_clahe)

h_eq_CLAHE, c_eq_CLAHE = ski.exposure.histogram(img_eq_CLAHE)

fig, axs = plt.subplots(5, 2, layout="constrained")

axs[0, 0].imshow(img_rgb, cmap='gray')
axs[0, 0].set_title("Original")
axs[0, 1].bar(c_orig, h_orig, 1.1)

axs[1, 0].imshow(img_clara, cmap='gray')
axs[1, 0].set_title("Aclarada 1")
axs[1, 1].bar(c_clara_1, h_clara_1, 1.1)

axs[2, 0].imshow(img_clara_2, cmap='gray')
axs[2, 0].set_title("Aclarada 2")
axs[2, 1].bar(c_clara_2, h_clara_2, 1.1)

axs[3, 0].imshow(img_eq, cmap='gray')
axs[3, 0].set_title("Ecualizada")
axs[3, 1].bar(c_eq, h_eq, 1.1)

axs[4, 0].imshow(img_eq_CLAHE, cmap='gray')
axs[4, 0].set_title("CLAHE")
axs[4, 1].bar(c_eq_CLAHE, h_eq_CLAHE, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
