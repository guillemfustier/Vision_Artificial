import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

# Función para aplicar ACLARAR a un canal individual
def aclarar_channel(channel):
    channel_float = ski.util.img_as_float(channel) * 255
    channel_aclarada = np.sqrt(255 * channel_float)
    return ski.util.img_as_ubyte(channel_aclarada / 255)

# Función para aplicar ECUAL a un canal individual
def ecualizar_channel(channel):
    eq = ski.exposure.equalize_hist(channel)
    return ski.util.img_as_ubyte(eq)

# Función para aplicar CLAHE a un canal individual
def clahe_channel(channel):
    eq_clahe = ski.exposure.equalize_adapthist(channel)
    return ski.util.img_as_ubyte(eq_clahe)

img_original = ski.io.imread("images/calle.png")
img_hsv = rgb2hsv(img_original)
img_rgb = hsv2rgb(img_hsv)

h_orig, c_orig = ski.exposure.histogram(img_rgb)

img_real = ski.util.img_as_float(img_original) * 255

# === ACLARAR ===
h_aclarar = img_hsv[:, :, 0]
s_aclarar = img_hsv[:, :, 1]
v_aclarar = aclarar_channel(img_hsv[:, :, 2])

img_clara = np.stack([h_aclarar, s_aclarar, v_aclarar], axis=-1)
img_clara = hsv2rgb(img_clara)
h_clara_1, c_clara_1 = ski.exposure.histogram(img_clara)

# === ECUALIZADA ===
h_eq = img_hsv[:, :, 0]
s_eq = img_hsv[:, :, 1]
v_eq = ecualizar_channel(img_hsv[:, :, 2])

img_eq = np.stack([h_eq, s_eq, v_eq], axis=-1)
img_eq = hsv2rgb(img_eq)
h_eq, c_eq = ski.exposure.histogram(img_eq)

# === CLAHE ===
h_clahe = img_hsv[:, :, 0]
s_clahe = img_hsv[:, :, 1]
v_clahe = clahe_channel(img_hsv[:, :, 2])

img_eq_CLAHE = np.stack([h_clahe, s_clahe, v_clahe], axis=-1)
img_eq_CLAHE = hsv2rgb(img_eq_CLAHE)
h_eq_CLAHE, c_eq_CLAHE = ski.exposure.histogram(img_eq_CLAHE)

fig, axs = plt.subplots(3, 4, layout="constrained")

axs[0, 0].imshow(img_rgb, cmap='gray')
axs[0, 0].set_title("Original")
axs[0, 1].bar(c_orig, h_orig, 1.1)

axs[1, 0].imshow(img_clara, cmap='gray')
axs[1, 0].set_title("Aclarada 1")
axs[1, 1].bar(c_clara_1, h_clara_1, 1.1)

#axs[2, 0].imshow(img_clara_2, cmap='gray')
axs[2, 0].set_title("Aclarada 2")
#axs[2, 1].bar(c_clara_2, h_clara_2, 1.1)

axs[0, 2].imshow(img_eq, cmap='gray')
axs[0, 2].set_title("Ecualizada")
axs[0, 3].bar(c_eq, h_eq, 1.1)

axs[1, 2].imshow(img_eq_CLAHE, cmap='gray')
axs[1, 2].set_title("CLAHE")
axs[1, 3].bar(c_eq_CLAHE, h_eq_CLAHE, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
