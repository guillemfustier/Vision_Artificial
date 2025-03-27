import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
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
r_orig, g_orig, b_orig = img_original[:, :, 0], img_original[:, :, 1], img_original[:, :, 2]
h_orig_r, c_orig_r = ski.exposure.histogram(r_orig)
h_orig_g, c_orig_g = ski.exposure.histogram(g_orig)
h_orig_b, c_orig_b = ski.exposure.histogram(b_orig)

img_real = ski.util.img_as_float(img_original) * 255

# === ACLARAR ===
r_aclarar = aclarar_channel(img_original[:, :, 0])
g_aclarar = aclarar_channel(img_original[:, :, 1])
b_aclarar = aclarar_channel(img_original[:, :, 2])

img_clara = np.stack([r_aclarar, g_aclarar, b_aclarar], axis=-1)
h_clara_r, c_clara_r = ski.exposure.histogram(r_aclarar)
h_clara_g, c_clara_g = ski.exposure.histogram(g_aclarar)
h_clara_b, c_clara_b = ski.exposure.histogram(b_aclarar)

# === ACLARAR 2 ===
r_aclarar_2 = aclarar_channel(img_clara[:, :, 0])
g_aclarar_2 = aclarar_channel(img_clara[:, :, 1])
b_aclarar_2 = aclarar_channel(img_clara[:, :, 2])

img_clara_2 = np.stack([r_aclarar_2, g_aclarar_2, b_aclarar_2], axis=-1)
h_clara_2_r, c_clara_2_r = ski.exposure.histogram(r_aclarar_2)
h_clara_2_g, c_clara_2_g = ski.exposure.histogram(g_aclarar_2)
h_clara_2_b, c_clara_2_b = ski.exposure.histogram(b_aclarar_2)

# === ECUALIZADA ===
r_eq = ecualizar_channel(img_original[:, :, 0])
g_eq = ecualizar_channel(img_original[:, :, 1])
b_eq = ecualizar_channel(img_original[:, :, 2])

img_eq = np.stack([r_eq, g_eq, b_eq], axis=-1)
h_eq_r, c_eq_r = ski.exposure.histogram(r_eq)
h_eq_g, c_eq_g = ski.exposure.histogram(g_eq)
h_eq_b, c_eq_b = ski.exposure.histogram(b_eq)

# === CLAHE ===
r_clahe = clahe_channel(img_original[:, :, 0])
g_clahe = clahe_channel(img_original[:, :, 1])
b_clahe = clahe_channel(img_original[:, :, 2])

img_eq_CLAHE = np.stack([r_clahe, g_clahe, b_clahe], axis=-1)
h_eq_CLAHE_r, c_eq_CLAHE_r = ski.exposure.histogram(r_clahe)
h_eq_CLAHE_g, c_eq_CLAHE_g = ski.exposure.histogram(g_clahe)
h_eq_CLAHE_b, c_eq_CLAHE_b = ski.exposure.histogram(b_clahe)


fig, axs = plt.subplots(5, 4, layout="constrained")

axs[0, 0].imshow(img_original, cmap='gray')
axs[0, 0].set_title("Original")
axs[0, 1].bar(c_orig_r, h_orig_r, 2, color='red')
axs[0, 2].bar(c_orig_g, h_orig_g, 2, color='green')
axs[0, 3].bar(c_orig_b, h_orig_b, 2, color='blue')

axs[1, 0].imshow(img_clara, cmap='gray')
axs[1, 0].set_title("Aclarada 1")
axs[1, 1].bar(c_clara_r, h_clara_r, 2, color='red')
axs[1, 2].bar(c_clara_g, h_clara_g, 2, color='green')
axs[1, 3].bar(c_clara_b, h_clara_b, 2, color='blue')

axs[2, 0].imshow(img_clara_2, cmap='gray')
axs[2, 0].set_title("Aclarada 2")
axs[2, 1].bar(c_clara_2_r, h_clara_2_r, 2, color='red')
axs[2, 2].bar(c_clara_2_g, h_clara_2_g, 2, color='green')
axs[2, 3].bar(c_clara_2_b, h_clara_2_b, 2, color='blue')

axs[3, 0].imshow(img_eq, cmap='gray')
axs[3, 0].set_title("Ecualizada")
axs[3, 1].bar(c_eq_r, h_eq_r, 2, color='red')
axs[3, 2].bar(c_eq_g, h_eq_g, 2, color='green')
axs[3, 3].bar(c_eq_b, h_eq_b, 2, color='blue')

axs[4, 0].imshow(img_eq_CLAHE, cmap='gray')
axs[4, 0].set_title("CLAHE")
axs[4, 1].bar(c_eq_CLAHE_r, h_eq_CLAHE_r, 2, color='red')
axs[4, 2].bar(c_eq_CLAHE_g, h_eq_CLAHE_g, 2, color='green')
axs[4, 3].bar(c_eq_CLAHE_b, h_eq_CLAHE_b, 2, color='blue')

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 4):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
