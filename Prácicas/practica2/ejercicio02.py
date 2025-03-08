import skimage as ski
import matplotlib.pyplot as plt

img_original = ski.io.imread("images/girl.pgm")
h_orig, c_orig = ski.exposure.histogram(img_original)

fig, axs = plt.subplots(4, 4, layout="constrained")

axs[0, 0].imshow(img_original, cmap='gray')
axs[0, 0].set_title("Original")
axs[0, 1].bar(c_orig, h_orig, 1.1)

kernels = [8, 2]
clips = [0.05, 0.01, 0.1]

i = 1
for k in kernels:
    for c in clips:
        img_eq_CLAHE = ski.exposure.equalize_adapthist(img_original, kernel_size=k, clip_limit=c)
        img_eq_CLAHE = ski.util.img_as_ubyte(img_eq_CLAHE)
        h_eq_CLAHE, c_eq_CLAHE = ski.exposure.histogram(img_eq_CLAHE)

        tittle = f"CLAHE (Kern={k}, clip_l={c})"
        if i < 4:
            axs[i, 0].imshow(img_eq_CLAHE, cmap='gray')
            axs[i, 0].set_title(tittle)
            axs[i, 1].bar(c_eq_CLAHE, h_eq_CLAHE, 1.1)
        else:
            axs[i - 4, 2].imshow(img_eq_CLAHE, cmap='gray')
            axs[i - 4, 2].set_title(tittle)
            axs[i - 4, 3].bar(c_eq_CLAHE, h_eq_CLAHE, 1.1)
        i += 1

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
    axs_lineal[i].set_axis_off()
    axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])

plt.show()
