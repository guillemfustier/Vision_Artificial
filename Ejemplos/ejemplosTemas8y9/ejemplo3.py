import matplotlib.pyplot as plt
import skimage as ski
import numpy as np


def generar_pseudo_color(img):
    maximo = img.max()
    random_color_map = np.random.rand(maximo, 3)
    res_color = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.max()):
        res_color[img == i] = random_color_map[i]
    return res_color


img = ski.io.imread("Ejemplos/ejemplosTemas8y9/images/toys.png")

segment_qshift = ski.segmentation.quickshift(img)
segment_slic = ski.segmentation.slic(img, n_segments=100)
segment_felsen = ski.segmentation.felzenszwalb(img, scale=300)

# Visualizar resultados
fig, axs = plt.subplots(nrows=1, ncols=4, layout="constrained")
for ax in axs.ravel():
    ax.set_axis_off()
fig.suptitle("Segmentación en color", fontsize=24)

axs[0].imshow(img)
axs[0].set_title("Original", fontsize=16)

axs[1].imshow(generar_pseudo_color(segment_qshift))
axs[1].set_title("Quick Shift", fontsize=16)

axs[2].imshow(generar_pseudo_color(segment_slic))
axs[2].set_title("SLIC", fontsize=16)

axs[3].imshow(generar_pseudo_color(segment_felsen))
axs[3].set_title("Felzenszwalb", fontsize=16)

plt.show()
