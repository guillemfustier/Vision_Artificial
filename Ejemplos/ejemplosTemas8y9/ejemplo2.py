import skimage as ski
import matplotlib.pyplot as plt

image = ski.io.imread("Ejemplos/ejemplosTemas8y9/images/texto.png")

umbral_global = ski.filters.threshold_otsu(image)
binaria_global = image > umbral_global

umbrales_locales = ski.filters.threshold_local(image, 25, offset=10)
binaria_local = image > umbrales_locales

# Visualizar resultados
fig, axs = plt.subplots(2, 2, layout="constrained")
for ax in axs.ravel():
    ax.set_axis_off()
fig.suptitle("Umbralización global vs local", fontsize=24)

axs[0, 0].imshow(image, cmap="gray")
axs[0, 0].set_title("Original", fontsize=16)

axs[1, 0].imshow(binaria_global, cmap="gray")
axs[1, 0].set_title("Global", fontsize=16)

axs[1, 1].imshow(binaria_local, cmap="gray")
axs[1, 1].set_title("Local", fontsize=16)

plt.show()
