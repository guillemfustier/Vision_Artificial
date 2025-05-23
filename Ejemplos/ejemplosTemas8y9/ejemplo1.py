import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

img = ski.io.imread("Ejemplos/ejemplosTemas8y9/images/llaves_monedas.png")

umbral_otsu = ski.filters.threshold_otsu(img)
img_umbralizada = img < umbral_otsu  # Objetos más oscuros que el fondo
print(f"Umbral seleccionado (Otsu): {umbral_otsu}")

##### Tema 9 #####
st_elem = ski.morphology.disk(12)
img_cierre = ski.morphology.binary_closing(img_umbralizada, footprint=st_elem)
##################

img_etiquetada = ski.morphology.label(img_cierre)

props = ski.measure.regionprops(img_etiquetada)
for p in props:
    print(f"Etiqueta: {p.label} Área: {p.area} Excentricidad: {p.eccentricity:.2f}")

img_monedas = np.zeros(img.shape)
contador_monedas = 0
for p in props:
    if p.eccentricity < 0.75:
        img_monedas[img_etiquetada == p.label] = p.label
        contador_monedas += 1
print(f"Detectadas {contador_monedas} monedas")

# Visualizar resultados
fig, axs = plt.subplots(2, 3, layout="constrained")
for ax in axs.ravel():
    ax.set_axis_off()
fig.suptitle("Segmentación por umbralización", fontsize=24)

axs[0, 0].imshow(img, cmap="gray")
axs[0, 0].set_title("Original", fontsize=16)

histograma, centros = ski.exposure.histogram(img)
axs[0, 1].bar(centros, histograma, 1.1)
axs[0, 1].set_title("Histograma", fontsize=16)
axs[0, 1].set_axis_on()
axs[0, 1].set_xticks([0, 64, 128, 192, 255])

axs[0, 2].imshow(img_umbralizada, cmap="gray")
axs[0, 2].set_title("Umbralizada", fontsize=16)

axs[1, 0].imshow(img_cierre, cmap="gray")
axs[1, 0].set_title("Cierre", fontsize=16)

axs[1, 1].imshow(img_etiquetada, interpolation="None", cmap="jet")
axs[1, 1].set_title("Componentes conexas", fontsize=16)

axs[1, 2].imshow(img_monedas, interpolation="None", cmap="jet")
axs[1, 2].set_title("Monedas", fontsize=16)

plt.show()
