import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

monedas1 = ski.io.imread("Prácticas/practica7/images/monedas1.png")
monedas2 = ski.io.imread("Prácticas/practica7/images/monedas2.png")
monedas3 = ski.io.imread("Prácticas/practica7/images/monedas3.png")

imgs_monedas = [monedas1, monedas2, monedas3]
imgs_resultados = []
i = 0

for img in imgs_monedas:
    umbral_otsu = ski.filters.threshold_otsu(img)
    img_umbralizada = img > umbral_otsu  # Objetos más oscuros que el fondo
    print(f"Umbral seleccionado (Otsu): {umbral_otsu}")

    st_elem = ski.morphology.disk(2)
    
    img_cierre = ski.morphology.binary_closing(img_umbralizada, footprint=st_elem)

    st_elem1 = ski.morphology.disk(11)
    st_elem2 = ski.morphology.disk(7)
    img_erosion = ski.morphology.binary_erosion(img_cierre, footprint=st_elem1)
    img_cierre = ski.morphology.binary_dilation(img_erosion, footprint=st_elem2)

    img_etiquetada = ski.morphology.label(img_cierre)

    props = ski.measure.regionprops(img_etiquetada)
    for p in props:
        print(f"Etiqueta: {p.label} Área: {p.area} Excentricidad: {p.eccentricity:.2f}")

    resultado = ski.color.gray2rgb(img*0)
    contador_monedas_euro = 0
    contador_monedas_cent = 0
    ecc_list = [(0.10, 0.07), (0.20, 0.13)]
    for p in props:
        if p.eccentricity < 0.50:
            if p.area < 750:
                resultado[img_etiquetada == p.label] = (0, 255, 0)
                contador_monedas_cent += 1
            else:
                resultado[img_etiquetada == p.label] = (255, 0, 0)
                contador_monedas_euro += 1
    print(f"Detectadas {contador_monedas_euro} monedas de 1 euro")
    print(f"Detectadas {contador_monedas_cent} monedas de 10 céntimos")
    imgs_resultados.append(resultado)
    i += 1

# Visualizar resultados
fig, axs = plt.subplots(2, 3, layout="constrained")
for ax in axs.ravel():
    ax.set_axis_off()

for i in range(3):
    axs[0, i].imshow(imgs_monedas[i], cmap="gray")
    axs[0, i].set_title(f"Original {i+1}", fontsize=16)

    axs[1, i].imshow(imgs_resultados[i], cmap="gray")
    axs[1, i].set_title("Monedas detectadas", fontsize=16)

plt.show()
