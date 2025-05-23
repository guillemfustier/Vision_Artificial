import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Leer la imagen y crear mapa de bordes usando el gradiente de Sobel
image = ski.io.imread("Ejemplos/ejemplosTema7/images/monedas.jpg")
image = ski.color.rgb2gray(image)  # Convertir a niveles de gris

gradiente = ski.filters.sobel(image)

maximo = gradiente.max()
low = maximo * 0.2
high = maximo * 0.4
mapa_bordes = ski.filters.apply_hysteresis_threshold(gradiente, low, high)

mapa_bordes = ski.morphology.thin(mapa_bordes)  # Reduce el grosor de los bordes a un solo píxel

# Transformada de Hough para círculos
radios_posibles = np.arange(10, 30, 2)  # Buscará círculos con radios entre 10 y 30 de 2 en 2
hough_res = ski.transform.hough_circle(mapa_bordes, radios_posibles)
accums, cx, cy, radii = ski.transform.hough_circle_peaks(hough_res, radios_posibles, min_xdistance=10, min_ydistance=10,
                                                         threshold=hough_res.max() / 2)  # El threshold no debería ser necesario. Se supone que es el valor por defecto

# Dibujar los círculos en la imagen resultado
resultado = np.zeros(image.shape)
for fila, col, radio in zip(cy, cx, radii):
    circy, circx = ski.draw.circle_perimeter(fila, col, radio, shape=image.shape)  # Dibuja un círculo
    resultado[circy, circx] = 1

# Visualizar resultados
fig, axs = plt.subplots(nrows=1, ncols=3, layout="constrained")
fig.suptitle("Ejemplo - Transformada de Hough para círculos", fontsize=24)

axs[0].imshow(image, cmap='gray')
axs[0].set_title("Imagen original", size=16)

axs[1].imshow(mapa_bordes, cmap='gray')
axs[1].set_title("Mapa de bordes (Sobel + thin)", size=16)

axs[2].imshow(resultado, cmap='gray')
axs[2].set_title("HT para círculos", size=16)

for ax in axs:
    ax.set_axis_off()

plt.show()
