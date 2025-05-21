import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Mejorar el código usando bucles for, en vez de copiar y pegar.

# Leer la imagen y crear mapa de bordes usando el gradiente de Sobel
image = ski.io.imread("Prácticas/practica6/images/monedas1.png")
image2 = ski.io.imread("Prácticas/practica6/images/monedas2.png")
image3 = ski.io.imread("Prácticas/practica6/images/monedas3.png")

#image = ski.color.rgb2gray(image)  # Convertir a niveles de gris

mapa_bordes = ski.feature.canny(image, sigma=1.2)

mapa_bordes = ski.morphology.thin(mapa_bordes)  # Reduce el grosor de los bordes a un solo píxel

# Transformada de Hough para círculos
radios_posibles = np.arange(17, 25, 2)  # Buscará círculos con radios entre 10 y 30 de 2 en 2
hough_res = ski.transform.hough_circle(mapa_bordes, radios_posibles)
accums, cx, cy, radii = ski.transform.hough_circle_peaks(hough_res, radios_posibles, min_xdistance=10, min_ydistance=10,
                                                         threshold=hough_res.max() / 2)  # El threshold no debería ser necesario. Se supone que es el valor por defecto

# Dibujar los círculos en la imagen resultado
resultado = np.zeros(image.shape)
for fila, col, radio in zip(cy, cx, radii):
    circy, circx = ski.draw.circle_perimeter(fila, col, radio, shape=image.shape)  # Dibuja un círculo
    resultado[circy, circx] = 1

mapa_bordes2 = ski.feature.canny(image2, sigma=1.2)

mapa_bordes2 = ski.morphology.thin(mapa_bordes2)  # Reduce el grosor de los bordes a un solo píxel

# Transformada de Hough para círculos
radios_posibles = np.arange(17, 25, 2)  # Buscará círculos con radios entre 10 y 30 de 2 en 2
hough_res2 = ski.transform.hough_circle(mapa_bordes2, radios_posibles)
accums2, cx2, cy2, radii2 = ski.transform.hough_circle_peaks(hough_res2, radios_posibles, min_xdistance=10, min_ydistance=10,
                                                         threshold=hough_res2.max() / 2)  # El threshold no debería ser necesario. Se supone que es el valor por defecto
resultado2 = np.zeros(image2.shape)
for fila, col, radio in zip(cy2, cx2, radii2):
    circy, circx = ski.draw.circle_perimeter(fila, col, radio, shape=image.shape)  # Dibuja un círculo
    resultado2[circy, circx] = 1

mapa_bordes3 = ski.feature.canny(image3, sigma=1.2)

mapa_bordes3 = ski.morphology.thin(mapa_bordes3)  # Reduce el grosor de los bordes a un solo píxel

# Transformada de Hough para círculos
radios_posibles = np.arange(17, 25, 2)  # Buscará círculos con radios entre 10 y 30 de 2 en 2
hough_res3 = ski.transform.hough_circle(mapa_bordes3, radios_posibles)
accums, cx3, cy3, radii3 = ski.transform.hough_circle_peaks(hough_res3, radios_posibles, min_xdistance=10, min_ydistance=10,
                                                         threshold=hough_res3.max() / 2)  # El threshold no debería ser necesario. Se supone que es el valor por defecto

# Dibujar los círculos en la imagen resultado


# Dibujar los círculos en la imagen resultado
resultado3 = np.zeros(image.shape)
for fila, col, radio in zip(cy3, cx3, radii3):
    circy, circx = ski.draw.circle_perimeter(fila, col, radio, shape=image.shape)  # Dibuja un círculo
    resultado3[circy, circx] = 1

# Visualizar resultados
fig, axs = plt.subplots(nrows=3, ncols=3, layout="constrained")
fig.suptitle("Ejemplo - Transformada de Hough para círculos", fontsize=24)

axs[0,0].imshow(image, cmap='gray')
axs[0,0].set_title("Imagen original", size=16)

axs[0,1].imshow(mapa_bordes, cmap='gray')
axs[0,1].set_title("Mapa de bordes (Canny)", size=16)

axs[0,2].imshow(resultado, cmap='gray')
axs[0,2].set_title("HT para círculos", size=16)

axs[1,0].imshow(image2, cmap='gray')
axs[1,0].set_title("Imagen original", size=16)

axs[1,1].imshow(mapa_bordes2, cmap='gray')
axs[1,1].set_title("Mapa de bordes (Canny)", size=16)

axs[1,2].imshow(resultado2, cmap='gray')
axs[1,2].set_title("HT para círculos", size=16)

axs[2,0].imshow(image3, cmap='gray')
axs[2,0].set_title("Imagen original", size=16)

axs[2,1].imshow(mapa_bordes3, cmap='gray')
axs[2,1].set_title("Mapa de bordes (Canny)", size=16)

axs[2,2].imshow(resultado3, cmap='gray')
axs[2,2].set_title("HT para círculos", size=16)

for a in axs.ravel():
    a.set_axis_off()
plt.show()