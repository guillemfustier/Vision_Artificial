import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

# Leer la imagen y crear mapa de bordes usando el gradiente de Sobel
image = ski.io.imread("Ejemplos/ejemplosTema7/images/cuadros.png")

gradiente = ski.filters.sobel(image)

maximo = gradiente.max()
low = maximo * 0.1
high = maximo * 0.2
mapa_bordes = ski.filters.apply_hysteresis_threshold(gradiente, low, high)

mapa_bordes = ski.morphology.thin(mapa_bordes)  # Reduce el grosor de los bordes a un solo píxel

# Transformada de Hough
posibles_angulos = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)  # Ángulos cada 0.5 grados
hough_space, angulos, distancias = ski.transform.hough_line(mapa_bordes, theta=posibles_angulos)
hs_selected, ang_selected, dists_selected = ski.transform.hough_line_peaks(hough_space, angulos, distancias,
                                                                           threshold=20)

# Transfromada de Hough Probabilística y Proresiva
segmentos = ski.transform.probabilistic_hough_line(mapa_bordes, threshold=10, line_length=5, line_gap=3)

# Visualización de resultados
fig, ax = plt.subplots(nrows=2, ncols=3, layout="constrained")
fig.suptitle("Ejemplo - Transformada de Hough", fontsize=24)
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title("Imagen original", size=16)

ax[0, 1].imshow(gradiente, cmap='gray')
ax[0, 1].set_title("Gradiente Sobel", size=16)

ax[0, 2].imshow(mapa_bordes, cmap='gray')
ax[0, 2].set_title("Mapa de bordes (thin)", size=16)

hough_logscale = np.log(hough_space + 1)
hough_logscale /= hough_logscale.max()
ax[1, 0].imshow(hough_logscale, cmap='gray')
ax[1, 0].set_title("Espacio de Hough (escala logarítmica)", size=16)

ax[1, 1].set_title("Líneas infinitas", size=16)
ax[1, 1].imshow(np.zeros(image.shape), cmap='gray')
ax[1, 1].set_xlim((0, image.shape[1] - 1))
ax[1, 1].set_ylim((image.shape[0] - 1, 0))
for angulo, dist in zip(ang_selected, dists_selected):
    (x0, y0) = dist * np.array([np.cos(angulo), np.sin(angulo)])
    ax[1, 1].axline((x0, y0), slope=np.tan(angulo + np.pi / 2), color='b')  # Dibujar línea infinita

ax[1, 2].set_title("Segmentos con PPHF", size=16)
ax[1, 2].imshow(np.zeros(image.shape), cmap='gray')
for segmento in segmentos:
    p0, p1 = segmento
    ax[1, 2].plot((p0[0], p1[0]), (p0[1], p1[1]), color='r')  # Dibujar segmento

for a in ax.ravel():
    a.set_axis_off()
plt.show()
