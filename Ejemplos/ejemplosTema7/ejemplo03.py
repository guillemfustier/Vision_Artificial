import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Leer la imagen y crear mapa de bordes usando el gradiente de Sobel
image = ski.io.imread("Ejemplos/ejemplosTema7/images/cuadros.png")
image = image - image.min()  # El valor del fondo debe ser 0

gradiente = ski.filters.sobel(image)

maximo = gradiente.max()
low = maximo * 0.1
high = maximo * 0.2
mapa_bordes = ski.filters.apply_hysteresis_threshold(gradiente, low, high)

mapa_bordes = ski.morphology.thin(mapa_bordes)  # Reduce el grosor de los bordes a un solo píxel

# Transformada de Radon de la imagen original
angulos = np.arange(0, 180, 0.5)  # 180 valores. Precisión de 1 grado. Cambiar 1 por 10 para probar menos precisión
sinogram = ski.transform.radon(image, theta=angulos, circle=True)

# Reconstrucción de la imagen original con la transformada inversa
reconstruccion = ski.transform.iradon(sinogram, theta=angulos, filter_name='hann')

# Transformada de Radon del mapa de bordes
sinogram_bordes = ski.transform.radon(mapa_bordes, theta=angulos, circle=True)

# Detección de picos
hs_selected, ang_selected, dists_selected = ski.transform.hough_line_peaks(sinogram_bordes,
                                                                           np.arange(sinogram_bordes.shape[1]),
                                                                           np.arange(sinogram_bordes.shape[0]),
                                                                           threshold=sinogram_bordes.max() / 8)
# Seleccionamos solo los picos
sinograma_solo_picos = np.zeros(sinogram_bordes.shape)  # Imagen en negro sobre la que marcamos los picos detactados a 1
for hs, ang, dist in zip(hs_selected, ang_selected, dists_selected):
    sinograma_solo_picos[dist, ang] = 1

# Reconstrucción a partir únicamente de los picos
reconstruccion_bordes = ski.transform.iradon(sinograma_solo_picos, theta=angulos, filter_name='hann')

# Visualización de los resultados
fig, axs = plt.subplots(nrows=2, ncols=3, layout="constrained")
fig.suptitle("Ejemplo - Transformada de Radon", fontsize=24)

axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title("Imagen original", size=16)

axs[0, 1].imshow(sinogram, cmap='gray')
axs[0, 1].set_title("Radon Space", size=16)

axs[0, 2].imshow(reconstruccion, cmap='gray')
axs[0, 2].set_title("Transformada inversa", size=16)

axs[1, 0].imshow(mapa_bordes, cmap='gray')
axs[1, 0].set_title("Mapa de bordes", size=16)

sinogram_bordes_log = np.log(sinogram_bordes + 1)
sinogram_bordes_log /= sinogram_bordes_log.max()
axs[1, 1].imshow(sinogram_bordes_log, cmap='gray')
axs[1, 1].set_title("Radon Space (escala logarítmica)", size=16)

axs[1, 2].imshow(reconstruccion_bordes, cmap='gray')
axs[1, 2].set_title("Transf. Inversa (solo los picos)", size=16)

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()
