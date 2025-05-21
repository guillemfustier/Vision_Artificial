import skimage as ski
import matplotlib.pyplot as plt
import math
import numpy as np

imagen = ski.io.imread("Prácticas/practica6/images/cuadros.png")

# Añadir ruido Gaussiano
gaussian_values = (0.001,
                   0.0015,
                   0.0025)
total_images = []
total_images.append(imagen)
for i in range(len(gaussian_values)):
    img_noise = ski.util.random_noise(imagen, mode="gaussian", var=gaussian_values[i])
    total_images.append(img_noise)

def filtrar_sobel(images):
    filtradas = []
    for image in images:
        img_todo = ski.filters.sobel(imagen)          # magnitud
        maximo = img_todo.max()
        low = maximo * 0.1  # Probar otros valores
        high = maximo * 0.2
        img_bordes = ski.filters.apply_hysteresis_threshold(img_todo, low, high)
        img_thin = ski.morphology.thin(img_bordes)
        filtradas.append(img_thin)
    return filtradas

def get_hough(imgs_list):
    imgs_hough = []
    for imagen_bordes in imgs_list:
        img_hough = ski.transform.probabilistic_hough_line(imagen_bordes)
        imgs_hough.append(img_hough)
    return imgs_hough

def canny(total_images, sigma, *args, **kwargs):
    imgs_canny = []
    for image in total_images:
        img = ski.feature.canny(image, sigma=sigma, *args, **kwargs)
        imgs_canny.append(img)
    return imgs_canny


imgs_sobel = filtrar_sobel(total_images)
imgs_hough_sobel = get_hough(imgs_sobel)
imgs_canny = canny(total_images, 3)
imgs_hough_canny = get_hough(imgs_canny)


imagenes_conjunto = [total_images, imgs_sobel, imgs_hough_sobel, imgs_canny, imgs_hough_canny]

fig, ax = plt.subplots(nrows=4, ncols=5, layout="constrained")
fig.suptitle("Ejercicio 1", fontsize=24)
for i in range(len(imagenes_conjunto)):
    for j in range(len(imagenes_conjunto[0])):
        if i == 4 or i == 2:
            image = imagenes_conjunto[i-1][j]
            segmentos = ski.transform.probabilistic_hough_line(imagenes_conjunto[i-1][j], threshold=10, line_length=5, line_gap=3)
            ax[j, i].imshow(np.zeros(image.shape), cmap='gray')
            for segmento in segmentos:
                p0, p1 = segmento
                ax[j, i].plot((p0[0], p1[0]), (p0[1], p1[1]), color='r')  # Dibujar segmento
        else:
            ax[j, i].imshow(imagenes_conjunto[i][j], cmap='gray')
for a in ax.ravel():
    a.set_axis_off()

for a in ax.ravel():
    a.set_axis_off()
plt.show()