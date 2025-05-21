import skimage as ski
import matplotlib.pyplot as plt
import math

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

def filtrar(images, nombre_filtro):
    filtradas = []
    for image in images:
        if nombre_filtro == "roberts":      # en lugar de hacerlo en "h,v" lo hace en diagonal
            dir1 = "_neg_diag"
            dir2 = "_pos_diag"
        else:
            dir1 = "_h"     # para horizontal
            dir2 = "_v"     # para vertical
        img_horizontal = eval("ski.filters." + nombre_filtro + dir1 + "(image)")      # respuesta en h
        img_vertical = eval("ski.filters." + nombre_filtro + dir2 + "(image)")      # respuesta en v
        img_todo = eval("ski.filters." + nombre_filtro + "(image)")             # magnitud
        maximo = img_todo.max()
        low = maximo * 0.1  # Probar otros valores
        high = maximo * 0.2
        img_bordes = ski.filters.apply_hysteresis_threshold(img_todo, low, high)
        img_thin = ski.morphology.thin(img_bordes)
        filtradas.append(img_thin)
    return filtradas

imgs_sobel = filtrar(total_images, "sobel")
img_hough = ski.transform.probabilistic_hough_line(imgs_sobel[0])

fig, ax = plt.subplots(nrows=4, ncols=5, layout="constrained")
ax[0, 0].imshow(total_images[0], cmap='gray')
ax[1, 0].imshow(total_images[1], cmap='gray')
ax[2, 0].imshow(total_images[2], cmap='gray')
ax[3, 0].imshow(total_images[3], cmap='gray')

ax[0, 1].imshow(imgs_sobel[0], cmap='gray')
ax[1, 1].imshow(imgs_sobel[1], cmap='gray')
ax[2, 1].imshow(imgs_sobel[2], cmap='gray')
ax[3, 1].imshow(imgs_sobel[3], cmap='gray')

ax[0, 2].imshow(img_hough, cmap='gray')
# ax[1, 2].imshow(imgs_sobel[1], cmap='gray')
# ax[2, 2].imshow(imgs_sobel[2], cmap='gray')
# ax[3, 2].imshow(imgs_sobel[3], cmap='gray')

# ax[0, 3].imshow(imgs_sobel[0], cmap='gray')
# ax[1, 3].imshow(imgs_sobel[1], cmap='gray')
# ax[2, 3].imshow(imgs_sobel[2], cmap='gray')
# ax[3, 3].imshow(imgs_sobel[3], cmap='gray')

# ax[0, 4].imshow(imgs_sobel[0], cmap='gray')
# ax[1, 4].imshow(imgs_sobel[1], cmap='gray')
# ax[2, 4].imshow(imgs_sobel[2], cmap='gray')
# ax[3, ].imshow(imgs_sobel[3], cmap='gray')


for a in ax.ravel():
    a.set_axis_off()
plt.show()