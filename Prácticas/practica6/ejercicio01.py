import skimage as ski
import matplotlib.pyplot as plt
import math

imagen = ski.io.imread("Prácticas/practica6/images/cuadros.png")

# Añadir ruido Gaussiano
gaussian_values = (0.001,
                   0.0015,
                   0.0025)
gaussian_images = []
for i in range(len(gaussian_values)):
    img_noise = ski.util.random_noise(imagen, mode="gaussian", var=gaussian_values[i])
    gaussian_images.append(img_noise)

def filtrar(image, nombre_filtro):
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
    img_threshold = ski.filters.apply_hysteresis_threshold(img_todo, low, high)
    return [image, img_horizontal, img_vertical, img_todo, img_threshold]

filtradas = filtrar(imagen, "sobel")



fig, ax = plt.subplots(nrows=5, ncols=2, layout="constrained")
ax[0, 0].imshow(imagen, cmap='gray')
ax[0, 1].imshow(gaussian_images[0], cmap='gray')
ax[1, 0].imshow(gaussian_images[1], cmap='gray')
ax[1, 1].imshow(gaussian_images[2], cmap='gray')
ax[2, 0].imshow(filtradas[0], cmap='gray')
ax[2, 1].imshow(filtradas[1], cmap='gray')
ax[3, 0].imshow(filtradas[2], cmap='gray')
ax[3, 1].imshow(filtradas[3], cmap='gray')
ax[4, 0].imshow(filtradas[4], cmap='gray')
for a in ax.ravel():
    a.set_axis_off()
plt.show()