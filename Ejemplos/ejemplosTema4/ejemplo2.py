# Objetivo
# -Filtros de distintos tamaños:
#  * mean y median -> footprint
#  * gaussiano -> sigma

import skimage as ski
import matplotlib.pyplot as plt
import math

imagen = ski.io.imread("images/boat.512.tiff")
imagen = ski.transform.rescale(imagen, 0.5, order=5)

amount_sp_noise = 0.05  # 5%
amount_gaussian_noise = 0.0025  # sigma = 0.05

ruido_sp = ski.util.random_noise(imagen, mode="s&p", amount=amount_sp_noise)
ruido_gaussiano = ski.util.random_noise(imagen, mode="gaussian", var=amount_gaussian_noise)

images = [ruido_sp, ruido_gaussiano]

# Filtro Media

mean_sizes = [3, 5, 7]
mean_filter = []
for img in images:
    for i in range(len(mean_sizes)):
        filtered_image = ski.filters.rank.mean(ski.util.img_as_ubyte(img),
                                               footprint=ski.morphology.footprint_rectangle((mean_sizes[i], mean_sizes[i])))
        mean_filter.append(filtered_image)

# Filtro Gaussiano
sigma_values = [0.5, 1, 2]  # Ojo: En el filtro sigma. Al generar ruido la varianza
gaussian_filter = []
for img in images:
    for i in range(len(sigma_values)):
        filtered_image = ski.filters.gaussian(img, sigma=sigma_values[i])
        gaussian_filter.append(filtered_image)

# Filtro mediana
median_sizes = [3, 5, 7]
median_filter = []
for img in images:
    for i in range(len(median_sizes)):
        filtered_image = ski.filters.median(img, footprint=ski.morphology.footprint_rectangle((median_sizes[i], median_sizes[i])))
        median_filter.append(filtered_image)


# Mostrar Resultados
def mostrar_por_filtro(titulo, tipo, images, values, results):
    fig, axs = plt.subplots(2, len(values) + 1, layout="constrained")
    fig.suptitle(titulo, size=24)
    for i, img in enumerate(images):
        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 0].set_title(
            f"Ruido S&P {amount_sp_noise * 100:0.0f}%" if i == 0 else f"Ruido Gaussiano $\\sigma$ = {math.sqrt(amount_gaussian_noise):0.2f}")

        for j in range(len(values)):
            axs[i, j + 1].imshow(results[i * len(values) + j], cmap='gray')
            axs[i, j + 1].set_title(f"{values[j]}x{values[j]}" if tipo == "x" else f"$\\sigma$ = {values[j]:0.1f}")

    ax = axs.ravel()
    for a in ax:
        a.set_axis_off()
    plt.show()


mostrar_por_filtro("Filtro media", "x", images, mean_sizes, mean_filter)
mostrar_por_filtro("Filtro Gaussiano", "%", images, sigma_values, gaussian_filter)
mostrar_por_filtro("Filtro mediana", "x", images, median_sizes, median_filter)
