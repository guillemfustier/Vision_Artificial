# Objetivo
# -Filtros de distintos tamaños:
#  * mean y median -> footprint
#  * gaussiano -> sigma

import skimage as ski
import matplotlib.pyplot as plt
import math

imagen = ski.io.imread("Prácticas/practica4/images/hombre_con_ruido.png")
imagen = ski.transform.rescale(imagen, 0.5, order=5)

amount_sp_noise = 0.05  # 5%
amount_gaussian_noise = 0.0025  # sigma = 0.05

# Filtro Media
mean_sizes = [3, 5, 7]
mean_filter = []
for i in range(len(mean_sizes)):
    filtered_image = ski.filters.rank.mean(ski.util.img_as_ubyte(imagen),
                                            footprint=ski.morphology.footprint_rectangle((mean_sizes[i], mean_sizes[i])))
    mean_filter.append(filtered_image)

# Filtro Gaussiano
sigma_values = [1, 1.25, 1.75]  # Ojo: En el filtro sigma. Al generar ruido la varianza
gaussian_filter = []
for i in range(len(sigma_values)):
    filtered_image = ski.filters.gaussian(imagen, sigma=sigma_values[i])
    gaussian_filter.append(filtered_image)

# Filtro mediana
median_sizes = [3, 5, 7]
median_filter = []
for i in range(len(median_sizes)):
    filtered_image = ski.filters.median(imagen, footprint=ski.morphology.footprint_rectangle((median_sizes[i], median_sizes[i])))
    median_filter.append(filtered_image)


# Mostrar Resultados
def mostrar_por_filtro(titulo, tipo, imagen_original, values, results):
    fig, axs = plt.subplots(2, len(values) + 1, layout="constrained")
    fig.suptitle(titulo, size=24)
    axs[0, 0].imshow(imagen_original, cmap='gray')
    axs[0, 0].set_title("Hombre con ruido")

    for j in range(len(values)):
        axs[0, j + 1].imshow(results[j], cmap='gray')
        axs[0, j + 1].set_title(f"{values[j]}x{values[j]}" if tipo == "x" else f"$\\sigma$ = {values[j]:0.1f}")

    ax = axs.ravel()
    for a in ax:
        a.set_axis_off()
    plt.show()

def mostrar_los_tres_filtros(imagen_original,
                             mean_sizes, mean_filter,
                             sigma_values, gaussian_filter,
                             median_sizes, median_filter):

    # 1) Determina cuántas columnas necesitarás (la original + el número máximo de parámetros):
    n_cols = 1 + max(len(mean_sizes), len(sigma_values), len(median_sizes))

    # 2) Crea una figura con 3 filas (una por cada filtro):
    fig, axs = plt.subplots(nrows=3, ncols=n_cols, layout="constrained", figsize=(4*n_cols, 12))

    # ---------------------------------------------------------------------
    # FILA 0 -> Filtro Media
    # ---------------------------------------------------------------------
    axs[0, 0].imshow(imagen_original, cmap='gray')
    axs[0, 0].set_title("Original con ruido")
    for i, size in enumerate(mean_sizes):
        axs[0, i+1].imshow(mean_filter[i], cmap='gray')
        axs[0, i+1].set_title(f"Media {size}x{size}")

    # ---------------------------------------------------------------------
    # FILA 1 -> Filtro Gaussiano
    # ---------------------------------------------------------------------
    axs[1, 0].imshow(imagen_original, cmap='gray')
    axs[1, 0].set_title("Original con ruido")
    for i, sigma in enumerate(sigma_values):
        axs[1, i+1].imshow(gaussian_filter[i], cmap='gray')
        axs[1, i+1].set_title(f"Gauss σ={sigma:.1f}")

    # ---------------------------------------------------------------------
    # FILA 2 -> Filtro Mediana
    # ---------------------------------------------------------------------
    axs[2, 0].imshow(imagen_original, cmap='gray')
    axs[2, 0].set_title("Original con ruido")
    for i, size in enumerate(median_sizes):
        axs[2, i+1].imshow(median_filter[i], cmap='gray')
        axs[2, i+1].set_title(f"Mediana {size}x{size}")

    # ---------------------------------------------------------------------
    # Ajustes finales: quitar ejes y mostrar
    # ---------------------------------------------------------------------
    for row in axs:
        for ax in row:
            ax.set_axis_off()

    fig.suptitle("Comparación de los tres filtros", fontsize=20)
    plt.show()


"""
mostrar_por_filtro("Filtro media", "x", imagen, mean_sizes, mean_filter)
mostrar_por_filtro("Filtro Gaussiano", "%", imagen, sigma_values, gaussian_filter)
mostrar_por_filtro("Filtro mediana", "x", imagen, median_sizes, median_filter)
"""
mostrar_los_tres_filtros(imagen, mean_sizes, mean_filter, sigma_values, gaussian_filter, median_sizes, median_filter)