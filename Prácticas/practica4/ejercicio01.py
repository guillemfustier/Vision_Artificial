import skimage as ski
import matplotlib.pyplot as plt
import math

# Cargar y reescalar la imagen
imagen = ski.io.imread("Prácticas/practica4/images/ojo_azul.png")

# Añadir ruido Sal y Pimienta (S&P)
sp_values = (0.01,  # 1%
             0.05,  # 5%
             0.25)  # 10%

sp_noise = []
for sp in sp_values:
    # 'amount' controla el porcentaje de ruido sal y pimienta
    img_noise = ski.util.random_noise(imagen, mode="s&p", amount=sp)
    sp_noise.append(img_noise)

# Añadir ruido Gaussiano
gaussian_values = (0.001,  
                   0.005,  
                   0.010)  

gaussian_noise = []
for var_ in gaussian_values:
    img_noise = ski.util.random_noise(imagen, mode="gaussian", var=var_)
    gaussian_noise.append(img_noise)

# Función para mostrar en una misma figura
def mostrar_por_filtro(titulo, img_original, sp_vals, gauss_vals, sp_imgs, gauss_imgs):
    fig, axs = plt.subplots(2, max(len(sp_vals), len(gauss_vals)) + 1, layout="constrained")
    fig.suptitle(titulo, size=24)

    # Mostrar imagen original
    axs[0, 0].imshow(img_original)
    axs[0, 0].set_title("Original")

    # Mostrar cada imagen con ruido Sal y Pimienta
    for i, sp_val in enumerate(sp_vals):
        axs[0, i+1].imshow(sp_imgs[i])
        axs[0, i+1].set_title(f"S&P {sp_val * 100:.0f}%")

    # Mostrar cada imagen con ruido Gaussiano
    for i, var_ in enumerate(gauss_vals):
        sigma = math.sqrt(var_)
        axs[1, i+1].imshow(gauss_imgs[i])
        axs[1, i+1].set_title(f"Gaussian σ={sigma:.2f}")

    # Ajustes estéticos
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.show()

# Mostrar los resultados
mostrar_por_filtro("Imágenes con ruido", imagen, sp_values, gaussian_values, sp_noise, gaussian_noise)
