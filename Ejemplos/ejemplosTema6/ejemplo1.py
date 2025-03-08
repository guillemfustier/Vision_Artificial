import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

image1 = ski.io.imread("images/sintetica.png")  # Probar también con cuadros
image2 = ski.io.imread("images/sintetica_ruidosa.png")


def filtrar(image, nombre_filtro):
    if nombre_filtro == "roberts":
        dir1 = "_neg_diag"
        dir2 = "_pos_diag"
    else:
        dir1 = "_h"
        dir2 = "_v"
    img1 = eval("ski.filters." + nombre_filtro + dir1 + "(image)")
    img2 = eval("ski.filters." + nombre_filtro + dir2 + "(image)")
    img3 = eval("ski.filters." + nombre_filtro + "(image)")
    maximo = img3.max()
    low = maximo * 0.1  # Probar otros valores
    high = maximo * 0.2
    img4 = ski.filters.apply_hysteresis_threshold(img3, low, high)
    return [image, img1, img2, img3, img4]


def mostrar(img_fila1, img_fila2, msg_fila1, msg_fila2, titulo):
    fig, ax = plt.subplots(nrows=2, ncols=len(img_fila1), layout="constrained")
    fig.suptitle(titulo, fontsize=24)
    for i in range(len(img_fila1)):
        ax[0, i].imshow(img_fila1[i], cmap='gray')
        ax[0, i].set_title(msg_fila1[i], fontsize=16)
        ax[1, i].imshow(img_fila2[i], cmap='gray')
        ax[1, i].set_title(msg_fila2[i], fontsize=16)
    for a in ax.ravel():
        a.set_axis_off()
    plt.show()


filtros = ["prewitt", "sobel", "roberts", "scharr", "farid"]

for nf in filtros:
    images1 = filtrar(image1, nf)
    images2 = filtrar(image2, nf)
    mostrar(images1, images2, ["Img sintética", "H", "V", "Módulo", "Bordes"],
            ["Img ruidosa", "H", "V", "Módulo", "Bordes"], nf)

canny_sigmas = [1, 3, 5]


def canny(image, sigmas, *args, **kwargs):
    images = [image]
    for sigma in sigmas:
        img = ski.feature.canny(image, sigma=sigma, *args, **kwargs)
        images.append(img)
    return images


images1 = canny(image1, canny_sigmas)
images2 = canny(image2, canny_sigmas)
mostrar(images1, images2, ["Img sintética", "$\\sigma=1$", "$\\sigma=3$", "$\\sigma=5$"],
        ["Img ruidosa", "$\\sigma=1$", "$\\sigma=3$", "$\\sigma=5$"], "Canny (umbrales auto)")

images1 = canny(image1, canny_sigmas, low_threshold=20, high_threshold=30)
images2 = canny(image2, canny_sigmas, low_threshold=20, high_threshold=30)
mostrar(images1, images2, ["Img sintética", "$\\sigma=1$", "$\\sigma=3$", "$\\sigma=5$"],
        ["Img ruidosa", "$\\sigma=1$", "$\\sigma=3$", "$\\sigma=5$"], "Canny (umbrales manuales)")

# Laplaciana

lp1 = ski.filters.laplace(image1)
images1 = [image1, lp1]

lp2 = ski.filters.laplace(image2)
images2 = [image2, lp2]
mostrar(images1, images2, ["Img sintética", "Laplaciana", "Valor absoluto"],
        ["Img ruidosa", "Laplaciana", "Valor absoluto"], "Laplaciana")
