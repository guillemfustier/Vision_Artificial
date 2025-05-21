import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

gaussian_noise=0.001

imagen = ski.io.imread("Prácticas/practica6/images/cuadros.png")  # Probar también con cuadros

imagen = ski.util.random_noise(imagen, mode="gaussian", var=gaussian_noise)

def filtrar(image, nombres_filtros):
    images = [image]
    for nf in nombres_filtros:
        if nf == "moravec":
            img = my_moravec(image)
        else:
            if nf == "fast":
                param = ", 9"
            else:
                param = ""
            img = eval("ski.feature.corner_" + nf + "(image" + param + ")")
            if nf == "foerstner":
                img = img[0]
            elif nf == "kitchen_rosenfeld":
                img = np.abs(img)
        images.append(img)
    return images


def detectar_picos(images, umbral):
    resultados = [images[0]]
    for i in range(1, len(images)):
        img = ski.feature.corner_peaks(images[i], indices=False, min_distance=10, threshold_rel=umbral)
        resultados.append(img)
    return resultados


def my_moravec(cimage, window_size=1):
    rows = cimage.shape[0]
    cols = cimage.shape[1]
    out = np.zeros(cimage.shape)
    for r in range(window_size + 1, rows - window_size - 1):
        for c in range(window_size + 1, cols - window_size - 1):
            min_msum = float('inf')
            for br in range(r - 1, r + 2):
                for bc in range(c - 1, c + 2):
                    if br != r or bc != c:  #### En scikit-image aquí aparece un AND !!!!
                        msum = 0
                        for mr in range(- window_size, window_size + 1):
                            for mc in range(- window_size, window_size + 1):
                                t = cimage[r + mr, c + mc] - cimage[br + mr, bc + mc]
                                msum += t * t
                        min_msum = min(msum, min_msum)

            out[r, c] = min_msum
    return out


def mostrar(titulo, resultados1, resultados2, nombres):
    fig, ax = plt.subplots(nrows=2, ncols=len(resultados1), layout="constrained")
    fig.suptitle(titulo, fontsize=24)
    for i in range(len(resultados1)):
        ax[0, i].imshow(resultados1[i], cmap='gray')
        ax[0, i].set_title(nombres[i], fontsize=16)
        ax[1, i].imshow(resultados2[i], cmap='gray')
        ax[1, i].set_title("Ruidosa" if i == 0 else nombres[i], fontsize=16)
    for a in ax.ravel():
        a.set_axis_off()
    plt.show()


filtros = ["kitchen_rosenfeld", "foerstner", "moravec", "harris", "fast"]
UMBRAL = 0.01
angulos_rotacion = [0, 22.5, 45, 67.5, 90]

for angulo in angulos_rotacion:
    imagen = ski.transform.rotate(imagen, angulo)
    images = filtrar(imagen, filtros)
    picos1 = detectar_picos(images, UMBRAL)
    mostrar(f"Esquinas detectadas (Ángulo = {angulo}º)", images, picos1, ["Original"] + filtros)
