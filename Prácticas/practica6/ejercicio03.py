import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

monedas1 = ski.io.imread("Prácticas/practica6/images/monedas1.png")  # Probar también con sintética
monedas2 = ski.io.imread("Prácticas/practica6/images/monedas2.png")  # Probar también con sintética
monedas3 = ski.io.imread("Prácticas/practica6/images/monedas3.png")  # Probar también con sintética

imgs_monedas = [monedas1, monedas2, monedas3]

imgs_filtradas = []
for imagen in imgs_monedas:
    imagen = ski.filters.sobel(imagen)      # magnitud
    maximo = imagen.max()
    low = maximo * 0.1  # Probar otros valores
    high = maximo * 0.2
    img_filtrada = ski.filters.apply_hysteresis_threshold(imagen, low, high)
    imgs_filtradas.append(img_filtrada)

imgs_hough_euro = []
for imagen in imgs_filtradas:
    # radius(px) -> (24.5mm / 23mm) * 50px = 46.93
    hough_euro = ski.transform.hough_circle(imagen, radius=47)
    imgs_hough_euro.append(hough_euro)


imagenes_conjunto = [imgs_monedas, imgs_hough_euro]

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
plt.show()