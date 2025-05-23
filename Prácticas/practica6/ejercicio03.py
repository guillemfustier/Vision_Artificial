import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

monedas1 = ski.io.imread("Prácticas/practica6/images/monedas1.png")  # Probar también con sintética
monedas2 = ski.io.imread("Prácticas/practica6/images/monedas2.png")  # Probar también con sintética
monedas3 = ski.io.imread("Prácticas/practica6/images/monedas3.png")  # Probar también con sintética

imgs_monedas = [monedas1, monedas2, monedas3]

imgs_bordes = []
for imagen in imgs_monedas:
    imagen_borde = ski.feature.canny(imagen, sigma=1.2)
    imagen_borde = ski.morphology.thin(imagen_borde)
    imgs_bordes.append(imagen_borde)
    
imgs_hough_euro = []
cy_list = []
cx_list = []
radii_list = []
for imagen in imgs_bordes:
    # radius(px) -> (24.5mm / 23mm) * 50px = 46.93
    radios = np.arange(17, 25, 2)
    hough_euro = ski.transform.hough_circle(imagen, radius=radios)
    accums, cx, cy, radii = ski.transform.hough_circle_peaks(hough_euro, radios, min_xdistance=10, min_ydistance=10,
                                                         threshold=hough_euro.max() / 2)  # El threshold no debería ser necesario. Se supone que es el valor por defecto
    cy_list.append(cy)
    cx_list.append(cx)
    radii_list.append(radii)
    imgs_hough_euro.append(hough_euro)

# Dibujar los círculos en la imagen resultado
imgs_resultado = []
for i in range (3):
    resultado = ski.color.gray2rgb(monedas1*0)
    for fila, col, radio in zip(cy_list[i], cx_list[i], radii_list[i]):
        circy, circx = ski.draw.circle_perimeter(fila, col, radio, shape=monedas1.shape)  # Dibuja un círculo
        if radio == 23:
            resultado[circy, circx] = (255, 0, 0)   # moneda de 1 euro
        else:
            resultado[circy, circx] = (0, 255, 0)       # moneda de 20 cts
    imgs_resultado.append(resultado)

# Visualizar resultados
fig, axs = plt.subplots(nrows=3, ncols=3, layout="constrained")
fig.suptitle("Ejemplo - Transformada de Hough para círculos", fontsize=24)

for i in range (3):
    axs[i,0].imshow(imgs_monedas[i], cmap='gray')
    axs[i,0].set_title("Imagen original", size=16)
    axs[i,1].imshow(imgs_bordes[i], cmap='gray')
    axs[i,1].set_title("Mapa de bordes (Canny)", size=16)
    axs[i,2].imshow(imgs_resultado[i], cmap='gray')
    axs[i,2].set_title("HT para círculos", size=16)

for a in axs.ravel():
    a.set_axis_off()
plt.show()