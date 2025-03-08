import skimage as ski
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np


image = ski.io.imread("images/folleto.jpg")

######################################
# Selección de puntos con el ratón   #
# No nos interesa. Se puede ignorar  #
######################################
fig, axs = plt.subplots(1, 1, layout="constrained")
axs.imshow(image)
axs.set_axis_off()
descr_puntos = ["superior izquierdo", "superior derecho", "inferior derecho", "inferior izquierdo"]
puntos = []
nPunto = 0
def on_click(event):
    global nPunto
    if event.button is MouseButton.LEFT:
        x = round(event.xdata)
        y = round(event.ydata)
        print(f'Punto seleccionado: [{x}, {y}]')
        axs.plot(x, y, '.r')
        puntos.append([x,y])
        plt.show()
        nPunto += 1
        if nPunto==4:
            plt.disconnect(binding_id)
            plt.close()
        else:
            print(f"Introduzca el punto {descr_puntos[nPunto]}...")

binding_id = plt.connect('button_press_event', on_click)
print(f"Introduzca el punto {descr_puntos[nPunto]}...")
plt.show()
######################################

print("Puntos leídos:")
src = np.array(puntos)
xs = src[ : , 0]
ys = src[ : , 1]
print("X: ", xs)
print("Y:", ys)

# Calculamos máximos en X e Y
minx = np.min(xs)
maxx = np.max(xs)
miny = np.min(ys)
maxy = np.max(ys)

# Generamos los puntos finales
dst = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]
print("Puntos finales:")
dst = np.array(dst)
xs = dst[ : , 0]
ys = dst[ : , 1]
print("X: ", xs)
print("Y:", ys)

tform = ski.transform.PiecewiseAffineTransform()
tform.estimate(src, dst)
img_t = ski.transform.warp(image, inverse_map=tform.inverse)

area_seleccionada = img_t[miny:maxy, minx:maxx, :]

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(image)
axs[1].imshow(img_t)
axs[2].imshow(area_seleccionada)
axs[0].set_axis_off()
axs[1].set_axis_off()
axs[2].set_axis_off()
plt.show()