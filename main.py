import sounddevice as sd
import numpy as np
import matplotlib
import pyformulas as pf
import matplotlib.pyplot as plt
import time

# START = time.time()

def update_img(indata, outdata, frames, tim, status):
    volume_norm = np.linalg.norm(indata)*10

    t = time.time() - 0
    # import pudb; pudb.set_trace()
    x = np.linspace(t-3, t, 10)
    # y = np.sin(2*np.pi*x) + np.sin(3*np.pi*x)
    y = np.ones(10)*volume_norm
    plt.xlim(t-3,t)
    plt.ylim(-3,3)
    plt.plot(x, y, c='black')

    # If we haven't already shown or saved the plot, then we need to draw the figure first...
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    screen.update(image)


# def print_sound(indata, outdata, frames, tim, status):
#     volume_norm = np.linalg.norm(indata)*10
#     print ("|" * int(volume_norm))


with sd.Stream(callback=update_img):
    fig = plt.figure()

    screen = pf.screen(title='Plot')


    sd.sleep(10000)
