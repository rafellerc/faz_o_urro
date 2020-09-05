import argparse
import queue
import sys
import os
import psutil

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import time as timem
from PIL import Image
# from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

class VictoryException(Exception):
    pass


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=2000, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])


def init_game():
    img_path = os.path.join(PROJ_DIR, 'urro.jpeg')
    sound_path = os.path.join(PROJ_DIR, 'faz_o_urro.wav')
    image = Image.open(img_path)
    image.show()
    # timem.sleep(5)
    image.close()
    song = AudioSegment.from_wav(sound_path)
    play(song)
    # playsound(sound_path)
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()

def Victory():
    img_path = os.path.join(PROJ_DIR, 'urro_valeu.png')
    sound_path = os.path.join(PROJ_DIR, 'valeu_papai.wav')
    image = Image.open(img_path)
    image.show()
    song = AudioSegment.from_wav(sound_path)
    play(song)
    timem.sleep(5)
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()




def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    global start
    global THRESHOLD
    global mult
    global max_
    if timem.time() - start > 10:
        print(timem.time())
        start = timem.time()
        mult += 0.5
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        ampdata = np.linalg.norm(data)

        ampdata *= mult
        max_ = ampdata
        data = np.ones((shift, 1), dtype=np.float32)*ampdata
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    if max_ > THRESHOLD:
        ani.event_source.stop()
        Victory()
        # raise VictoryException()
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines


try:
    init_game()
    max_ = 0
    mult = 0.5
    THRESHOLD = 6
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    horline = ax.axhline(y=THRESHOLD, c='r')
    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), 0, 10))
    # ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    start = timem.time()
    with stream:
        plt.show()
except VictoryException as v:
    import pudb; pudb.set_trace()
# except Exception as e:
#     import pudb; pudb.set_trace()
#     parser.exit(type(e).__name__ + ': ' + str(e))

