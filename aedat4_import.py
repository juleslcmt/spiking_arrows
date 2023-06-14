import aedat
import scipy
from numpy.lib import recfunctions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
decoder = aedat.Decoder('movingtarget-2023_05_13_14_16_23.aedat4')
"""
decoder is a packet iterator with an additional method id_to_stream
id_to_stream returns a dictionary with the following structure:
{
    <int>: {
        "type": <str>,
    }
}cc
type is one of "events", "frame", "imus", "triggers"
if type is "events" or "frame", its parent dictionary has the following structure:
{
    "type": <str>,
    "width": <int>,
    "height": <int>,
}
"""
print(decoder.id_to_stream())

events = np.empty((0, 4), dtype=np.uint64)

for packet in decoder:
    """
    packet is a dictionary with the following structure:
    {
        "stream_id": <int>,
    }
    packet also has exactly one of the following fields:
        "events", "frame", "imus", "triggers"
    """
    if "events" in packet:
        """
        packet["events"] is a structured numpy array with the following dtype:
            [
                ("t", "<u8"),
                ("x", "<u2"),
                ("y", "<u2"),
                ("on", "?"),
            ]
        """
        events = np.vstack((events, recfunctions.structured_to_unstructured(packet['events'])))

# Make times start at 0 (ms?)
df=pd.DataFrame(events,columns=['t','x','y','on'])
df['t']-=df['t'].min()
df.set_index('t',inplace=True)
print(df.head())

# remove off events
df = df[df["on"] != 0]
# group events into frames
df=df.groupby('t')[['x','y']].apply(lambda x: list(map(tuple,x.values)))
print(df.head())

# iterate over frames
for i,frame in df[::1000].items():
    fig,axes=plt.subplots(3,1,figsize=(6.4,4.8*3))

    # plot frame
    axes[0].scatter(*zip(*frame),s=1,c='r')
    axes[0].set_xlim(0,640)
    axes[0].set_ylim(0,480)
    axes[0].set_title("Frame")

    # plot fft
    np_frame = np.zeros((480, 640), dtype=np.uint8)
    for x, y in frame:
        np_frame[y, x] = 255
    fft= scipy.fft.fft2(np_frame)
    axes[1].imshow(np.log10(np.abs(fft)), aspect="auto")
    axes[1].set_title("FFT")

    # plot reconstruction
    scaler= MinMaxScaler(feature_range=(0,255))
    ifft = scipy.fft.ifft2(fft)
    # ifft = scaler.fit_transform(np.abs(ifft))
    axes[2].imshow(np.abs(ifft), aspect="auto")
    axes[2].set_title("IFFT")

    fig.suptitle(rf"Time: {i/1e6} s")

    fig.tight_layout()
    fig.show()
