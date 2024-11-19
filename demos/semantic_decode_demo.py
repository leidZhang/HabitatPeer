import cv2
import numpy as np
from av import VideoFrame
import matplotlib.pyplot as plt

from remote.comm_utils import encode_to_rgba, decode_to_semantic


if __name__ == '__main__':
    data = np.load('demos/test_data.npz', allow_pickle=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    semantic0 = np.zeros((480, 640, 1), dtype=np.int32) # data['semantic'][0]
    im0 = axs[0].imshow(semantic0, cmap='jet', vmin=0, vmax=800)
    axs[0].set_title('Semantic Original')
    fig.colorbar(im0, ax=axs[0])

    semantic1 = np.zeros((480, 640, 1), dtype=np.int32) # data['semantic'][0]
    im1 = axs[1].imshow(semantic1, cmap='jet', vmin=0, vmax=800)
    axs[1].set_title('Semantic Decoded')
    fig.colorbar(im1, ax=axs[1])

    plt.ion()
    while True:
        for i in range(len(data['semantic'])):
            semantic0 = data['semantic'][i]
            # encode to rgba for preprocessing
            rgba: np.ndarray = encode_to_rgba(semantic0)
            # drop the high 8 bits in mock video stream track
            rgb: np.ndarray = rgba[:, :, :3] # drop alpha channel since it is hight 8 bits
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # convert rgb to bgr
            frame: VideoFrame = VideoFrame.from_ndarray(rgb, format='bgr24')
            # reformat to yuv420p in aiortc to match mock video stream track
            frame.reformat(format='yuv420p')
            # decode back to rgb in mock receiver side
            decoded_rgb: np.ndarray = frame.to_ndarray(format='bgr24')
            decoded_rgb = cv2.cvtColor(decoded_rgb, cv2.COLOR_BGR2RGB)
            # convert back to senmantic
            semantic1: np.ndarray = decode_to_semantic(decoded_rgb)
            # validate decoded senmantic
            im0.set_data(semantic0)
            im1.set_data(semantic1)
            plt.pause(0.1)

    plt.ioff()
    plt.show()