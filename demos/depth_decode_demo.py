import cv2
import numpy as np
from av import VideoFrame

from remote.comm_utils import encode_to_rgba, decode_to_depth


def depth_decode_demo():
    data = np.load('demos/test_data.npz', allow_pickle=True)
    while True:
        for i in range(len(data['depth'])):
            depth: np.ndarray = data['depth'][i]
            # encode to rgba for preprocessing
            rgba: np.ndarray = encode_to_rgba(depth)
            # drop the low 8 bits in mock video stream track
            rgb: np.ndarray = rgba[:, :, 1:] # we drop the r channel since it is low 8 bits
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # convert rgb to bgr
            frame: VideoFrame = VideoFrame.from_ndarray(rgb, format='bgr24')
            # reformat to yuv420p in aiortc to match mock video stream track
            frame.reformat(format='yuv420p')
            # decode back to rgb in mock receiver side
            decoded_rgb: np.ndarray = frame.to_ndarray(format='bgr24')
            decoded_rgb = cv2.cvtColor(decoded_rgb, cv2.COLOR_BGR2RGB)
            # convert back to depth
            decoded_float_arr: np.ndarray = decode_to_depth(decoded_rgb)
            # validate decoded depth
            cv2.imshow('depth', depth)
            cv2.imshow('decoded', decoded_float_arr)
            cv2.waitKey(100)


if __name__ == '__main__':
    depth_decode_demo()