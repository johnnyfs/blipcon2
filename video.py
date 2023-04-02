import numpy as np

import cv2
import pygame

class VideoWriter:
    def __init__(self, path, dims, fps=60):
        videodims = dims
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.video = cv2.VideoWriter(path, fourcc, fps, videodims, True)

    def frame_from_surface(self, surface):
        as_string = pygame.image.tostring(surface, 'RGB', False)
        as_np = np.frombuffer(as_string, dtype='uint8').reshape((surface.get_height(), surface.get_width(), 3))
        frame = cv2.cvtColor(as_np, cv2.COLOR_RGB2BGR)
        self.video.write(frame)

    def close(self):
        self.video.release()