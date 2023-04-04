import torch
from torchvision import transforms as T

import pygame

import lz4.frame
import PIL

to_tensor = T.ToTensor()
to_image = T.ToPILImage()

# Load an lz4 compressed state file
def load_states(filename):
    with lz4.frame.open(filename, 'r') as f:
        # The first full line is the image state as text
        prev_gdstr = None
        prev_buttons = None
        s = 0
        while True:
            header = f.read(11)
            if header is None:
                return
            gdstr = f.read(256 * 240 * 4)
            if gdstr is None or len(gdstr) != 256 * 240 * 4:
                return
            buttons = f.read(8)
            if buttons is None or len(buttons) != 8:
                return

            # Poor man's compression
            if gdstr == prev_gdstr and buttons == prev_buttons:
                yield None

            # Images are stored as bytes ARGB, where A is always 0
            PIL.Image.frombytes
            img = PIL.Image.new('RGB', (256, 240))
            i = 0
            for y in range(240):
                for x in range(256):
                    img.putpixel((x, y), (gdstr[i + 1], gdstr[i + 2], gdstr[i + 3]))
                    i += 4
            state = to_tensor(img)
            s += 1

            # Action is button states as text "ABsSUDLR"
            # (where s = select and S = start)
            action = torch.tensor([float(x) for x in buttons])

            yield (state, action)


def pil_to_surface(pil):
    return pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode).convert()
