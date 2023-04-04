import os

import PIL.Image

dir_ = 'sample_images'
out = 'samples_downsized'
for file in os.listdir(dir_):
    img = PIL.Image.open(os.path.join(dir_, file))
    img = img.resize((256, 224), PIL.Image.NEAREST)
    img.save(os.path.join(out, file))