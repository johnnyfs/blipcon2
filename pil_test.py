import random
import PIL

def create_image():
    image = PIL.Image.new('RGB', (256, 240))
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
    return image