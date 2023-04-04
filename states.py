import os, pickle, random

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


def sample_pickled_states(dir_, n_states, length, cache={}):
    # iterate over all the summary files in the directory
    summaries = {}
    total_total = 0
    if len(cache) == 0:
        for filename in os.listdir(dir_):
            if not filename.endswith('_summary.pkl'):
                continue
            with open(os.path.join(dir_, filename), 'rb') as f:
                summary = pickle.load(f)
                prefix = filename[:-len('_summary.pkl')]
                summaries[prefix] = summary
                total_total += summary['total']
        cache['summaries'] = summaries
    samples = []
    for _ in range(n_states):
        i = random.randrange(0, total_total)
        seq_len = None
        for prefix, summary in summaries.items():
            if i < summary['total']:
                seq_len = summary['seq_len']
                break
            i -= summary['total']
        # if the remaining length is less than the requested length
        # then back up enough to get the requested length
        if i + length > summary['total']:
            i -= length - (summary['total'] - i)
        state_file = os.path.join(dir_, f'{prefix}_{i // seq_len}.pkl')
        # Load the first half of the sequence
        if state_file not in cache:
            with open(state_file, 'rb') as f:
                cache[state_file] = pickle.load(f)
                sample = cache[state_file][i % seq_len:]
        if len(sample) < length:
            next_state_file = os.path.join(dir_, f'{prefix}_{i // seq_len + 1}.pkl')
            sample += cache[next_state_file][0:length - len(sample)]
        elif len(sample) > length:
            sample = sample[0:length]
        
        # Sanity test
        assert(len(sample) == length)

        samples.append(sample)
    return samples
