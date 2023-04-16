import os, pickle, random

import torch
from torchvision import transforms as T

import pygame

import lz4.frame
import PIL

from collections import OrderedDict

to_tensor = T.ToTensor()
to_image = T.ToPILImage()

def from_raw_state(gdstr):
    # Images are stored as bytes ARGB, where A is always 0
    img = PIL.Image.new('RGB', (256, 240))
    i = 11 # Skip the header
    for y in range(240):
        for x in range(256):
            img.putpixel((x, y), (gdstr[i + 1], gdstr[i + 2], gdstr[i + 3]))
            i += 4
    return to_tensor(img)

# Load an lz4 compressed state file
def load_states(filename):
    with lz4.frame.open(filename, 'r') as f:
        # The first full line is the image state as text
        prev_gdstr = None
        prev_buttons = None
        s = 0
        while True:
            gdstr = f.read(256 * 240 * 4 + 11)
            if gdstr is None or len(gdstr) != 256 * 240 * 4 + 11:
                return
            buttons = f.read(8)
            if buttons is None or len(buttons) != 8:
                return

            # Poor man's compression
            if gdstr == prev_gdstr and buttons == prev_buttons:
                yield None

            state = from_raw_state(gdstr)
            s += 1

            # Action is button states as text "ABsSUDLR"
            # (where s = select and S = start)
            action = torch.tensor([float(x) for x in buttons])

            yield (state, action)


def _load_or_add(path, cache, max_size=512):
    if path in cache['files']:
        return cache['files'][path]
    with open(path, 'rb') as f:
        data = pickle.load(f)
        if len(cache['files']) > max_size - 1:
            cache['files'].popitem(last=False)
        cache['files'][path] = data
        return data


def sample_pickled_states(dir_, n_states, length, cache=None, max_cache_size=512):
    # iterate over all the summary files in the directory
    summaries = {}
    if cache is None:
        cache = {}
    if len(cache) == 0:
        for filename in os.listdir(dir_):
            if not filename.endswith('_summary.pkl'):
                continue
            with open(os.path.join(dir_, filename), 'rb') as f:
                summary = pickle.load(f)
                prefix = filename[:-len('_summary.pkl')]
                summaries[prefix] = summary
        cache['summaries'] = summaries
        cache['files'] = OrderedDict()
    else:
        summaries = cache['summaries']
    
    samples = []
    for _ in range(n_states):
        # Decide a file
        prefix = random.choice(list(summaries.keys()))
        offset = random.randrange(0, summaries[prefix]['total'] - length)
        end = offset + length
        # Load the file
        sample = []
        seq_len = summaries[prefix]['seq_len']
        for i in range(offset, end):
            file_idx = i // seq_len
            file_offset = i % seq_len
            filename = f'{prefix}_{file_idx}.pkl'
            path = os.path.join(dir_, filename)
            states = _load_or_add(path, cache, max_cache_size)   
            sample.append(states[file_offset])
        
        # Sanity test
        assert(len(sample) == length)

        samples.append(sample)
    return samples
