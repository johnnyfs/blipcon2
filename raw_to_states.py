from processing.states import load_states
import pickle

# Collect command line arguments infile, outfile
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filename', help='lz4 compressed state file')
parser.add_argument('outdir', help='output directory', default='.')
parser.add_argument('--seq_len', type=int, default=256)
args = parser.parse_args()

filepart = args.filename.split('/')[-1]
name = '.'.join(filepart.split('.')[:-1])
dir_ = args.outdir

print(f'Loading {args.filename} into {dir_}...')

total = 0
n_pairs = 0
pairs = []
for state, action in load_states(args.filename):
    total += 1
    pairs.append((state, action))
    if len(pairs) < args.seq_len:
        print(f'Loaded {len(pairs)} pairs, skipping', end='\r')
        continue
    # Save pairs as a pickle
    with open(f'{dir_}/{name}_{n_pairs}.pkl', 'wb') as f:
        pickle.dump(pairs, f)
    n_pairs += 1
    print(f'{n_pairs} pairs saved', end='\r')
    pairs = []

if len(pairs) > 0:
    with open(f'{dir_}/{name}_{n_pairs}.pkl', 'wb') as f:
        pickle.dump(pairs, f)

summary = {
    'total': total,
    'seq_len': args.seq_len,
}
print(f'summary: {summary}')
with open(f'{dir_}/{name}_summary.pkl', 'wb') as f:
    pickle.dump(summary, f)
