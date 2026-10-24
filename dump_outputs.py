"""Print text outputs of the executed notebook (for review); flags errors/tracebacks/stderr and counts images.

Usage (from this folder):
    ../.venv/bin/python dump_outputs.py                                  # default notebook
    ../.venv/bin/python dump_outputs.py some.ipynb 3000 --save-images DIR  # max chars per output, dump PNGs
"""
import base64
import sys
from pathlib import Path

import nbformat

args = [a for a in sys.argv[1:]]
img_dir = None
if '--save-images' in args:
    i = args.index('--save-images')
    img_dir = Path(args[i + 1])
    img_dir.mkdir(parents=True, exist_ok=True)
    del args[i:i + 2]
nb_path = args[0] if args else 'dz_topic_12_Джафарлі_Фарід.ipynb'
MAX_TEXT = int(args[1]) if len(args) > 1 else 3000

nb = nbformat.read(nb_path, as_version=4)
n_err = n_img = n_stderr = img_bytes = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    print(f'\n===== cell {i} =====')
    print(c.source.splitlines()[0][:100] if c.source else '')
    print('-----')
    for o in c.get('outputs', []):
        if o.output_type == 'error':
            n_err += 1
            print('ERROR:', o.ename, o.evalue)
        elif o.output_type == 'stream':
            text = o.text
            if o.name == 'stderr':
                n_stderr += 1
            if len(text) > MAX_TEXT:
                text = text[:MAX_TEXT // 2] + f'\n[... {len(text) - MAX_TEXT} chars ...]\n' + text[-MAX_TEXT // 2:]
            print(f'[{o.name}]', text)
        else:
            data = o.get('data', {})
            if 'image/png' in data:
                n_img += 1
                raw = base64.b64decode(data['image/png'])
                img_bytes += len(raw)
                if img_dir:
                    (img_dir / f'cell{i:02d}_{n_img:02d}.png').write_bytes(raw)
                print(f'[image/png {len(raw) / 1e3:.0f} kB]')
            elif 'text/plain' in data:
                print(data['text/plain'][:MAX_TEXT])
print(f'\nerrors: {n_err}, stderr outputs: {n_stderr}, images: {n_img} ({img_bytes / 1e6:.1f} MB)')
