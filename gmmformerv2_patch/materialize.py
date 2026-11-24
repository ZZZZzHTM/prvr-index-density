"""Write an exact upstream-to-experiment Git patch to standard output."""
from pathlib import Path
import re
import subprocess
import sys

revision = '5bacac96f77408c2bef27c25637e2cf67fb062b9'
root = Path(sys.argv[1])
template = Path(__file__).with_name('gmmformerv2.patch').read_bytes()
pattern = rb'^-<UPSTREAM_LINE (src/[A-Za-z0-9_/]+\.py) ([0-9]+)>$'

def upstream_line(match):
    path = match[1].decode()
    blob = subprocess.check_output(['git', '-C', str(root), 'show', revision + ':' + path])
    return b'-' + blob.splitlines()[int(match[2]) - 1]

sys.stdout.buffer.write(re.sub(pattern, upstream_line, template, flags=re.MULTILINE))
