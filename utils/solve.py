import sys
from .loader import solve_file

filename = sys.argv[1]

df = solve_file(filename)

for r in df.itertuples():
    min_v = round(r[1], 1)
    mean_v = round(r[2], 1)
    max_v = round(r[3], 1)
    print(f"{r.Index}: <{min_v:.1f}/{mean_v:.1f}/{max_v:.1f}>")
