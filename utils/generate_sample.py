import sys
import random
from .loader import solve_file
import numpy as np

filename = sys.argv[1]
num_measurements = int(sys.argv[2])

print(f"Creating {num_measurements} measuremnets", file=sys.stderr)

df = solve_file(filename)

values = np.random.uniform(-99.9, 99.9, num_measurements)
cities = []
for i in range(len(df)):
    cities.append(df.iloc[i].name)

for i in range(num_measurements):
    city_i = random.randint(0, len(cities)-1)
    r = cities[city_i]
    v = values[i]
    print(r, f"{v:.1f}", sep=';')
