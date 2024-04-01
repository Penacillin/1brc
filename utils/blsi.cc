#include "stdio.h"
#include <bit>

int main(int argc, char **argv) {
  short x = 0b010100;
  unsigned y = 67108864;
  signed z = -1;
  unsigned zz = std::bit_cast<unsigned>(z);
  unsigned long long zzz = (unsigned)z;
  y >>= 32;
  printf("%d %u %llu\n", z, zz, zzz);
  // printf("%b %b %b %b %d\n", x & -x, argc & -argc, -x, y);
  return 0;
}
