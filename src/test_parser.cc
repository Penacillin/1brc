#include "parsers.h"

int main(int argc, char **argv) {

  char buf[] = "-321x321xx21x-93x123-346xx94x-23";

  for (int i = 0; i < 32; ++i) {
    if (buf[i] == 'x')
      buf[i] = '0';
  }

  read_temp_multi(buf);

  return 0;
}

/*
         3 2 - x 4 9 0 0
zdiff 0x0302fdd00409d0d0
ltzms 0x0000ff0000000000
sgnsh 0x00ffff0000000000

      0x000000ff00000000
      0x000000ff00000000

*/
