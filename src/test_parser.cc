#include "parsers.h"

int main(int argc, char **argv) {

  char buf[] = "-321032100210-930123-34605940-23";
  char buf2[] =
      "-321032100210-930123-34605940-23-321032100210-930123-34605940-23";

  auto test_ans = get_answer(buf);
  auto signed_res_16 = read_temp_multi(buf);
  int16_t tmp[32 / 2];
  _mm256_storeu_si256((__m256i *)tmp, signed_res_16);
  for (int i = 0; i < 32 / 4; i += 1) {
    auto sign_val = tmp[i * 2 + 1];
    assert(test_ans[i] == sign_val);
    // assert(is_same_sign(sign_val, test_ans[i]));
  }

  read_temp_multi_top(buf);

  return 0;
}

/*
         3 2 - 0 4 9 5 0
zdiff 0x0302fd0004090500

ltzms 0x0000ff0000000000
ltz16 0x0ff0000000000000

gezms 0xffff00ffffffffff
gez16 0xf00ff000fffff000
gez+1 0xf110f1010000f101
gez+2 0xf211f2020101f202

         1 2 3 0 1 2 3 -
zdiff 0x01020300010203fd

gezms 0xffffffffffffff00
gez16 0xfffff000fff00000
gez+2 0x0101f20201f20202

*/
