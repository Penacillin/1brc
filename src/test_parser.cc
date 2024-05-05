#include "parsers.hpp"

void check_parse(char const *data, int exp_len, TempT exp) {
  char const *data_end;
  auto res = read_temp_v(data, &data_end);
  assert(data_end - data == exp_len);
  if (res != exp)
    printf("incorrect %d != %d\n", exp, res);
}

int main(int argc, char **argv) {

  char const bufa[] = "-12.3";
  check_parse(bufa, 5, -123);
  char const bufb[] = "34.5";
  check_parse(bufb, 4, 345);
  char const bufc[] = "9.8";
  check_parse(bufc, 3, 98);
  char const bufd[] = "-5.8";
  check_parse(bufd, 4, -58);

  alignas(32) char buf[] = "-321032100210-930123-34605940-23";
  char buf2[] =
      "-321032100210-930123-34605940-23-321032100210-930123-34605940-23";

  auto test_ans = get_answer(buf);
  auto signed_res_16 = read_temp_multi(buf);
  int16_t tmp[32 / 2];
  _mm256_storeu_si256((__m256i *)tmp, signed_res_16);
  for (int i = 0; i < 32 / 4; i += 1) {
    auto sign_val = tmp[i * 2 + 1];
    assert(test_ans[i] == sign_val);
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
