#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

inline static auto const SEMI_COLONS_128_8 = _mm_set1_epi8(';');
inline static auto const SEMI_COLONS_256_8 = _mm256_set1_epi8(';');
inline static auto const NEW_LINE_256_8 = _mm256_set1_epi8('\n');
inline static auto const ZERO_ASCII_256_8 = _mm256_set1_epi8('0');
inline static auto const ZERO_256_8 = _mm256_set1_epi8(0);
inline static auto const ZERO_256_16 = _mm256_set1_epi16(0);
inline static auto const NEG1_256_8 = _mm256_set1_epi8(-1);
inline static auto const ONE_256_8 = _mm256_set1_epi8(1);
inline static auto const TWO_256_8 = _mm256_set1_epi8(2);
inline static auto const NEGATIVE_256_8 = _mm256_set1_epi8('-');
inline static auto const TEMP_MULT_FACTORS =
    _mm256_set_epi8(1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0,
                    1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0);

static const inline __m64 PERIOD_64_8 = _mm_set1_pi8('.');
static const inline auto ZERO_ASCII_64_8 = _mm_set1_pi8('0');
inline static auto const TEMP_MULT_FACTORS_64 =
    _mm_set_pi8(0, 0, 0, 0, 1, 0, 10, 100);

/*
'.' = 46
'-' = 45
'0' = 48
*/

using TempT = int16_t;

#ifndef NDEBUG
#include <assert.h>
#include <stdio.h>
inline int16_t ans_ret[32 / 2];
inline auto get_answer(char const d[32]) {
  for (int i = 0; i < 32; i += 4) {
    int16_t res;
    int16_t isNeg = 1;
    if (d[i] == '-' || d[i + 1] == '-')
      isNeg = -1;
    if (d[i + 1] == '-') {
      res = d[i + 2] * 10 + d[i + 3] - ('0' * (10 + 1));
    } else {
      res = d[i + 1] * 100 + d[i + 2] * 10 + d[i + 3] - ('0' * (100 + 10 + 1));
    }
    ans_ret[i / 4] = res * isNeg;
    printf("%d,", ans_ret[i / 4]);
  }
  printf("\n");

  return ans_ret;
}
#endif

inline auto is_same_sign(auto y, auto x) { return ((y < 0) == (x < 0)); }

inline TempT read_temp_v(char const *data, char const **data_end) {
  uint64_t y;
  unsigned isNeg = data[0] == '-';
  assert(isNeg == 0 || isNeg == 1);

  std::memcpy(&y, data, sizeof(y));
  uint64_t neg_mask = _mm_cvtm64_si64(_mm_cmpeq_pi8((__m64)y, PERIOD_64_8));
  unsigned period_pos = std::countr_zero(neg_mask) / 8;
  assert(period_pos >= 1 && period_pos < 4);
  auto z_diff = _mm_sub_pi8((__m64)y, ZERO_ASCII_64_8);

  auto z_diff_shift = z_diff >> (isNeg * 8);
  unsigned temp_shift = (period_pos ^ isNeg) & 1;
  auto const temp_mult = TEMP_MULT_FACTORS_64 >> (temp_shift * 8);

  auto split_unsigned_16 = _mm_maddubs_pi16(z_diff_shift, temp_mult);
  auto split_unsigned_16_upper = split_unsigned_16 >> 16;
  split_unsigned_16 += split_unsigned_16_upper;
  split_unsigned_16 &= 0xFFFF;
  /*
          period_pos   isNeg  temp_shift
  -32.1   3             1       0
  32.1    2             0       0
  -2.1    2             1       1
  2.1     1             0       1
   */

  split_unsigned_16 = isNeg ? -split_unsigned_16 : split_unsigned_16;
  *data_end = data + period_pos + 2;
  return (TempT)(uint64_t)split_unsigned_16;
}

inline auto read_temp_multi(char const d[32]) {
  assert((int64_t)d % 32 == 0);
  auto r = _mm256_load_si256((const __m256i *)d);
  auto z_diff = _mm256_sub_epi8(r, ZERO_ASCII_256_8);
  auto ltz_mask = _mm256_cmpgt_epi8(ZERO_256_8, z_diff);

  // unsigned number calcs
  auto num_only = _mm256_blendv_epi8(z_diff, ZERO_256_8, ltz_mask);
  auto split_unsigned_16 = _mm256_maddubs_epi16(num_only, TEMP_MULT_FACTORS);
  auto upper_unsigned_16 = _mm256_slli_epi32(split_unsigned_16, 16);
  auto unsigned_result_16 =
      _mm256_add_epi16(upper_unsigned_16, split_unsigned_16);

  // sign calcs
  auto ltz_upper = _mm256_slli_epi32(ltz_mask, 24);
  auto ltz_lower = _mm256_slli_epi32(ltz_mask, 16);
  auto neg_16_lower = _mm256_or_si256(ltz_upper, ltz_lower);
  neg_16_lower = _mm256_or_si256(neg_16_lower, r);

  // int16_t tmp[32 / 2];
  // int16_t tmp2[32 / 2];
  // for (int i = 0; i < 32 / 4; i += 1) {
  //   auto sign_val = tmp[i * 2 + 1];
  //   auto sign_val2 = tmp2[i * 2 + 1];
  //   printf("%d %d %x %d\n", test_ans[i], sign_val, sign_val, sign_val2);
  //   // assert(is_same_sign(sign_val, test_ans[i]));
  // }

  auto signed_res_16 = _mm256_sign_epi16(unsigned_result_16, neg_16_lower);

  return signed_res_16;
}

inline auto read_temp_multi_top(char const d[32]) {
  static auto const mult_factors = _mm256_set_epi8(
      1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0,
      1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0);

  auto r = _mm256_loadu_si256((const __m256i *)d);
  // auto r = _mm256_shuffle_epi8(ro, reverse_dw_shuffle);
  auto z_diff = _mm256_sub_epi8(r, ZERO_ASCII_256_8);

  auto ltz_mask = _mm256_cmpgt_epi8(ZERO_256_8, z_diff);

  // numbers
  auto num_only = _mm256_blendv_epi8(z_diff, ZERO_256_8, ltz_mask);
  auto split_unsigned_16 = _mm256_maddubs_epi16(num_only, mult_factors);
  auto upper_unsigned_16 = _mm256_srli_epi32(split_unsigned_16, 16);
  auto unsigned_result_16 =
      _mm256_add_epi16(upper_unsigned_16, split_unsigned_16);

  // sign
  auto ltz_upper_right = ltz_mask;
  auto ltz_lower_right = _mm256_slli_epi32(ltz_mask, 8);
  auto neg_16_lower = _mm256_or_si256(ltz_upper_right, ltz_lower_right);
  neg_16_lower = _mm256_or_si256(neg_16_lower, r);

  // int16_t tmp[32 / 2];
  // int16_t tmp2[32 / 2];
  // for (int i = 0; i < 32 / 4; i += 1) {
  //   auto sign_val = tmp[i * 2];
  //   auto sign_val2 = tmp2[i * 2];
  //   printf("%d %d %x %d\n", test_ans[i], sign_val, sign_val, sign_val2);
  //   // assert(is_same_sign(sign_val, test_ans[i]));
  // }

  auto signed_res_16 = _mm256_sign_epi16(unsigned_result_16, neg_16_lower);

  return signed_res_16;
}

inline auto read_temp_multi_double(char const d[64]) {
  static auto const mult_factors = _mm256_set_epi8(
      1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0,
      1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0);

  // auto test_ans = get_answer(d);
  // auto test_ans_r = get_answer(d + 32);

  auto r = _mm256_loadu_si256((const __m256i *)d);
  auto rr = _mm256_loadu_si256((const __m256i *)(d + 32));

  auto z_diff = _mm256_sub_epi8(r, ZERO_ASCII_256_8);
  auto z_diff_right = _mm256_sub_epi8(rr, ZERO_ASCII_256_8);

  auto ltz_mask = _mm256_cmpgt_epi8(ZERO_256_8, z_diff);
  auto ltz_mask_r = _mm256_cmpgt_epi8(ZERO_256_8, z_diff_right);

  // numbers
  auto num_only = _mm256_blendv_epi8(z_diff, ZERO_256_8, ltz_mask);
  auto num_only_r = _mm256_blendv_epi8(z_diff_right, ZERO_256_8, ltz_mask_r);
  auto split_unsigned_16 = _mm256_maddubs_epi16(num_only, mult_factors);
  auto split_unsigned_16_r = _mm256_maddubs_epi16(num_only_r, mult_factors);

  auto upper_unsigned_16 = _mm256_slli_epi32(split_unsigned_16, 16);
  auto upper_unsigned_16_r = _mm256_srli_epi32(split_unsigned_16_r, 16);

  auto blended_uppers =
      _mm256_blend_epi16(upper_unsigned_16, upper_unsigned_16_r, 0b10101010);
  auto blended_lowers =
      _mm256_blend_epi16(split_unsigned_16_r, split_unsigned_16, 0b10101010);

  auto unsigned_result_16 = _mm256_add_epi16(blended_uppers, blended_lowers);

  // sign
  auto ltz_upper_right = ltz_mask;
  auto ltz_lower_right = _mm256_slli_epi32(ltz_mask, 8);
  auto neg_16_lower = _mm256_or_si256(ltz_upper_right, ltz_lower_right);
  neg_16_lower = _mm256_or_si256(neg_16_lower, r);

  // int16_t tmp[32 / 2];
  // int16_t tmp2[32 / 2];
  // for (int i = 0; i < 32 / 4; i += 1) {
  //   auto sign_val = tmp[i * 2];
  //   auto sign_val2 = tmp2[i * 2];
  //   printf("%d %d %x %d\n", test_ans[i], sign_val, sign_val, sign_val2);
  //   // assert(is_same_sign(sign_val, test_ans[i]));
  // }

  auto signed_res_16 = _mm256_sign_epi16(unsigned_result_16, neg_16_lower);

  // _mm256_storeu_si256((__m256i *)tmp, signed_res_16);
  // for (int i = 0; i < 32 / 4; i += 1) {
  //   auto sign_val = tmp[i * 2];
  //   printf("%d %d\n", test_ans[i], sign_val);
  //   // assert(is_same_sign(sign_val, test_ans[i]));
  // }

  return signed_res_16;
}
