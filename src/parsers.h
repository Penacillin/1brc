#include <immintrin.h>
#include <xmmintrin.h>

inline static auto const SEMI_COLONS_128_8 = _mm_set1_epi8(';');
inline static auto const SEMI_COLONS_256_8 = _mm256_set1_epi8(';');
inline static auto const NEW_LINE_256_8 = _mm256_set1_epi8('\n');
inline static auto const ZERO_ASCII_256_8 = _mm256_set1_epi8('0');
inline static auto const ZERO_256_8 = _mm256_set1_epi8(0);
inline static auto const ZERO_256_16 = _mm256_set1_epi16(0);
inline static auto const NEG1_256_8 = _mm256_set1_epi8(-1);
inline static auto const ONE_256_8 = _mm256_set1_epi8(1);
inline static auto const NEGATIVE_256_8 = _mm256_set1_epi8('-');

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

inline auto read_temp_multi(char const d[32]) {
  static auto const mult_factors = _mm256_set_epi8(
      1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0,
      1, 10, 100, 0, 1, 10, 100, 0, 1, 10, 100, 0);

  static auto const sign_shuffle =
      _mm256_set_epi8(13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0, 13,
                      12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0);

  auto test_ans = get_answer(d);
  auto r = _mm256_loadu_si256((const __m256i *)d);
  auto z_diff = _mm256_sub_epi8(r, ZERO_ASCII_256_8);
  // auto diff_shifted = _mm256_srli_epi16(z_diff, 8);
  auto ltz_mask = _mm256_cmpgt_epi8(ZERO_256_8, z_diff);
  auto gez_mask = _mm256_cmpgt_epi8(z_diff, NEG1_256_8);

  auto num_only = _mm256_blendv_epi8(z_diff, ZERO_256_8, ltz_mask);

  // auto neg_nz_16_lower = _mm256_shuffle_epi8(ltz_mask, sign_shuffle);n
  auto neg_nz_16_lower = _mm256_slli_epi32(ltz_mask, 16);
  auto neg_zero_16_lower = _mm256_slli_epi32(gez_mask, 16);
  auto neg_zero_nz_16_lower = _mm256_add_epi8(neg_zero_16_lower, ONE_256_8);
  auto neg_gtz_16_lower = _mm256_sign_epi16(neg_nz_16_lower, neg_nz_16_lower);
  auto neg_mask_16_lower = _mm256_cmpgt_epi16(neg_gtz_16_lower, ZERO_256_16);

  int16_t tmp[32 / 2];
  _mm256_storeu_si256((__m256i *)tmp, neg_zero_nz_16_lower);
  for (int i = 0; i < 32 / 4; i += 1) {
    auto sign_val = tmp[i * 2 + 1];
    printf("%d %d %x\n", test_ans[i], sign_val, sign_val);
    // assert(is_same_sign(sign_val, test_ans[i]));
  }

  auto split_unsigned_16 = _mm256_maddubs_epi16(num_only, mult_factors);

  auto upper_unsigned_16 = _mm256_slli_epi32(split_unsigned_16, 16);

  auto unsigned_result_16 =
      _mm256_add_epi16(upper_unsigned_16, split_unsigned_16);

  auto signed_res_16 = _mm256_sign_epi16(unsigned_result_16, neg_mask_16_lower);

  // _mm256_srli_epi16()
  // _mm256_dpwssds_epi32

  /*

  hadd epi16 - 2/1. 16 additions

  add epi16 - 1/0.3  16 additions
  srl epi16 - 1/0.5 16 shifts
  */

  return signed_res_16;
}
