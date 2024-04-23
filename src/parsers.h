#include <immintrin.h>
#include <xmmintrin.h>

inline static auto const SEMI_COLONS_128_8 = _mm_set1_epi8(';');
inline static auto const SEMI_COLONS_256_8 = _mm256_set1_epi8(';');
inline static auto const NEW_LINE_256_8 = _mm256_set1_epi8('\n');
inline static auto const ZERO_ASCII_256_8 = _mm256_set1_epi8('0');
inline static auto const ZERO_256_8 = _mm256_set1_epi8(0);
inline static auto const NEG1_256_8 = _mm256_set1_epi8(-1);
inline static auto const NEGATIVE_256_8 = _mm256_set1_epi8('-');

auto read_temp_multi(char const d[32]) {
  static auto const mult_factors = _mm256_set_epi8(
      0, 100, 10, 1, 0, 100, 10, 1, 0, 100, 10, 1, 0, 100, 10, 1, 0, 100, 10, 1,
      0, 100, 10, 1, 0, 100, 10, 1, 0, 100, 10, 1);

  static auto const sign_shuffle =
      _mm256_set_epi8(15, 13, 13, 12, 11, 9, 9, 8, 7, 5, 5, 4, 3, 1, 1, 0, 15,
                      13, 13, 12, 11, 9, 9, 8, 7, 5, 5, 4, 3, 1, 1, 0);

  auto r = _mm256_load_si256((const __m256i *)d);
  auto z_diff = _mm256_sub_epi8(r, ZERO_ASCII_256_8);
  // auto diff_shifted = _mm256_srli_epi16(z_diff, 8);
  auto ltz_mask = _mm256_cmpgt_epi8(ZERO_256_8, z_diff);
  auto signed_16 = _mm256_shuffle_epi8(ltz_mask, sign_shuffle);
  auto num_only = _mm256_blendv_epi8(r, ZERO_256_8, ltz_mask);
  auto temp_split_unsigned_16 = _mm256_maddubs_epi16(num_only, mult_factors);
  // _mm256_dpwssds_epi32

  return d[0];
}
