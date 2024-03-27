#include <cstdio>
#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <utility>
#include <xmmintrin.h>

template <int i> auto constexpr minpos_index(__m256i const v) noexcept {
  auto const minpos0 = _mm_minpos_epu16(_mm256_extractf128_si256(v, i));
  return std::make_pair(_mm_extract_epi16(minpos0, 0),
                        _mm_extract_epi16(minpos0, 1));
}

auto minvalindex128_epu16(__m128i const v) noexcept {
  auto const minpos0 = _mm_minpos_epu16(v);
  auto const minv0 = _mm_extract_epi16(minpos0, 0);
  auto const mini0 = _mm_extract_epi16(minpos0, 1);
  return std::make_pair(minv0, mini0);
}

auto minvalindex128_epu8(__m128i const v) noexcept {
  auto const [minv0, mini0] = minvalindex128_epu16(_mm_cvtepu8_epi16(v));
  auto const [minv1, mini1] =
      minvalindex128_epu16(_mm_cvtepu8_epi16(_mm_bsrli_si128(v, 8)));
  printf("x %d(%d)\n", mini0, minv0);
  printf("x %d(%d)\n", mini1, minv1);
  return minv0 < minv1 ? std::make_pair(minv0, mini0)
                       : std::make_pair(minv1, mini1 + 8);
}

// auto const dr_16 = _mm256_cvtepu8_epi16(dr);
// auto const dr_16 = _mm_cvtepu8_epi16(dr);
// auto const dr_16_2 = _mm_cvtepu8_epi16(_mm_bsrli_si128(dr, 16));
// auto const sub_res = _mm256_sub_epi16(dr_16, SEMI_COLONS_16);
// auto const sub_res = _mm_sub_epi16(dr_16, SEMI_COLONS128_16);
// auto const sub_res2 = _mm_sub_epi16(dr_16_2, SEMI_COLONS128_16);
// auto const [minVal, minIndex] = minvalindex_epu16(sub_res);

int main(int argc, char **argv) {
  //   auto const d1 = _mm256_lddqu_si256((const __m256i *)(argv[1]));
  auto const inp_8 = _mm_lddqu_si128((const __m128i *)(argv[1]));
  auto const semi_colons_8 = _mm_set1_epi8(';');

  auto const r = _mm_sub_epi8(inp_8, semi_colons_8);

  char arr[64];
  _mm_storeu_si128((__m128i *)arr, r);

  auto const [minval0, minpos0] = minvalindex128_epu8(r);
  // auto const [minval1, minpos1] = minpos_index<1>(r);
  // auto const [minval0, minpos0] = minpos_index<0>(r);
  // auto const [minval1, minpos1] = minpos_index<1>(r);

  // printf("ans %d(%d) %d (%d)\n", minpos0, minval0, minpos1, minval1);
  printf("ans %d(%d)\n", minpos0, minval0);

  for (int i = 0; i < 16; ++i) {
    printf("%c %d\n", argv[1][i], arr[i]);
  }

  return 0;
}
