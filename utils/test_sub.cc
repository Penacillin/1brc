#include <cstdio>
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

int main(int argc, char **argv) {
  //   auto const d1 = _mm256_lddqu_si256((const __m256i *)(argv[1]));
  auto const inp_8 = _mm_lddqu_si128((const __m128i *)(argv[1]));
  auto const inp_16 = _mm256_cvtepu8_epi16(inp_8);
  auto const semi_colons_16 = _mm256_set1_epi16(';');

  auto const r = _mm256_sub_epi16(inp_16, semi_colons_16);

  short arr[64];
  _mm256_storeu_si256((__m256i *)arr, r);

  auto const [minval0, minpos0] = minpos_index<0>(r);
  auto const [minval1, minpos1] = minpos_index<1>(r);

  printf("ans %d(%d) %d (%d)\n", minpos0, minval0, minpos1, minval1);

  for (int i = 0; i < 32; ++i) {
    printf("%c %d\n", argv[1][i], arr[i]);
  }

  return 0;
}
