#include "hash_map.hpp"
#include <bits/align.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <map>
#include <memory>
#include <smmintrin.h>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>
#include <xmmintrin.h>

void set_affinity(int core) {
  pthread_t thread = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))
    throw std::runtime_error("unable to set thread");
}

void print_sv(std::string_view s, FILE *__stream) {
  for (auto c : s) {
    fputc(c, __stream);
  }
}

template <typename _Tp> struct free_deleter {
public:
  /// Default constructor
  constexpr free_deleter() noexcept = default;

  /** @brief Converting constructor.
   *
   * Allows conversion from a deleter for arrays of another type, such as
   * a const-qualified version of `_Tp`.
   *
   * Conversions from types derived from `_Tp` are not allowed because
   * it is undefined to `delete[]` an array of derived types through a
   * pointer to the base type.
   */
  template <typename _Up>
    requires(std::is_convertible_v<_Up (*)[], _Tp (*)[]>)
  constexpr free_deleter(const free_deleter<_Tp[]> &) noexcept

  {}
  /// Calls `delete[] __ptr`
  template <typename _Up>
  constexpr typename std::enable_if<
      std::is_convertible<_Up (*)[], _Tp (*)[]>::value>::type
  operator()(_Up *__ptr) const {
    static_assert(sizeof(_Tp) > 0, "can't delete pointer to incomplete type");
    std::free(__ptr);
  }
};

// #define DO_FULL_FLOAT_PARSE

struct StringHasher {
  constexpr size_t operator()(std::string_view const &sx) const {
    const char *s = sx.data();
    int64_t i = 0;
    int64_t sz = sx.size();

    size_t hash = 0;

    while (i + (int)sizeof(size_t) < sz) {
      // printf("%ld %ld\n", i, sz - sizeof(size_t));
      hash += *((size_t *)(s + i));
      hash += (hash << 10);
      hash ^= (hash >> 6);
      i += sizeof(size_t);
    }
    while (i < sz) {
      hash += *(s + i);
      hash += (hash << 10);
      hash ^= (hash >> 6);
      ++i;
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return hash;
  }
};

using TempT = int16_t;

template <typename T> struct stream_iterator {
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = ptrdiff_t;
  using size_type = size_t;
  using reference = T const &;
  using pointer = T const *;
  static_assert(sizeof(T) == 1, "byte iteration");
  static constexpr size_t STREAM_SIZE = 16;
  static constexpr size_t BATCH_SIZE = 8;
  static constexpr size_t LOCAL_CACHE_SIZE = STREAM_SIZE * BATCH_SIZE;

  stream_iterator(T const *src, T const *end) : mSrc(src), mEnd(end) {
    assert(((uint64_t)mSrc % STREAM_SIZE) == 0);
    assert(mEnd >= mSrc);
    refresh_cache();
  }

  reference operator*() const { return local_cache[cache_offset()]; }
  pointer operator->() const { return &local_cache[cache_offset()]; }

  stream_iterator &operator++() {
    check_and_refresh(1);
    ++mSrc;
    return *this;
  }

  void refresh_cache() {
    assert(cache_offset() + bytes_left_in_cache() <= LOCAL_CACHE_SIZE);
    std::memmove(local_cache, local_cache + cache_offset(),
                 bytes_left_in_cache());
    auto const *ahead_src = mSrc + bytes_left_in_cache();
    assert((uint64_t)ahead_src % LOCAL_CACHE_SIZE == 0);
    int i = 0;
    while (i < BATCH_SIZE && mSrc < mEnd) {
      _mm_store_si128((__m128i *)(local_cache + i * STREAM_SIZE),
                      _mm_stream_load_si128((__m128i *)(ahead_src)));
      ahead_src += STREAM_SIZE;
    }
  }

  void check_and_refresh(int bytes) {
    if (bytes_left_in_cache() < bytes) [[unlikely]]
      refresh_cache();
  }

  constexpr auto bytes_left_in_cache() const noexcept {
    return LOCAL_CACHE_SIZE - ((uint64_t)mSrc % LOCAL_CACHE_SIZE);
  }

  constexpr auto cache_offset() const noexcept {
    return (uint64_t)mSrc % LOCAL_CACHE_SIZE;
  }

  alignas(STREAM_SIZE) T local_cache[LOCAL_CACHE_SIZE];
  T const *mSrc = nullptr;
  T const *mEnd = nullptr;
};

template <typename CharT>
constexpr TempT read_temp2(stream_iterator<CharT> &data) noexcept {
  int isNeg = 1;
  if (*data == '-') {
    ++data;
    isNeg = -1;
  }
  TempT v = 0;
  while (*data != '.') {
    v = (v * 10) + ((*data) - '0');
    ++data;
  }
  ++data;
  v *= 10;

  v += ((*data) - '0');

  // if (data_end != nullptr)
  ++data;

  return v * isNeg;
}

constexpr TempT read_temp(char const *data, char const **data_end) noexcept {
  int isNeg = 1;
  if (data[0] == '-') {
    ++data;
    isNeg = -1;
  }
  // TempT v = 0;

  if (data[1] == '.') {
    *data_end = data + 3;
    return ((data[0] * 10) + data[2] - ('0' * (10 + 1))) * isNeg;
  }
  *data_end = data + 4;
  return ((data[0] * 100) + data[1] * 10 + data[3] - ('0' * (100 + 10 + 1))) *
         isNeg;
  // while (*data != '.') {
  //   v = (v * 10) + ((*data) - '0');
  //   ++data;
  // }
  // ++data;
  // v *= 10;

  // v += ((*data) - '0');

  // // if (data_end != nullptr)
  // *data_end = data + 1;

  // return v * isNeg;
}

// static_assert(read_temp("5.1", nullptr) == 51);
// static_assert(read_temp("-5.1", nullptr) == -51);
// static_assert(read_temp("51.3", nullptr) == 513);
// static_assert(read_temp("-51.1", nullptr) == -511);

struct Metrics {
  TempT mMin = 999;
  TempT mMax = -999;
  int mSum = 0.;
  int mCount = 0;

  Metrics &operator+=(const Metrics &rhs) noexcept {
    mMin = std::min(rhs.mMin, mMin);
    mMax = std::max(rhs.mMax, mMax);
    mSum += rhs.mSum;
    mCount += rhs.mCount;
    return *this;
  }

  Metrics &operator+=(TempT val) noexcept {
    mMin = std::min(mMin, val);
    mMax = std::max(mMax, val);
    mSum += val;
    mCount += 1;
    return *this;
  }

  void update_batch(std::vector<TempT> const &vals, size_t sz) noexcept {
    for (unsigned i = 0; i < sz; ++i) {
      mMin = std::min(mMin, vals[i]);
    }
    for (unsigned i = 0; i < sz; ++i) {
      mMax = std::max(mMax, vals[i]);
    }
    for (unsigned i = 0; i < sz; ++i) {
      mSum += vals[i];
    }
    mCount += sz;
  }

  void update_batch(TempT const *vals_in, size_t sz) noexcept {
    TempT const *vals = std::assume_aligned<32>(vals_in);
    for (unsigned i = 0; i < sz; ++i) {
      mMin = std::min(mMin, vals[i]);
    }
    for (unsigned i = 0; i < sz; ++i) {
      mMax = std::max(mMax, vals[i]);
    }
    for (unsigned i = 0; i < sz; ++i) {
      mSum += vals[i];
    }
    mCount += sz;
  }
};

static_assert(sizeof(Metrics) == 12, "lmao");
using WorkerOutput = flat_hash_map<std::string_view, Metrics>;
using WorkerOutput2 = std::vector<std::pair<std::string_view, Metrics>>;

struct LmaoEqual {
  constexpr bool operator()(std::string_view lhs,
                            std::string_view rhs) const noexcept {
    if (lhs.size() != rhs.size())
      return false;
    for (unsigned i = 0; i < lhs.size(); ++i) {
      if (lhs[i] != rhs[i])
        return false;
    }
    return true;
  };
};

struct BatchTemps {
  int mOutputIndex;
  int mSize;
  std::unique_ptr<TempT[], free_deleter<TempT>> mTemps;
};

static auto const SEMI_COLONS_16 = _mm256_set1_epi16(';');
static auto const SEMI_COLONS128_16 = _mm_set1_epi16(';');
auto minvalindex_epu16(__m256i const v) noexcept {
  auto const minpos0 = _mm_minpos_epu16(_mm256_extractf128_si256(v, 0));
  auto const minpos1 = _mm_minpos_epu16(_mm256_extractf128_si256(v, 1));
  auto const minv0 = _mm_extract_epi16(minpos0, 0);
  auto const mini0 = _mm_extract_epi16(minpos0, 1);
  auto const minv1 = _mm_extract_epi16(minpos1, 0);
  auto const mini1 = _mm_extract_epi16(minpos1, 1) + (128 / 16);
  return minv0 < minv1 ? std::make_pair(minv0, mini0)
                       : std::make_pair(minv1, mini1);
}

auto minvalindex128_epu16(__m128i const v) noexcept {
  auto const minpos0 = _mm_minpos_epu16(v);
  auto const minv0 = _mm_extract_epi16(minpos0, 0);
  auto const mini0 = _mm_extract_epi16(minpos0, 1);
  return std::make_pair(minv0, mini0);
}

auto minvalindex128_epu8(__m128i const v) noexcept {
  auto const semi_colons_8 = _mm_set1_epi8(';');
  auto const r = _mm_sub_epi8(v, semi_colons_8);
  auto const [minv0, mini0] = minvalindex128_epu16(_mm_cvtepu8_epi16(r));
  auto const [minv1, mini1] =
      minvalindex128_epu16(_mm_cvtepu8_epi16(_mm_bsrli_si128(r, 8)));
  return minv0 <= minv1 ? std::make_pair(minv0, mini0)
                        : std::make_pair(minv1, mini1 + 8);
}

std::string_view read_city_name2(char const *data,
                                 char const **data_end) noexcept {

  auto *const curr_start = data;
  while (true) {
    auto const dr = _mm_lddqu_si128((const __m128i *)data);
    auto const [minVal, minIndex] = minvalindex128_epu8(dr);
    if (minVal == 0) [[likely]] {
      *data_end = data + minIndex;
      return {curr_start, (size_t)(*data_end - curr_start)};
    }
    data += 128 / 8;
  }
}

std::string_view read_city_name(char const *data,
                                char const **data_end) noexcept {

  auto *const curr_start = data;
  while (*(++data) != ';')
    ;
  *data_end = data;
  return {curr_start, (size_t)(data - curr_start)};
}

bool is_valid(std::string_view s) {
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == '\n' || s[i] == ';') {
      fprintf(stderr, "invalid string: '");
      print_sv(s, stderr);
      fprintf(stderr, "'\n");
      return false;
    }
  }
  return true;
}

void serial_processor_2(char const *data, char const *const data_end,
                        WorkerOutput2 *output) {

  static constexpr int TEMP_VEC_BATCH = 4096;
  flat_hash_map<std::string_view, BatchTemps, std::hash<std::string_view>,
                LmaoEqual>
      city_temps;
  city_temps.reserve(800);
  static_assert(sizeof(decltype(city_temps)::mapped_type) == 16, "map");

  auto data_it = stream_iterator(data, data_end);

  char const *prev_s_data = nullptr;
  for (; data < data_end; ++data) {
    auto const *string_start_p = data;
    auto s = read_city_name2(data, &data);
    assert(*data == ';');
    assert(is_valid(s));
    auto const val = read_temp(++data, &data);
    auto entryIt = city_temps.find(s);
    if (entryIt == city_temps.end()) [[unlikely]] {
      auto arr = std::unique_ptr<TempT[], free_deleter<TempT>>(
          (TempT *)std::aligned_alloc(32, sizeof(TempT) * TEMP_VEC_BATCH));
      auto [newIt, _inserted] =
          city_temps.insert({s, BatchTemps{.mOutputIndex = (int)output->size(),
                                           .mSize = 0,
                                           .mTemps = std::move(arr)}});
      output->push_back({s, {}});
      assert(_inserted);
      entryIt = newIt;
    } else if (string_start_p > prev_s_data + 64) {
      _mm_prefetch(data + 64, _MM_HINT_NTA);
      _mm_prefetch(data + 128, _MM_HINT_NTA);
      prev_s_data = string_start_p;
    }

    entryIt->second.mTemps[entryIt->second.mSize++] = val;

    if (entryIt->second.mSize == TEMP_VEC_BATCH) [[unlikely]] {
      auto &outEntry1 = (*output)[entryIt->second.mOutputIndex];
      outEntry1.second.update_batch(entryIt->second.mTemps.get(),
                                    TEMP_VEC_BATCH);
      entryIt->second.mSize = 0;
    }
  }

  for (auto const &[cityS, temps] : city_temps) {
    auto &outEntry = (*output)[temps.mOutputIndex];
    outEntry.second.update_batch(temps.mTemps.get(), temps.mSize);
  }
}

void serial_processor(char const *data, char const *const data_end,
                      WorkerOutput2 *output) {

  static constexpr int TEMP_VEC_BATCH = 512;
  flat_hash_map<std::string_view, BatchTemps, std::hash<std::string_view>,
                LmaoEqual>
      city_temps;
  city_temps.reserve(800);
  static_assert(sizeof(decltype(city_temps)::mapped_type) == 16, "map");

  char const *prev_s_data = nullptr;
  for (; data < data_end; ++data) {
    auto const *string_start_p = data;
    auto s = read_city_name(data, &data);
    assert(*data == ';');
    assert(is_valid(s));
    auto const val = read_temp(++data, &data);
    auto entryIt = city_temps.find(s);
    if (entryIt == city_temps.end()) [[unlikely]] {
      auto arr = std::unique_ptr<TempT[], free_deleter<TempT>>(
          (TempT *)std::aligned_alloc(32, sizeof(TempT) * TEMP_VEC_BATCH));
      auto [newIt, _inserted] =
          city_temps.insert({s, BatchTemps{.mOutputIndex = (int)output->size(),
                                           .mSize = 0,
                                           .mTemps = std::move(arr)}});
      output->push_back({s, {}});
      assert(_inserted);
      entryIt = newIt;
    } else if (string_start_p > prev_s_data + 64) {
      // _mm_clflushopt(string_start_p - ((uint64_t)string_start_p % 64));
      // _mm_clflushopt(string_start_p - 64);
      _mm_prefetch(data + 64, _MM_HINT_NTA);
      _mm_prefetch(data + 128, _MM_HINT_NTA);
      prev_s_data = string_start_p;
    }

    entryIt->second.mTemps[entryIt->second.mSize++] = val;

    if (entryIt->second.mSize == TEMP_VEC_BATCH) [[unlikely]] {
      auto &outEntry1 = (*output)[entryIt->second.mOutputIndex];
      outEntry1.second.update_batch(entryIt->second.mTemps.get(),
                                    TEMP_VEC_BATCH);
      entryIt->second.mSize = 0;
    }
  }

  for (auto const &[cityS, temps] : city_temps) {
    auto &outEntry = (*output)[temps.mOutputIndex];
    outEntry.second.update_batch(temps.mTemps.get(), temps.mSize);
  }
}

void serial_processor_no_batch(char const *data, char const *const data_end,
                               WorkerOutput *output) {
  for (; data < data_end; ++data) {
    auto *curr_start = data;
    while (*(++data) != ';')
      ;
    std::string_view s{curr_start, (size_t)(data - curr_start)};
    auto const val = read_temp(++data, &data);

    auto &entry = (*output)[s];
    entry += val;
  }
}

void worker4(int core_id, char const *data, char const *const data_end,
             bool forward, WorkerOutput *output) {
  set_affinity(core_id);
  auto t0 = std::chrono::high_resolution_clock::now();

  auto data_it = stream_iterator(data, data_end);

  if (forward) {
    while (*data_it != '\n')
      ++data_it;
    ++data_it;
  }

  static constexpr int TEMP_VEC_BATCH = 4096;
  flat_hash_map<std::string_view, std::vector<TempT>,
                std::hash<std::string_view>, LmaoEqual>
      city_indices;
  city_indices.reserve(800);
  output->reserve(800);

  for (; data_it.mSrc < data_end; ++data_it) {
    auto *curr_start = data_it.mSrc;
    while (*(++data_it) != ';')
      ;
    std::string_view s{curr_start, (size_t)(data_it.mSrc - curr_start)};
    auto const val = read_temp2(++data_it);

    auto entryIt = city_indices.find(s);
    if (entryIt == city_indices.end()) [[unlikely]] {
      auto [newIt, _inserted] =
          city_indices.insert({s, decltype(city_indices)::mapped_type{}});
      assert(_inserted);
      newIt->second.reserve(TEMP_VEC_BATCH);
      entryIt = newIt;
    }

    entryIt->second.push_back(val);

    if (entryIt->second.size() == TEMP_VEC_BATCH) [[unlikely]] {
      auto &outEntry = (*output)[s];
      outEntry.update_batch(entryIt->second, TEMP_VEC_BATCH);
      entryIt->second.clear();
    }
  }

  for (auto const &[cityS, temps] : city_indices) {
    auto &outEntry = (*output)[cityS];
    outEntry.update_batch(temps, temps.size());
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "%2d finished %ld (%ld)\n", core_id, city_indices.size(),
          (t1 - t0).count());
}

void worker2(int core_id, char const *data, char const *const data_end,
             bool forward, WorkerOutput2 *output) {
  set_affinity(core_id);
  auto t0 = std::chrono::high_resolution_clock::now();

  if (forward) {
    while (*data != '\n')
      ++data;
    ++data;
  }

  output->reserve(800);
  serial_processor(data, data_end, output);

  auto t1 = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "%2d finished (%ld)\n", core_id, (t1 - t0).count());
}

void worker3(int core_id, char const *data, char const *const data_end,
             bool forward, WorkerOutput *output) {
  set_affinity(core_id);
  auto t0 = std::chrono::high_resolution_clock::now();

  if (forward) {
    while (*data != '\n')
      ++data;
    ++data;
  }

  output->reserve(800);
  serial_processor_no_batch(data, data_end, output);

  auto t1 = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "%2d finished (%ld)\n", core_id, (t1 - t0).count());
}

int main(int argc, char **argv) {
  auto t00 = std::chrono::high_resolution_clock::now();

  if (argc < 3) {
    fprintf(stderr, "usage: %s <filename> <list of cores...>\n", argv[0]);
    exit(EINVAL);
  }

  char *file = argv[1];

  int fd = open(file, O_RDONLY);
  if (!fd) {
    perror("error opening file");
    exit(ENOENT);
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("error getting file size");
    exit(EXIT_FAILURE);
  }
  size_t sz = (size_t)sb.st_size;
  auto const *data = reinterpret_cast<const char *>(
      mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0));
  if (data == MAP_FAILED) [[unlikely]] {
    perror("error mmapping file");
    exit(EXIT_FAILURE);
  }
  if (((uint64_t)data & ((1 << 5) - 1)) != 0) [[unlikely]] {
    fprintf(stderr, "File is not 32B aligned\n");
    exit(EXIT_FAILURE);
  }

  std::vector<WorkerOutput2> outputs;
  std::vector<std::thread> workers;
  int const num_workers = argc - 2;
  outputs.reserve(num_workers);
  for (int i = 0; i < num_workers; ++i)
    outputs.emplace_back();

  auto const chunk_size = sz / num_workers;
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 1; i < num_workers; ++i) {
    size_t start = chunk_size * i;
    size_t end = i == num_workers - 1 ? sz : chunk_size * (i + 1);

    workers.push_back(std::thread(worker2, std::atoi(argv[2 + i]),
                                  reinterpret_cast<char const *>(data) + start,
                                  data + end, true, &outputs[i]));
  }

  {
    worker2(std::atoi(argv[2]), reinterpret_cast<char const *>(data),
            data + chunk_size, false, &outputs[0]);
  }

  for (unsigned i = 0; i < workers.size(); ++i) {
    workers[i].join();
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  std::map<std::string_view, Metrics> cities;
  for (auto &output : outputs) {
    for (auto &kv : output) {
      auto &metrics = cities[kv.first];
      metrics += kv.second;
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  for (auto const &kv : cities) {
    for (unsigned k = 0; k < kv.first.size(); ++k)
      putchar(kv.first[k]);

    printf(": <%.1f/%.1f/%.1f> %d\n", (float)kv.second.mMin / 10.,
           (float)kv.second.mSum / kv.second.mCount / 10.,
           (float)kv.second.mMax / 10., kv.second.mSum);
    // printf(": <%.1f/%.1f/%.1f>\n", std::round(kv.second.mMin) / 10.,
    //        std::round(kv.second.mSum / kv.second.mCount) / 10.,
    //        std::round(kv.second.mMax) / 10.);
  }

  auto t3 = std::chrono::high_resolution_clock::now();

  fprintf(stderr, "start: %ld\n", (t0 - t00).count());
  fprintf(stderr, "work: %ld\n", (t1 - t0).count());
  fprintf(stderr, "merge: %ld\n", (t2 - t1).count());
  fprintf(stderr, "print: %ld\n", (t3 - t2).count());
  return 0;
}
