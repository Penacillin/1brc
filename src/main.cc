#include "hash_map.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <map>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

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

static constexpr auto BATCH_SIZE = 64;

// #define DO_FULL_FLOAT_PARSE

struct StringHasher {
  constexpr size_t operator()(std::string_view const &sx) const {
    const char *s = sx.data();
    int64_t i = 0;
    int64_t sz = sx.size();

    size_t hash = 0;

    while (i + sizeof(size_t) < sz) {
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

constexpr TempT read_temp(char const *data, char const **data_end) noexcept {
  int isNeg = 1;
  if (data[0] == '-') {
    ++data;
    isNeg = -1;
  }
  TempT v = 0;
  while (*data != '.') {
    v = (v * 10) + (data[0] - '0');
    ++data;
  }
  ++data;
  v *= 10;

  v += (data[0] - '0');

  // if (data_end != nullptr)
  *data_end = data + 1;

  return v * isNeg;
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
    for (int i = 0; i < sz; ++i) {
      mMin = std::min(mMin, vals[i]);
    }
    for (int i = 0; i < sz; ++i) {
      mMax = std::max(mMax, vals[i]);
    }
    for (int i = 0; i < sz; ++i) {
      mSum += vals[i];
    }
    mCount += sz;
  }
};

struct BatchMetrics {
  std::array<int, BATCH_SIZE> mMetricsIndices;
  std::array<TempT, BATCH_SIZE> mMin;
  std::array<TempT, BATCH_SIZE> mMax;
  std::array<int, BATCH_SIZE> mSum = {0};
  std::array<int, BATCH_SIZE> mCount = {0};

  BatchMetrics() { clear(); }

  void clear() {
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mMin[i] = 999;
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mMax[i] = -999;
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mSum[i] = 0;
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mCount[i] = 0;
    }
  }

  BatchMetrics &operator+=(const BatchMetrics &rhs) noexcept {
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mMin[i] = std::min(mMin[i], rhs.mMin[i]);
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mMax[i] = std::max(mMax[i], rhs.mMax[i]);
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mSum[i] += rhs.mSum[i];
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mCount[i] += rhs.mCount[i];
    }
    return *this;
  }

  BatchMetrics &operator+=(std::array<TempT, BATCH_SIZE> const &vals) noexcept {
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mMin[i] = std::min(mMin[i], vals[i]);
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mMax[i] = std::max(mMax[i], vals[i]);
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mSum[i] += vals[i];
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
      mCount[i] += 1;
    }
    return *this;
  }

  void update_single(int metrics_index, TempT val) noexcept {
    assert(metrics_index >= 0 && metrics_index < BATCH_SIZE);
    mMin[metrics_index] = std::min(mMin[metrics_index], val);
    mMax[metrics_index] = std::max(mMax[metrics_index], val);
    mSum[metrics_index] += val;
    mCount[metrics_index] += 1;
  }

  void set_batch(size_t i, int metrics_index, Metrics const &metrics) noexcept {
    assert(i >= 0 && i < BATCH_SIZE);
    mMetricsIndices[i] = metrics_index;
    mMin[i] = metrics.mMin;
    mMax[i] = metrics.mMax;
    mSum[i] = metrics.mSum;
    mCount[i] = metrics.mCount;
  }

  void extract_batch(size_t bi, Metrics *metrics) const noexcept {
    auto &m = *metrics;
    m.mMin = mMin[bi];
    m.mMax = mMax[bi];
    m.mSum = mSum[bi];
    m.mCount = mCount[bi];
  }
};

static_assert(sizeof(Metrics) == 12, "lmao");
using WorkerOutput = flat_hash_map<std::string_view, Metrics>;
using WorkerOutput2 = std::vector<std::pair<std::string_view, Metrics>>;

struct CityMetrics {
  std::string_view mName;
  Metrics mMetric;
};

void worker3(int core_id, char const *data, char const *const data_end,
             bool forward, WorkerOutput *output) {
  set_affinity(core_id);
  auto t0 = std::chrono::high_resolution_clock::now();

  if (forward) {
    while (*data != '\n')
      ++data;
    ++data;
  }

  int total_temps = 0;
  static constexpr int TEMP_VEC_BATCH = 4096 * 2;
  flat_hash_map<std::string_view, std::vector<TempT>> city_indices;
  city_indices.reserve(2000);

  for (; data < data_end; ++data) {
    auto *curr_start = data;
    while (*(++data) != ';')
      ;
    std::string_view s{curr_start, (size_t)(data - curr_start)};
    auto const val = read_temp(++data, &data);

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

    ++total_temps;
  }

  for (auto const &[cityS, temps] : city_indices) {
    auto &outEntry = (*output)[cityS];
    outEntry.update_batch(temps, temps.size());
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "%2d finished (%ld)\n", core_id, (t1 - t0).count());
}

void worker2(int core_id, char const *data, char const *const data_end,
             bool forward, WorkerOutput2 *output) {
  set_affinity(core_id);
  auto t0 = std::chrono::high_resolution_clock::now();

  auto &all_metrics = *output;
  flat_hash_map<std::string_view, int> city_indices;

  city_indices.reserve(2000);
  output->reserve(2000);

  std::array<TempT, BATCH_SIZE> batch_vals;
  BatchMetrics batch_metrics;

  if (forward) {
    while (*data != '\n')
      ++data;
    ++data;
  }

  int bi;
  while (true) {
    for (bi = 0; bi < BATCH_SIZE;) {
      auto *curr_start = data;
      while (*(++data) != ';')
        ;

      std::string_view s{curr_start, (size_t)(data - curr_start)};
      auto const val = read_temp(++data, &data);
      batch_vals[bi] = val;

      auto [it, inserted] = city_indices.insert({s, all_metrics.size()});
      if (inserted) [[unlikely]] {
        all_metrics.emplace_back(s, Metrics());
      }
      auto const metrics_index = it->second;

      for (int existing_mi = 0; existing_mi < bi; ++existing_mi) {
        if (batch_metrics.mMetricsIndices[existing_mi] == metrics_index)
            [[unlikely]] {
          batch_metrics.update_single(existing_mi, val);
          goto batch_updater;
        }
      }
      {
        auto &entry = all_metrics[it->second];
        batch_metrics.set_batch(bi, metrics_index, entry.second);
        ++bi;
      }

    batch_updater:

      if (++data >= data_end) [[unlikely]] {
        batch_metrics += batch_vals;
        goto finish_all;
      }
    }
    batch_metrics += batch_vals;
    assert(bi == BATCH_SIZE);
    for (int bi_extract = 0; bi_extract < BATCH_SIZE; ++bi_extract) {
      auto &batch_metric =
          all_metrics[batch_metrics.mMetricsIndices[bi_extract]];
      batch_metrics.extract_batch(bi_extract, &batch_metric.second);
    }
  }

finish_all:
  for (int bi_extract = 0; bi_extract < bi + 1; ++bi_extract) {
    auto &batch_metric = all_metrics[batch_metrics.mMetricsIndices[bi_extract]];
    batch_metrics.extract_batch(bi_extract, &batch_metric.second);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "%2d finished (%ld)\n", core_id, (t1 - t0).count());
}

void worker(int core_id, char const *data, char const *const data_end,
            bool forward, WorkerOutput *output) {
  set_affinity(core_id);
  auto t0 = std::chrono::high_resolution_clock::now();

  output->reserve(400);

  if (forward) {
    while (*data != '\n')
      ++data;
    ++data;
  }

#ifndef NDEBUG
  char const *const orig_start = data;
  {
    size_t thread_first_size = 0;
    while (data[++thread_first_size] != ';')
      ;
    std::string_view s{data, thread_first_size};
    fprintf(stderr, "%d (%p, %p) start: %s\n", core_id, data, data_end,
            std::string(s).c_str());
  }
#endif

  std::string_view s{nullptr, 0};
  for (; data < data_end; ++data) {

    // [[maybe_unused]] auto iter_t0 = __rdtsc();

    auto *curr_start = data;
    while (*(++data) != ';')
      ;

    s = {curr_start, (size_t)(data - curr_start)};

#ifndef NDEBUG
    // for (int k = 0; k < s.size(); ++k)
    //   putchar(s[k]);
#endif

    ++data;

    curr_start = data;
#ifndef DO_FULL_FLOAT_PARSE
    char const *end;
    auto val = read_temp(curr_start, &end);
    data = end;
#else
    char *end;
    float val = std::strtof(curr_start, &end) * 10.;
    data = end;
#endif

    // [[maybe_unused]] auto iter_t1 = __rdtsc();

#ifndef NDEBUG
    // printf(": %f\n", val);
#endif
    auto &entry = (*output)[s];
    entry += val;

    // [[maybe_unused]] auto iter_t2 = __rdtsc();
#ifdef PERF
    read_time += iter_t1 - iter_t0;
    sum_time += iter_t2 - iter_t1;
    ++iters;
#endif
  }

#ifndef NDEBUG
  if (true) {
    fprintf(stderr, "%d (%p, %p) end: %s\n", core_id, orig_start, data,
            std::string(s).c_str());
  } else
#endif
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%2d finished (%ld)\n", core_id, (t1 - t0).count());
  }
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
  if (data == MAP_FAILED) {
    perror("error mmapping file");
    exit(EXIT_FAILURE);
  }

  std::vector<WorkerOutput> outputs;
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

    workers.push_back(std::thread(worker3, std::atoi(argv[2 + i]),
                                  reinterpret_cast<char const *>(data) + start,
                                  data + end, true, &outputs[i]));
  }

  {
    worker3(std::atoi(argv[2]), reinterpret_cast<char const *>(data),
            data + chunk_size, false, &outputs[0]);
  }

  for (int i = 0; i < workers.size(); ++i) {
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
    for (int k = 0; k < kv.first.size(); ++k)
      putchar(kv.first[k]);

    printf(": <%.1f/%.1f/%.1f>\n", (float)kv.second.mMin / 10.,
           (float)kv.second.mSum / kv.second.mCount / 10.,
           (float)kv.second.mMax / 10.);
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
