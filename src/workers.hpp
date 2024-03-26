
#include <array>

static constexpr auto BATCH_SIZE = 64;

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
