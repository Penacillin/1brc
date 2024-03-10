#include "hash_map.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <fcntl.h>
#include <map>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

#ifndef NDEBUG
#include <iostream>
#endif

void set_affinity(int core) {
  pthread_t thread = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))
    throw std::runtime_error("unable to set thread");
}

struct Metrics {
  float mMin = 999.9;
  float mMax = -999.9;
  float mSum = 0.;
  int mCount = 0;

  Metrics &operator+=(const Metrics &rhs) noexcept {
    mMin = std::min(rhs.mMin, mMin);
    mMax = std::max(rhs.mMax, mMax);
    mSum += rhs.mSum;
    mCount += rhs.mCount;
    return *this;
  }

  friend Metrics operator+(const Metrics &lhs, const Metrics &rhs) noexcept {
    Metrics o;
    o.mMin = std::min(lhs.mMin, rhs.mMin);
    o.mMax = std::max(lhs.mMax, rhs.mMax);
    o.mSum = lhs.mSum + rhs.mSum;
    o.mCount = lhs.mCount + rhs.mCount;
    return o;
  }
};

using WorkerOutput = flat_hash_map<std::string_view, Metrics>;

void worker(int core_id, char const *data, char const *const end, bool forward,
            WorkerOutput *output) {
  set_affinity(core_id);

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
    fprintf(stderr, "%d (%p, %p) start: %s\n", core_id, data, end,
            std::string(s).c_str());
  }
#endif

  std::string_view s{nullptr, 0};
  for (; data < end; ++data) {
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
    while (*(++data) != '\n')
      ;

    float val = std::strtof(curr_start, nullptr) * 10.;

#ifndef NDEBUG
    // printf(": %f\n", val);
#endif
    auto &entry = (*output)[s];
    entry.mSum += val;
    entry.mMin = std::min(entry.mMin, val);
    entry.mMax = std::max(entry.mMax, val);
    entry.mCount += 1;
  }

#ifndef NDEBUG
  {
    fprintf(stderr, "%d (%p, %p) end: %s\n", core_id, orig_start, data,
            std::string(s).c_str());
  }
#endif
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: %s <filename> <list of cores...>\n", argv[0]);
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

  for (int i = 1; i < num_workers; ++i) {
    size_t start = chunk_size * i;
    size_t end = i == num_workers - 1 ? sz : chunk_size * (i + 1);

    workers.push_back(std::thread(worker, std::atoi(argv[2 + i]),
                                  reinterpret_cast<char const *>(data) + start,
                                  data + end, true, &outputs[i]));
  }

  {
    worker(std::atoi(argv[2]), reinterpret_cast<char const *>(data),
           data + chunk_size, false, &outputs[0]);
  }

  for (int i = 0; i < workers.size(); ++i) {
    workers[i].join();
  }

  std::map<std::string_view, Metrics> cities;
  for (auto &output : outputs) {
    for (auto &kv : output) {
      auto &metrics = cities[kv.first];
      metrics += kv.second;
    }
  }

  for (auto const &kv : cities) {
    for (int k = 0; k < kv.first.size(); ++k)
      putchar(kv.first[k]);

    // printf(": <%.4f/%.4f/%.4f>\n", kv.second.mMin / 10.,
    //        kv.second.mSum / kv.second.mCount / 10., kv.second.mMax / 10.);
    printf(": <%.1f/%.1f/%.1f>\n", std::round(kv.second.mMin) / 10.,
           std::round(kv.second.mSum / kv.second.mCount) / 10.,
           std::round(kv.second.mMax) / 10.);
  }

  return 0;
}
