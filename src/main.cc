#include "hash_map.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <fcntl.h>
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

constexpr int read_temp(char const *data, char const **data_end) noexcept {
  int isNeg = 1;
  if (data[0] == '-') {
    ++data;
    isNeg = -1;
  }
  int v = 0;
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
  int mMin = 999;
  int mMax = -999;
  int mSum = 0.;
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

  Metrics &operator+=(int val) noexcept {
    mMin = std::min(mMin, val);
    mMax = std::max(mMax, val);
    mSum += val;
    mCount += 1;
    return *this;
  }
};

using WorkerOutput = flat_hash_map<std::string_view, Metrics>;


/*
t0: 1, 2, 4  (000 -> 001, 010, 100)
t1:          (001 ->)
t2: 3,       (010 -> 011)
t3:
t4: 5, 6     (100 -> 101, 110) 111
t5:
t6: 7
t7:

*/

void merge_outputs(int thread_id, int other_id) {}

void worker(int core_id, char const *data, char const *const end, bool forward,
            WorkerOutput *output) {
  set_affinity(core_id);
  auto const expected_elements = (end - data) / 16;
  output->reserve(expected_elements);

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
#ifndef DO_FULL_FLOAT_PARSE
    char const *end;
    auto val = read_temp(curr_start, &end);
    data = end;
#else
    char *end;
    float val = std::strtof(curr_start, &end) * 10.;
    data = end;
#endif

#ifndef NDEBUG
    // printf(": %f\n", val);
#endif
    auto &entry = (*output)[s];
    entry += val;
  }

#ifndef NDEBUG
  if (true) {
    fprintf(stderr, "%d (%p, %p) end: %s\n", core_id, orig_start, data,
            std::string(s).c_str());
  } else
#endif
  {
    fprintf(stderr, "%d finished\n", core_id);
  }
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

  // return 0;

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

    printf(": <%.1f/%.1f/%.1f>\n", (float)kv.second.mMin / 10.,
           (float)kv.second.mSum / kv.second.mCount / 10.,
           (float)kv.second.mMax / 10.);
    // printf(": <%.1f/%.1f/%.1f>\n", std::round(kv.second.mMin) / 10.,
    //        std::round(kv.second.mSum / kv.second.mCount) / 10.,
    //        std::round(kv.second.mMax) / 10.);
  }

  return 0;
}
