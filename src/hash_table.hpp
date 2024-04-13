#pragma once

#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>
#include <cstdio>

static inline size_t make_power_of_two(size_t v) {
  return v <= 1 ? 2 : 1 << (sizeof(size_t) * 8 - std::countl_zero(v - 1));
}

template <typename T, typename HashTraits> class hash_table final {
public:
  using key_type = T;
  using value_type = T;
  using hasher = typename HashTraits::EntryHash;
  // using key_equal = KeyMatchFunc;
  using reference = value_type &;
  using const_reference = value_type const &;
  using pointer = value_type *;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

private:
  static constexpr size_t MIN_BUCKETS = 32;
  template <typename IterTConst> struct forward_iter_impl {
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::remove_const_t<IterTConst>;
    using difference_type = ptrdiff_t;
    using size_type = size_t;
    using reference = IterTConst &;
    using pointer = IterTConst *;

    IterTConst *mCurrent = nullptr;
    IterTConst *mEnd = nullptr;
    forward_iter_impl() = default;
    forward_iter_impl(pointer start, pointer end)
        : mCurrent(start), mEnd(end){};
    forward_iter_impl(pointer start, pointer end, bool advance)
        : mCurrent(start), mEnd(end) {
      if (advance) {
        skip();
      }
    };
    void skip() {
      while (mCurrent != mEnd && !HashTraits::is_populated(*mCurrent)) {
        ++mCurrent;
      }
    }
    reference operator*() const { return *mCurrent; }
    pointer operator->() const { return mCurrent; }

    forward_iter_impl &operator++() {
      ++mCurrent;
      skip();
      return *this;
    }

    forward_iter_impl operator++(int) {
      forward_iter_impl tmp(*this);
      ++mCurrent;
      skip();
      return tmp;
    }
    bool operator==(forward_iter_impl const &rhs) const {
      return mCurrent == rhs.mCurrent;
    }
    bool operator!=(forward_iter_impl const &rhs) const {
      return mCurrent != rhs.mCurrent;
    }
  };

public:
  explicit hash_table(size_t initial_buckets = 0,
                      const HashTraits &hashTraits = HashTraits())
      : mHashTraits(hashTraits) {
    if (initial_buckets)
      alloc_table(initial_buckets);
  }

  ~hash_table() {
    if (mSlots) {
      for (T *t = mSlots; t != mSlotsEnd; ++t) {
        if (HashTraits::is_populated(*t)) {
          HashTraits::destroy(*t);
        }
      }
    }
    ::operator delete(mSlots);
  }

  using iterator = forward_iter_impl<T>;
  using const_iterator = forward_iter_impl<T const>;

  auto size() const { return mPopulated; }
  auto empty() const { return mPopulated == 0; }
  size_t bucket_count() const { return mSlots == nullptr ? 0 : mTablemask + 1; }
  float load_factor() const {
    return static_cast<float>(size()) / bucket_count();
  }

  HashTraits hash_traits() const { return mHashTraits; }
  auto hash_function() const { return hash_traits().entry_hash(); }

  iterator begin() noexcept { return iterator(mSlots, mSlotsEnd, true); }
  iterator end() noexcept { return iterator(mSlotsEnd, mSlotsEnd); }
  const_iterator begin() const noexcept {
    return const_iterator(mSlots, mSlotsEnd);
  }
  const_iterator cbegin() const noexcept { return begin(); };
  const_iterator end() const noexcept {
    return const_iterator(mSlotsEnd, mSlotsEnd);
  }
  const_iterator cend() const noexcept { return end(); }

  void alloc_table(size_t min_buckets) {
    assert(min_buckets > 0);
    const auto newTableSize =
        std::max<size_t>(MIN_BUCKETS, make_power_of_two(min_buckets));
    assert(newTableSize % 2 == 0);
    mSlots = static_cast<T *>(::operator new(newTableSize * sizeof(T)));
    mSlotsEnd = mSlots + newTableSize;
    for (T *t = mSlots; t != mSlotsEnd; ++t)
      HashTraits::set_empty(*t);
    mTablemask = newTableSize - 1;
  }

  void rehash() { rehash(bucket_count()); }

  void rehash(size_t num_slots) {
    if (!mSlots) [[unlikely]] {
      alloc_table(num_slots);
      return;
    }
    num_slots = std::max<size_t>(size() * 2, num_slots);
    hash_table other;
    std::swap(mSlots, other.mSlots);
    std::swap(mSlotsEnd, other.mSlotsEnd);

    mErased = 0;
    [[maybe_unused]] size_t total_probes = 0;

    alloc_table(num_slots);
    for (T *t = other.mSlots; t != other.mSlotsEnd; ++t) {
      if (HashTraits::is_populated(*t)) {
        size_t hash = hash_function()(*t);
        size_t index = hash_to_index(hash);
        size_t probes = 0;
        while (!HashTraits::is_empty(mSlots[index])) {
          ++probes;
          index = advance(index, probes, hash);
        }
        new (mSlots + index) T(std::move(*t));
        assert(HashTraits::is_populated(mSlots[index]));
        total_probes += probes;
      }
    }
    return;
  }

  bool reserve(size_t num_items) {
    size_t expandLimit = bucket_count() / 2;
    if (num_items + mErased < expandLimit)
      return false;
    rehash(num_items * 4);
    return true;
  }

  template <typename K> bool contains(K const &k, size_t hash) const {
    return find_pointer(k, hash) != mSlotsEnd;
  }

  template <typename K>
  std::pair<iterator, bool> find_or_add(K const &k, size_t hash) {
    [[maybe_unused]] bool rehashed = false;
    if (!mSlots) [[unlikely]] {
      reserve(mPopulated + 1);
      rehashed = true;
    }
    size_t probes = 0;
    size_t index = hash_to_index(hash);
    size_t insert_index = static_cast<size_t>(-1);
    size_t colls = 0;

    while (true) {
      T &cur = mSlots[index];
      if (hash_traits().is_empty(cur)) {
        if (reserve(mPopulated + 1)) [[unlikely]] {
          assert(!rehashed);
          rehashed = true;
          probes = 0;
          index = hash_to_index(hash);
          insert_index = static_cast<size_t>(-1);
          continue;
        }
        if (insert_index == static_cast<size_t>(-1)) {
          insert_index = index;
        } else {
          --mErased;
        }
        ++mPopulated;

        // if (colls > 0)
        //   fprintf(stderr, "colls %ld\n", colls);

        return std::make_pair(iterator(mSlots + insert_index, mSlotsEnd), true);
      } else if (HashTraits::is_deleted(cur)) {
        if (insert_index == static_cast<size_t>(-1)) {
          insert_index = index;
        }
      } else {
        ++colls;
        if (hash_traits().entry_equal(cur, k, hash)) {
          return std::make_pair(iterator(&cur, mSlotsEnd), false);
        }
      }

      ++probes;
      index = advance(index, probes, hash);
      assert(probes < bucket_count());
    }
  };

  template <typename K> iterator find(K const &k, size_t hash) {
    return iterator(find_pointer(k, hash), mSlotsEnd);
  }
  template <typename K> const_iterator find(K const &k, size_t hash) const {
    return const_iterator(find_pointer(k, hash), mSlotsEnd);
  }

  void erase_no_rv(iterator it) {
    assert(it != end());
    assert(HashTraits::is_populated(*it));
    HashTraits::destroy(*it);
    HashTraits::set_deleted(*it);
    ++mErased;
    --mPopulated;
  }

  iterator erase(iterator it) {
    erase_no_rv(it);
    return ++it;
  }

  void clear() {
    if (mSlots) {
      for (T *t = mSlots; t != mSlotsEnd; ++t) {
        if (HashTraits::is_populated(*t)) {
          HashTraits::destroy(*t);
          HashTraits::set_empty(*t);
        }
      }
      mPopulated = 0;
      mErased = 0;
    }
  }

  void swap(hash_table &rhs) {
    std::swap(mSlots, rhs.mSlots);
    std::swap(mSlotsEnd, rhs.mSlotsEnd);
    std::swap(mTablemask, rhs.mTablemask);
    std::swap(mPopulated, rhs.mPopulated);
    std::swap(mErased, rhs.mErased);
  }

private:
  template <typename K> inline T *find_pointer(K const &k, size_t hash) const {
    if (!mSlots) [[unlikely]]
      return mSlotsEnd;
    size_t probes = 0;
    size_t index = hash_to_index(hash);
    while (true) {
      T &cur = mSlots[index];

      if (HashTraits::is_empty(cur)) {
        return mSlotsEnd;
      } else if (HashTraits::is_deleted(cur)) {
        // do nothing
      } else if (hash_traits().entry_equal(cur, k, hash)) {
        return &cur;
      }

      ++probes;
      index = advance(index, probes, hash);
      assert(probes < bucket_count());
    }
  }

  inline size_t advance(size_t index, size_t probes,
                        size_t /*hash*/) const noexcept {
    return (index + 1) & mTablemask;
  }

  inline size_t hash_to_index(size_t hash) const noexcept {
    return (hash)&mTablemask;
  }

  T *mSlots = nullptr;
  T *mSlotsEnd = nullptr;
  size_t mTablemask = 0;
  size_t mPopulated = 0;
  size_t mErased = 0;

  [[no_unique_address]] HashTraits mHashTraits;
};
namespace _internal {
struct _FakeTraits {
  using EntryHash = int;
};
static_assert(sizeof(hash_table<int, _FakeTraits>) == 40);
} // namespace _internal
