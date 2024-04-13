#pragma once

#include <bit>
#include <functional>
#include <memory>
#include <cstdio>
#include <utility>

#include "hash_table.hpp"

template <typename K, typename T, typename Hash = std::hash<K>,
          typename KeyEqual = std::equal_to<K>,
          typename Allocator = std::allocator<std::pair<const K, T>>>
class flat_hash_map {
public:
  using key_type = K;
  using mapped_type = T;
  using value_type = std::pair<const K, T>;
  using hasher = Hash;
  using key_equal = KeyEqual;
  using reference = value_type &;
  using const_reference = value_type const &;
  using pointer = value_type *;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

private:
  enum EntryType { EMPTY = 0, DELETED = 0b10, USED = 0b1 };
  struct entry {
    union {
      size_t hash;
      uint32_t _hash_dwords[sizeof(size_t) / sizeof(uint32_t)];
    };
    value_type mValue;
    uint32_t &top_dword() noexcept {
      if constexpr (std::endian::native == std::endian::big) {
        return _hash_dwords[0];
      } else {
        return _hash_dwords[sizeof(size_t) / sizeof(uint32_t) - 1];
      }
    }
    uint32_t const &top_dword() const noexcept {
      if constexpr (std::endian::native == std::endian::big) {
        return _hash_dwords[0];
      } else {
        return _hash_dwords[sizeof(size_t) / sizeof(uint32_t) - 1];
      }
    }
  };
  static constexpr auto HASH_OFFSET = 0;
  static constexpr auto SIZE_32_BITS_DIFF =
      (sizeof(size_t) - sizeof(uint32_t)) * 8;
  struct HashTraits {
    HashTraits(const Hash &hash = Hash(), const key_equal &equal = key_equal())
        : mHash(hash), mKeyEqual(equal){};
    static bool is_empty(const entry &e) { return e.hash == EntryType::EMPTY; }
    static bool is_populated(const entry &e) {
      // return (e.top_dword() & EntryType::USED) == EntryType::USED;
      return (e.hash & EntryType::USED) == EntryType::USED;
    }
    static bool is_deleted(const entry &e) {
      // return e.top_dword() == EntryType::DELETED;
      return e.hash == EntryType::DELETED;
    }
    static void set_empty(entry &e) { e.hash = EntryType::EMPTY; }
    static void set_populated(entry &e, size_t hash) {
      e.hash = hash << HASH_OFFSET;
      e.hash |= EntryType::USED;
      // e.top_dword() |= EntryType::USED;
    }
    // static void set_deleted(entry& e) { e.top_dword() = EntryType::DELETED; }
    static void set_deleted(entry &e) { e.hash = EntryType::DELETED; }
    static void destroy(entry &e) { e.~entry(); }
    struct EntryHash {
      EntryHash(const Hash &hash) : mHash(hash) {}
      size_t operator()(const entry &e) const { return mHash(e.mValue.first); }
      Hash mHash;
    };
    EntryHash entry_hash() const noexcept { return EntryHash{mHash}; }
    bool entry_equal(const entry &a, const key_type &k,
                     size_t hash) const noexcept {
      // if constexpr (sizeof(key_type) <= sizeof(size_t) &&
      //               std::is_trivial_v<key_type>)
      //   return key_eq()(a.mValue.first, k);
      // if (a.hash == ((hash << HASH_OFFSET) | EntryType::USED) &&
      //     !key_eq()(a.mValue.first, k)) [[unlikely]] {
      //       fprintf(stderr, "excuse me\n");
      // }
      return a.hash == ((hash << HASH_OFFSET) | EntryType::USED) &&
             key_eq()(a.mValue.first, k);
      // return a.hash ==
      //            (hash | ((size_t)EntryType::USED << SIZE_32_BITS_DIFF)) &&
      //        key_eq()(a.mValue.first, k);
    }
    Hash hash_function() const noexcept { return mHash; }
    key_equal key_eq() const noexcept { return mKeyEqual; }

    [[no_unique_address]] Hash mHash;
    [[no_unique_address]] key_equal mKeyEqual;
  };
  static_assert(std::is_empty_v<HashTraits> && !std::is_final_v<HashTraits>);
  using HashTable = hash_table<entry, HashTraits>;

  template <typename BaseIterT, typename ValueT> struct iterator_impl {
  private:
    BaseIterT mIter;
    friend flat_hash_map;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<const K, T>;
    using difference_type = ptrdiff_t;
    using size_type = size_t;
    using reference = ValueT &;
    using pointer = ValueT *;

    iterator_impl(BaseIterT it) : mIter(it){};

    reference operator*() const { return mIter->mValue; }
    pointer operator->() const { return &mIter->mValue; }

    iterator_impl &operator++() {
      ++mIter;
      return *this;
    }
    iterator_impl operator++(int) {
      iterator_impl tmp = *this;
      ++mIter;
      return tmp;
    }
    bool operator==(const iterator_impl &other) const {
      return mIter == other.mIter;
    }
    bool operator!=(const iterator_impl &other) const {
      return mIter != other.mIter;
    }
  };

public:
  // dense_hash_map() = default;
  explicit flat_hash_map(size_t initial_buckets = 0, const Hash &hash = Hash(),
                         const key_equal &equal = key_equal())
      : mTable(initial_buckets, HashTraits{hash, equal}){};

  using iterator = iterator_impl<typename HashTable::iterator, value_type>;
  using const_iterator =
      iterator_impl<typename HashTable::const_iterator, const value_type>;

  iterator begin() { return mTable.begin(); }
  iterator end() { return mTable.end(); }

  auto empty() const { return mTable.empty(); }
  auto size() const { return mTable.size(); }
  auto bucket_count() const { return mTable.bucket_count(); }

  void rehash(size_t num_slots) { mTable.rehash(num_slots); }
  void reserve(size_t num_items) { mTable.reserve(num_items); }

  hasher hash_function() const { return mTable.hash_traits().hash_function(); }
  key_equal key_eq() const { return mTable.hash_traits().key_eq(); }

  std::pair<iterator, bool> insert(const value_type &x) {
    return insert_impl(x);
  }
  std::pair<iterator, bool> insert(value_type &&x) {
    return insert_impl(std::move(x));
  }

  template <typename P> std::pair<iterator, bool> insert(P &&value) {
    return insert_impl(std::forward<P>(value));
  }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args &&...args) {
    return insert(value_type(std::forward<Args>(args)...));
  }

  iterator find(key_type const &key) {
    return mTable.find(key, hash_function()(key));
  }
  const_iterator find(key_type const &key) const {
    return mTable.find(key, hash_function()(key));
  }
  bool contains(key_type const &key) {
    return mTable.contains(key, hash_function()(key));
  }

  template <typename P> mapped_type &operator[](P &&key) {
    return insert_impl(std::make_pair(std::forward<P>(key), mapped_type{}))
        .first->second;
  }

  size_t erase(key_type const &key) {
    auto it = find(key);
    if (it != end()) {
      mTable.erase_no_rv(it.mIter);
      return 1;
    } else {
      return 0;
    }
  }
  iterator erase(iterator pos) { return mTable.erase(pos.mIter); }

  void clear() noexcept { mTable.clear(); }

private:
  template <typename P> std::pair<iterator, bool> insert_impl(P &&value) {
    size_t hash = hash_function()(value.first);
    auto rv = mTable.find_or_add(value.first, hash);
    if (rv.second) {
      new (&rv.first->mValue) value_type(std::forward<P>(value));
      HashTraits::set_populated(*rv.first, hash);
    } else {
      assert(HashTraits::is_populated(*rv.first));
    }
    return rv;
  }

  HashTable mTable;
};
static_assert(sizeof(flat_hash_map<int, int>) == 40);
