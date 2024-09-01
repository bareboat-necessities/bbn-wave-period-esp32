#ifndef STL_RingBuffer_h
#define STL_RingBuffer_h

/*
  This code is available under the MIT license:

    Copyright (c) 2016 Evan Balster

    Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>

/*
  C++ fixed-size ringbuffer container.
    Its capacity is always a power of two.
*/
namespace mono_wedge {
  namespace detail {
    template <typename T> T next_power_of_two(const T n) {
      static const T bits = 8 * sizeof(T);
      T bitfill = std::max<T>(n, 1) - 1;
      for (T shift = 1; shift < bits; shift <<= 1) bitfill |= (bitfill >> shift);
      return bitfill + 1;
    }
  }

  template <class T, class Allocator = std::allocator<T>>
  class fixed_ringbuffer {
    public:
      typedef T value_type;
      typedef T& reference;
      typedef const T& const_reference;
  
      typedef typename std::allocator_traits<Allocator>::pointer pointer;
      typedef typename std::allocator_traits<Allocator>::const_pointer const_pointer;
  
      typedef size_t size_type;
      typedef std::ptrdiff_t difference_type;
      typedef Allocator allocator_type;
  
    public:
      /*
        Random-access iterator implementation.
      */
      template <class T_ringbuffer, class T_Value>
      class _iterator {
        public:
          typedef typename T_ringbuffer::difference_type difference_type;
          typedef T_Value value_type;
          typedef T_Value& reference;
          typedef T_Value* pointer;
          typedef std::random_access_iterator_tag iterator_category;
  
        public:
          _iterator() : _ring(NULL), _idx(0) {}
          _iterator(T_ringbuffer* ring, size_type index) : _ring(ring), _idx(index) {}
  
          _iterator(_iterator&& o) : _ring(o._ring), _idx(o._idx) {}
          _iterator(const _iterator& o) : _ring(o._ring), _idx(o._idx) {}
  
          _iterator& operator=(_iterator&& o) {
            _ring = o._ring;
            _idx = o._idx;
            return *this;
          }
          _iterator& operator=(const _iterator& o) {
            _ring = o._ring;
            _idx = o._idx;
            return *this;
          }
  
          T_Value& operator*() const {
            return _ring->_get(_idx);
          }
          T_Value* operator->() const {
            return &_ring->_get(_idx);
          }
          T_Value& operator[](difference_type offset) const {
            return *((*this) + offset);
          }
  
          bool operator==(const _iterator& other) const {
            return _idx == other._idx;
          }
          bool operator!=(const _iterator& other) const {
            return _idx != other._idx;
          }
          bool operator<(const _iterator& other) const {
            return (*this) - other < 0;
          }
          bool operator<=(const _iterator& other) const {
            return (*this) - other <= 0;
          }
          bool operator>(const _iterator& other) const {
            return (*this) - other > 0;
          }
          bool operator>=(const _iterator& other) const {
            return (*this) - other >= 0;
          }
  
          _iterator& operator++() {
            _idx = _ring->_incr(_idx);
            return *this;
          }
          _iterator& operator--() {
            _idx = _ring->_decr(_idx);
            return *this;
          }
          _iterator operator++(int) {
            _iterator r = *this;
            ++*this;
            return r;
          }
          _iterator operator--(int) {
            _iterator r = *this;
            --*this;
            return r;
          }
          _iterator& operator+=(difference_type offset) {
            _idx = _ring->_offset(_idx, offset);
            return *this;
          }
          _iterator& operator-=(difference_type offset) {
            _idx = _ring->_offset(_idx, -offset);
            return *this;
          }
  
          _iterator operator+(difference_type offset) const {
            _iterator r = *this;
            r += offset;
            return r;
          }
          _iterator operator-(difference_type offset) const {
            _iterator r = *this;
            r -= offset;
            return r;
          }
          difference_type operator-(const _iterator& other) const {
            return _ring->_difference(_idx, other._idx);
          }
  
        protected:
          friend class fixed_ringbuffer;
          T_ringbuffer* _ring;
          size_type _idx;
      };
  
      typedef _iterator<fixed_ringbuffer<T, Allocator>, T> iterator;
      typedef _iterator<const fixed_ringbuffer<T, Allocator>, const T>
      const_iterator;
  
      typedef std::reverse_iterator<iterator> reverse_iterator;
      typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  
    public:
      /*
        Public interface.
      */
      fixed_ringbuffer(size_type min_capacity, const Allocator& alloc = Allocator())
        : _alloc(alloc) {
        _ind_bits = _capacity_to_ind_bits(_capacity_at_least(min_capacity));
        _store = _alloc.allocate(capacity());
        _head = 0;
        _tail = 0;
      }
  
      ~fixed_ringbuffer() {
        if (_store) _deallocate();
      }
  
      // Copying
      fixed_ringbuffer& operator=(const fixed_ringbuffer& original) {
        _deallocate();
        _copy_from(original, original.capacity());
      }
      fixed_ringbuffer(const fixed_ringbuffer& original) {
        _copy_from(original, original.capacity());
      }
  
      // Moving
      fixed_ringbuffer& operator=(fixed_ringbuffer&& source) {
        _deallocate();
        _move_from(source);
      }
      fixed_ringbuffer(fixed_ringbuffer&& source) noexcept {
        _move_from(source);
      }
  
      // Request that the capacity be increased to at least min_capacity.
      void reserve(size_type min_capacity) {
        if (min_capacity > capacity()) _copy_from(*this, min_capacity);
      }
  
      // Accessors.
      const_reference operator[](size_type pos) const {
        return _get(pos);
      }
      reference operator[](size_type pos) {
        return _get(pos);
      }
      const_reference at(size_type pos) const {
        if (pos >= size()) _throw_out_of_range();
        return _get(pos);
      }
      reference at(size_type pos) {
        if (pos >= size()) _throw_out_of_range();
        return _get(pos);
      }
      const_reference front() const {
        return _get(_head);
      }
      reference front() {
        return _get(_head);
      }
      const_reference back() const {
        return _get(_decr(_tail, _ind_bits));
      }
      reference back() {
        return _get(_decr(_tail, _ind_bits));
      }
  
      // Mutators (no insert / erase / resize!)
      void clear() {
        for (T& t : *this) {
          _destroy(&t);
        }
        _head = _tail = 0;
      }
      void push_front(const T& v) {
        if (full()) throw std::bad_alloc();
        _head = _decr(_head);
        _create(&_get(_head), v);
      }
      void push_back(const T& v) {
        if (full()) throw std::bad_alloc();
        _create(&_get(_tail), v);
        _tail = _incr(_tail);
      }
      void pop_front() {
        if (empty()) return;
        _destroy(&_get(_head));
        _head = _incr(_head);
      }
      void pop_back() {
        if (empty()) return;
        _tail = _decr(_tail);
        _destroy(&_get(_tail));
      }
  
      // Erase elements.  This works similar to std::vector.
      const_iterator erase(const_iterator first, const_iterator last) {
        return const_iterator(this, _erase(first._idx, last._idx));
      }
      const_iterator erase(const_iterator position) {
        return erase(position, position + 1);
      }
      iterator erase(iterator first, iterator last) {
        return iterator(this, _erase(first._idx, last._idx));
      }
      iterator erase(iterator position) {
        return erase(position, position + 1);
      }
  
      // Swap contents with another ringbuffer
      void swap(fixed_ringbuffer& other) {
        std::swap(_alloc, other._alloc);
        std::swap(_store, other._store);
        std::swap(_head, other._head);
        std::swap(_tail, other._tail);
        std::swap(_ind_bits, other._ind_bits);
      }
  
      // Size and capacity.
      bool empty() const {
        return _head == _tail;
      }
      bool full() const {
        return _head == (_tail ^ capacity());
      }
      size_type size() const {
        return _size(_head, _tail);
      }
      size_type max_size() const {
        return capacity();
      }
      size_type capacity() const {
        return _ind_bits_to_capacity(_ind_bits);
      }
  
      // Iterators.
      iterator begin() {
        return iterator(this, _head);
      }
      iterator end() {
        return iterator(this, _tail);
      }
      const_iterator begin() const {
        return const_iterator(this, _head);
      }
      const_iterator end() const {
        return const_iterator(this, _tail);
      }
      const_iterator cbegin() const {
        return const_iterator(this, _head);
      }
      const_iterator cend() const {
        return const_iterator(this, _tail);
      }
  
      // Reverse iterators.
      reverse_iterator rbegin() {
        return reverse_iterator(end());
      }
      reverse_iterator rend() {
        return reverse_iterator(begin());
      }
      const_reverse_iterator rbegin() const {
        return const_reverse_iterator(end());
      }
      const_reverse_iterator rend() const {
        return const_reverse_iterator(begin());
      }
      const_reverse_iterator crbegin() const {
        return const_reverse_iterator(cend());
      }
      const_reverse_iterator crend() const {
        return const_reverse_iterator(cbegin());
      }
  
    private:
      Allocator _alloc;
      T* _store = nullptr;
      size_type _ind_bits = 0;  // Bits used to represent element index.  (2*capacity-1)
      size_type _head = 0;
      size_type _tail = 0;
  
    private:
      friend class _iterator<fixed_ringbuffer<T, Allocator>, T>;
      friend class _iterator<const fixed_ringbuffer<T, Allocator>, const T>;
  
      // Create & destroy
      void _create(T* t) {
        new (t) T();
      }
      void _create(T* t, const T& v) {
        new (t) T(v);
      }
      void _destroy(T* t) {
        t->~T();
      }
  
      void _throw_out_of_range() {
        throw std::out_of_range("fixed_ringbuffer::at() out of range");
      }
  
      // Access by internal index
      const_reference _get(size_type index) const {
        return _store[_slot(index)];
      }
      reference _get(size_type index) {
        return _store[_slot(index)];
      }
  
      // Index math implementation
      size_type _slot(size_type pos) const {
        return pos & (_ind_bits >> 1);
      }
      size_type _incr(size_type pos) const {
        return (pos + 1) & _ind_bits;
      }
      size_type _decr(size_type pos) const {
        return (pos + _ind_bits) & _ind_bits;
      }
      size_type _size(size_type begin, size_type end) const {
        return (end - begin) & _ind_bits;
      }
  
      size_type _offset(size_type pos, difference_type offset) const {
        return size_type((difference_type(pos) + offset) & _ind_bits);
      }
      size_type _offset(size_type pos, size_type offset) const {
        return size_type(pos + offset) & _ind_bits;
      }
  
      static size_type _capacity_at_least(size_type min_capacity) {
        return detail::next_power_of_two(min_capacity);
      }
      static size_type _capacity_to_ind_bits(size_type capacity) {
        return (capacity << 1) - 1;
      }
      static size_type _ind_bits_to_capacity(size_type ind_bits) {
        return (ind_bits + 1) >> 1;
      }
  
      void _deallocate() {
        clear();
        _alloc.deallocate(_store, capacity());
        _store = nullptr;
        _ind_bits = 0;
      }
  
      void _move_from(const fixed_ringbuffer& source) {
        _alloc = source._alloc;
        _store = source._store;
        _ind_bits = source._ind_bits;
        _head = source._head;
        _tail = source._tail;
        source._store = nullptr;
        source._ind_bits = 0;
        source._head = 0;
        source._tail = 0;
      }
      void _copy_from(const fixed_ringbuffer& original, size_type capacity) {
        // Clone original ringbuffer (which may be this object)
        size_t new_capacity =
          _capacity_at_least(std::max(capacity, original.capacity())),
          new_tail = 0;
        T* new_store = _alloc.allocate(new_capacity);
        for (const T& t : original) new_store[new_tail++] = t;
  
        // Deallocate our own storage
        if (_store) _deallocate();
  
        // Setup
        _ind_bits = _capacity_to_ind_bits(new_capacity);
        _store = new_store;
        _head = 0;
        _tail = new_tail;
      }
  
      size_type _erase(size_type first, size_type last) {
        if (last != _tail) {
          // Some elements must be moved
          size_type move_offset = _slot(first + capacity() - last);
          for (size_type i = _slot(last), ie = _slot(_tail); i != ie;
               i = _slot(i + 1))
            _get(i + move_offset) = _store[i];
        }
        _tail = (_tail + _ind_bits + 1 + first - last) & _ind_bits;
        return first;
      }
  
      difference_type _difference(size_type pos_term, size_type neg_term) const {
        return difference_type(_size(_head, pos_term)) -
               difference_type(_size(_head, neg_term));
      }
  };
} 

#endif
