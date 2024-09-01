#ifndef Monotonic_Wedge_h
#define Monotonic_Wedge_h

/*
  This code is available under the MIT license:

    Copyright (c) 2016 Evan Balster

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

  The original Lemire algorithm is "patent-free".  For more information on the Lemire algorithm:

    Code:  https://github.com/lemire/runningmaxmin

    Paper: https://arxiv.org/abs/cs/0610046

  This header presents algorithms for fast running minimum and maximum using
    the Daniel Lemire monotonic wedge algorithm with enhancements proposed
    by Ethan Fenn.

  The algorithm here is modeled on the C++ STL style and meant to be used
    with vector, deque or ring-buffer structures.

  The amortized complexity of the update operation is constant;
    IE, N updates can be performed in linear time on a given wedge.
    The worst-case complexity for a single update is below log2(N).

  Usage recommendations:

    This algorithm is most useful for "rolling" min / max evaluation.
    Most applications will thus prefer deques or ring-buffers as wedges.

    Generally, values beyond a certain age should be popped to limit the
    size of the wedge, though amortized complexity will remain linear even
    if this is not done.

    The wedge must be monotonic at all times with respect to Compare,
    EG. by only modifying the structure with wedge_update and pop_front.

    Adopted from: https://github.com/EvanBalster/STL_mono_wedge

*/

#include <algorithm>
#include <functional>
#include "STL_RingBuffer.h"

#if __cplusplus > 199711L
#include <utility> // For std::forward
#endif

namespace mono_wedge {
  /*
    mono_wedge_search(begin, end, value, comp)
  
    Search routine used to determine deletion range in mono_wedge_update.
  
    Similar to std::lower_bound, returns first element in range for which
      comp(value, element) returns false.
  
      Range must be sorted with regard to comp.
      Iterator must be a random access iterator.
      Complexity is below log2(N) with respect to wedge size.
      Facilitates amortized constant complexity in mono_wedge_update.
  */
  template<class Iterator, class T, class Compare>
  Iterator mono_wedge_search(Iterator begin, Iterator end, const T& value, Compare  comp) {
    size_t size = std::distance(begin, end);
    if (size <= 0ul) return end;
  
    // Linear search through at most J elements, where J = log2(N-J).
    Iterator search_pos = end; --search_pos;
    size_t i = 1ul;
    for (; ((size - i) >> i) > 0ul; ++i, --search_pos) {
      if (comp(*search_pos, value)) return ++search_pos;
    }
    // Afterwards run a binary search (use std::lower_bound)
    return std::lower_bound<Iterator, T, Compare>(begin, ++search_pos, value, comp);
  }
  
  /*
    mono_wedge_update(wedge, value, comp)
  
    Update a monotonic wedge with a new value.
  
      Erases values which do not satisfy comp(element, value), then
      appends value to the wedge via push_back.
  
      Complexity is less than log2(N) with respect to wedge size.
      Complexity of N calls is O(N), if wedge is initially empty.
      Thus, amortized complexity over many calls is constant.
  
    Wedge type must:
      - Produce random access iterators via begin/end.
      - Support push_back.
  
    A "less" comparator yields a min-wedge.
    A "greater" comparator yields a max-wedge.
  */
  template<class Wedge, class T, class Compare>
  void mono_wedge_update(Wedge& wedge, const T& value, Compare  comp) {
    auto i = mono_wedge_search(wedge.begin(), wedge.end(), value, comp);
    wedge.erase(i, wedge.end());
    wedge.push_back(value);
  }
  
  /*
    min_wedge_search(wedge, value)
    min_wedge_search(wedge, value)
  
    Convenience variants of mono_wedge_search for min and max wedges.
      These will use std::greater/less, which default to operator >/<.
  */
  template<class Iterator, class T>
  Iterator min_wedge_search(Iterator begin, Iterator end, const T& value) {
    return mono_wedge_search(begin, end, value, std::less<T>());
  }
  
  template<class Iterator, class T>
  Iterator max_wedge_search(Iterator begin, Iterator end, const T& value) {
    return mono_wedge_search(begin, end, value, std::greater<T>());
  }
  
  /*
    min_wedge_update(wedge, value)
    min_wedge_update(wedge, value)
  
    Convenience variants of mono_wedge_update for min and max wedges.
      These will use std::greater/less, which default to operator >/<.
  */
  template<class Wedge, class T>
  void min_wedge_update(Wedge& wedge, const T& value) {
    return mono_wedge_update(wedge, value, std::less<T>());
  }
  
  template<class Wedge, class T>
  void max_wedge_update(Wedge& wedge, const T& value) {
    return mono_wedge_update(wedge, value, std::greater<T>());
  }
  
  #if __cplusplus > 199711L
    /*
      C++11 variants of mono_wedge_update supporting rvalue references.
    */
    template<class Wedge, class T, class Compare>
    void mono_wedge_update(Wedge& wedge, T&& value, Compare comp) {
      typename Wedge::iterator i = mono_wedge_search(wedge.begin(), wedge.end(), value, comp);
      size_t erase_count = std::distance(i, wedge.end());
      while (erase_count--) wedge.pop_back();
      wedge.push_back(std::forward(value));
    }
    
    template<class Wedge, class T, class Compare>
    void min_wedge_update(Wedge& wedge, T&& value) {
      mono_wedge_update(wedge, std::forward(value), std::less<T>());
    }
    
    template<class Wedge, class T, class Compare>
    void max_wedge_update(Wedge& wedge, T&& value) {
      mono_wedge_update(wedge, std::forward(value), std::greater<T>());
    }
  #endif
}

#endif
