/*
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <tuple>
#include <functional>
#include <queue>

namespace py = pybind11;
using ScoreTuple = std::tuple<float, int>;  // score, index

// Comparator for min-heap (smallest element at top).
// We use std::greater-like logic because std::make_heap creates a max-heap by
// default. To get a min-heap behavior with std::push_heap/pop_heap, we need a
// comparator that returns true if 'a' should be closer to the root than 'b'
// (i.e., a > b for min-heap).
struct PyObjectGreater {
  bool operator()(const py::object& a, const py::object& b) const {
    // PyObject_RichCompareBool returns 1 (true), 0 (false), or -1 (error)
    int res = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_GT);
    if (res == -1) {
      throw py::error_already_set();
    }
    return res == 1;
  }
};

struct TupleGreater {
  bool operator()(const ScoreTuple& a, const ScoreTuple& b) const {
    if (std::get<0>(a) > std::get<0>(b)) return true;
    return false;
  }
};

template <typename T, typename Comparator>
class MinHeap : public std::priority_queue<T, std::vector<T>, Comparator> {
 public:
  using Base = std::priority_queue<T, std::vector<T>, Comparator>;

  MinHeap(size_t max_size) : max_size_(max_size) { this->c.reserve(max_size); }

  bool push(const T& item) {
    if (this->size() < max_size_) {
      Base::push(item);
      return true;
    } else {
      Comparator comp;
      if (comp(item, this->top())) {
        Base::pop();
        Base::push(item);
        return true;
      }
    }
    return false;
  }

  const std::vector<T>& items() const { return this->c; }

  size_t len() const { return this->size(); }

  void push_list(const std::vector<T>& items) {
    for (const auto& item : items) {
      push(item);
    }
  }

  void push_candidates_striped(py::array_t<float> scores,
                               py::array_t<int64_t> token_ids, int chunk_size,
                               long long eos_token_id) {
    auto scores_r = scores.unchecked<1>();
    auto token_ids_r = token_ids.unchecked<1>();

    size_t num_items = scores_r.size();
    if (num_items == 0) return;
    if (chunk_size <= 0) {
      throw std::invalid_argument("chunk_size must be positive");
    }
    if (num_items % chunk_size != 0)
      throw std::invalid_argument("num_items must be divisible by chunk_size");

    size_t num_chunks = num_items / chunk_size;
    std::vector<int> active_chunks(num_chunks);
    std::iota(active_chunks.begin(), active_chunks.end(), 0);
    size_t active_count = num_chunks;

    for (int i = 0; i < chunk_size; ++i) {
      if (active_count == 0) break;

      size_t write_pos = 0;
      for (size_t k = 0; k < active_count; ++k) {
        size_t chunk_idx = active_chunks[k];
        int idx = chunk_idx * chunk_size + i;

        if (idx >= num_items) continue;

        if (token_ids_r(idx) == eos_token_id) {
          active_chunks[write_pos++] = chunk_idx;
          continue;
        }
        if (push_inplace(scores_r(idx), idx)) {
          active_chunks[write_pos++] = chunk_idx;
        }
      }
      active_count = write_pos;
    }
  }

  std::vector<T> sorted_items() {
    std::vector<T> copy = this->c;
    std::sort(copy.begin(), copy.end(), Comparator());
    return copy;
  }

 private:
  bool push_inplace(float score, int idx) {
    if (this->size() < max_size_) {
      this->emplace(score, idx);
      return true;
    } else {
      const auto& top = this->top();
      if (score > std::get<0>(top)) {
        Base::pop();
        Base::emplace(score, idx);
        return true;
      }
    }
    return false;
  }

  size_t max_size_;
};

using MinHeapObject = MinHeap<py::object, PyObjectGreater>;
using MinHeapTuple = MinHeap<ScoreTuple, TupleGreater>;

PYBIND11_MODULE(minheap_cpp, m) {
  py::class_<MinHeapObject>(m, "MinHeap")
      .def(py::init<size_t>())
      .def("push", &MinHeapObject::push)
      .def("push_list", &MinHeapObject::push_list)
      .def("items", &MinHeapObject::items)
      .def("sorted_items", &MinHeapObject::sorted_items)
      .def("__len__", &MinHeapObject::len);

  py::class_<MinHeapTuple>(m, "MinHeapTuple")
      .def(py::init<size_t>())
      .def("push_candidates_striped", &MinHeapTuple::push_candidates_striped)
      .def("items", &MinHeapTuple::items)
      .def("sorted_items", &MinHeapTuple::sorted_items)
      .def("__len__", &MinHeapTuple::len);
}
