# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import numpy
import pybind11
from setuptools import Extension, setup


def get_ext_modules():
    # The C++ extension source is expected to be in vllm_gr/csrc/
    # This is based on the import of `minheap_cpp` in the project.
    csrc_dir = os.path.join("vllm_gr", "csrc")
    minheap_cpp_source = os.path.join(csrc_dir, "minheap.cpp")

    if not os.path.exists(minheap_cpp_source):
        # If there's no C++ source, we assume no extensions are to be built.
        # This will cause a runtime error if minheap_cpp is imported,
        # but it allows installation to proceed if the C++ part is optional
        # or located elsewhere.
        print(
            f"Warning: C++ extension source '{minheap_cpp_source}' not found. "
            "Skipping C++ extension build."
        )
        return []

    return [
        Extension(
            "minheap_cpp",
            [minheap_cpp_source],
            include_dirs=[pybind11.get_include(), numpy.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3"],
        )
    ]


setup(
    ext_modules=get_ext_modules(),
)
