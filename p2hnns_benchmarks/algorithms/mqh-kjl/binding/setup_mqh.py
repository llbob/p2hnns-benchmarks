import os
import sys
import platform

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

extra_args = ['-std=c++17', '-O3']
extra_link_args = ['-ltbb']
define_macros = []

module = Extension(
    'pymqhkjl',
    sources=['python_wrapper_mqh.cpp'], 
    library_dirs=["methods/"],
    define_macros=define_macros,
    extra_compile_args=extra_args,
    extra_link_args=extra_link_args,
    runtime_library_dirs=['methods/'],
    include_dirs=[
        'methods/', 
        'external/pybind11/include', 
        'libs',
        './',  # Add root directory to include path
    ])

setup(
    name='pymqhkjl',
    version='0.1',
    description='MQH-KJL Implementation without SIMD support',
    ext_modules=[module]
)