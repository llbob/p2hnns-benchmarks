import os
import sys

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

extra_args = ['-std=c++17', '-march=native', '-O3']
extra_link_args = ['-ltbb']

if sys.platform == 'darwin':
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++', '-Xclang']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'


module = Extension(
    'ah',
    sources=['python_wrapper_ah.cpp', 'methods/pri_queue.cc','methods/util.cc'], 
    library_dirs=["methods/"],
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
    name='ah',
    version='0.1',
    description='ah Implementation',
    ext_modules=[module]
)