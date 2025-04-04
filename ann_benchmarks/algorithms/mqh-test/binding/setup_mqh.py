import os
import sys
import platform

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

# Detect architecture
is_arm = platform.machine().startswith(('arm', 'aarch64'))

# Define specific flags for SIMD support - only on x86
simd_flags = []
# Define preprocessor macros for SIMD support
define_macros = []

if not is_arm:
    # x86 architecture - can use AVX/SSE2
    simd_flags = ['-mavx', '-msse2']
    define_macros = [('HAVE_X86INTRIN', '1')]
else:
    # ARM architecture - define ARM flag
    define_macros = [('MQH_ARM', '1')]

# Basic compiler flags
extra_args = ['-std=c++17', '-O3'] + simd_flags

# Add native architecture optimization only for non-ARM
if not is_arm:
    extra_args.append('-march=native')
else:
    # For ARM, you might want to add specific ARM optimizations
    # but be careful with cross-compilation scenarios
    pass

extra_link_args = ['-ltbb']

# Define platform-specific flags
if sys.platform != 'darwin':
    extra_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
else:
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++', '-Xclang', '-fopenmp']
    extra_link_args += ['-lomp']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'

module = Extension(
    'pymqh',
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
    name='pymqh',
    version='0.1',
    description='MQH Implementation',
    ext_modules=[module]
)