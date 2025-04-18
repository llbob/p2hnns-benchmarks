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
print(f"Detected platform: {platform.machine()}")

# Basic compiler flags
extra_args = ['-std=c++17', '-O3']
extra_link_args = ['-ltbb']
define_macros = []

# Architecture-specific optimizations
if is_arm:
    # ARM-specific optimizations (for M1/M2 Macs and other ARM platforms)
    define_macros.append(('MQH_ARM', '1'))
    define_macros.append(('MQH_NEON', '1'))  # Enable ARM NEON
    extra_args.append('-march=armv8-a+simd')  # Enable NEON instructions
    print("Enabling ARM NEON SIMD support")
else:
    # x86 architecture - enable all common SIMD extensions
    define_macros.append(('FORCE_X86_SIMD', '1'))
    extra_args.extend(['-msse2', '-mavx'])
    print("Enabling x86 SIMD (SSE2/AVX) support")

print(f"Compiler flags: {extra_args}")
print(f"Define macros: {define_macros}")

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
    description='MQH Implementation with ARM NEON/x86 SIMD support',
    ext_modules=[module]
)