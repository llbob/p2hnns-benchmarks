import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
import platform
import stat

from p2hnns_benchmarks.main import positive_int

def build_generate_queries():
    """Build the C++ component for generate_queries"""
    # get paths for source and output files
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2hnns_benchmarks", "generate_queries")
    source_file = os.path.join(script_dir, "generate.cc")
    output_file = os.path.join(script_dir, "generate")
    
    # Set compilation flags based on platform
    cxx_flags = "-std=c++11 -O3 -Wno-deprecated-declarations"
    if platform.system() == "Darwin":  # macOS
        cxx_flags += " -mmacosx-version-min=10.9"
    
    # Build the command
    cmd = f"g++ {cxx_flags} -o {output_file} {source_file}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        os.chmod(output_file, 
                os.stat(output_file).st_mode | 
                stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("successfully built generate_queries component")
        return True
    except Exception as e:
        print(f"Failed to build generate_queries component: {e}")
        print("you may have to build it manually, or for now skip using the bctree queries using Qiang et al's methods")
        return False



def build(library, args):
    print("Building %s..." % library)
    if args is not None and len(args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in args])
    else:
        q = ""

    try:
        subprocess.check_call(
            "docker build %s --rm -t p2hnns-benchmarks-%s -f" " p2hnns_benchmarks/algorithms/%s/Dockerfile  ." % (q, library, library),
            shell=True,
        )
        return {library: "success"}
    except subprocess.CalledProcessError:
        return {library: "fail"}


def build_multiprocess(args):
    return build(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--proc", default=1, type=positive_int, help="the number of process to build docker images")
    parser.add_argument("--algorithm", metavar="NAME", help="build only the named algorithm image", default=None)
    parser.add_argument("--build-arg", help="pass given args to all docker builds", nargs="+")
    parser.add_argument("--skip-cpp", action="store_true", help="skip building C++ components")
    args = parser.parse_args()

    print("Building base image...")
    subprocess.check_call(
         "docker build \
         --rm -t p2hnns-benchmarks -f p2hnns_benchmarks/algorithms/base/Dockerfile .",
         shell=True,
     )
    
    if args.algorithm:
        tags = [args.algorithm]
    elif os.getenv("LIBRARY"):
        tags = [os.getenv("LIBRARY")]
    else:
        tags = [fn.split(".")[-1] for fn in os.listdir("p2hnns_benchmarks/algorithms")]

    print("Building algorithm images... with (%d) processes" % args.proc)

    if args.proc == 1:
        install_status = [build(tag, args.build_arg) for tag in tags]
    else:
        pool = Pool(processes=args.proc)
        install_status = pool.map(build_multiprocess, [(tag, args.build_arg) for tag in tags])
        pool.close()
        pool.join()

    print("\n\nInstall Status:\n" + "\n".join(str(algo) for algo in install_status))

    # Exit 1 if any of the installations fail.
    for x in install_status:
        for (k, v) in x.items():
            if v == "fail":
                sys.exit(1)