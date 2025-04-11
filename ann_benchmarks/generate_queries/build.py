import os
import subprocess
import platform
import stat

def build_generate():
    #get the cur dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # define now source and output paths
    source_file = os.path.join(current_dir, "generate.cc")
    output_file = os.path.join(current_dir, "generate")
    
    # set compilation flags based on platform
    cxx_flags = "-std=c++11 -O3"
    
    # add platform-specific flags
    if platform.system() == "Darwin":  # macOS
        cxx_flags += " -mmacosx-version-min=10.9"
    
    # build the command
    cmd = f"g++ {cxx_flags} -o {output_file} {source_file}"
    
    # execute the compilation
    print(f"compiling this: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    
    # make sure the executable has the right permissions
    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    print(f"successfully complie of: {output_file}")

if __name__ == "__main__":
    build_generate()
