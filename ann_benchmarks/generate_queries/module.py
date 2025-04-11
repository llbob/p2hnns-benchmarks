import os
import subprocess
from typing import Tuple
import numpy as np
import tempfile
import sys
import platform
import stat

def generate_hyperplanes(points: np.ndarray, n_hyperplanes: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random hyperplane using the generate.cc C++ implementation.
    
    This method compiles and runs the C++ code to generate hyperplanes using Qiang et al method.
    
    Args:
        points (np.ndarray): Input data points
        n_hyperplanes (int, optional): Number of hyperplanes to generate. Defaults to 10000.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the hyperplane normals and biases
    """
    # create a temporary directory for files
    with tempfile.TemporaryDirectory() as temp_dir:
        # prepare file paths
        input_file = os.path.join(temp_dir, 'dataset.bin')
        output_folder = os.path.join(temp_dir, 'output')
        os.makedirs(output_folder, exist_ok=True)
        
        # get dimensions
        n, d = points.shape
        print(f"Generating {n_hyperplanes} hyperplanes for {n} points in {d} dimensions")
        
        # Write dataset to binary file
        print(f"Writing dataset to {input_file}")
        with open(input_file, 'wb') as f:
            for i in range(n):
                f.write(points[i].astype(np.float32).tobytes())
        
        # get the path to the generate executable
        script_dir = os.path.dirname(os.path.abspath(__file__))
        generate_exe = os.path.join(script_dir, 'generate')
        
        # if generate executable doesn't exist, compile it
        if not os.path.exists(generate_exe) or not os.access(generate_exe, os.X_OK):
            print(f"executable not found or not executable, compiling from source")
            
            # Include platform-specific flags
            cxx_flags = "-std=c++11 -O3"
            if platform.system() == "Darwin":
                cxx_flags += " -mmacosx-version-min=10.9"
                
            generate_src = os.path.join(script_dir, 'generate.cc')
            compile_cmd = f"g++ {cxx_flags} -o {generate_exe} {generate_src}"
            
            try:
                subprocess.run(compile_cmd, shell=True, check=True)
                # make sure it's executable
                os.chmod(generate_exe, 
                         os.stat(generate_exe).st_mode | 
                         stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except subprocess.CalledProcessError as e:
                print(f"Failed to compile: {e}")
                raise RuntimeError("Failed to compile C++ generator") from e
        
        # run the generate executable
        cmd = [
            generate_exe, 
            str(n),         # cardinality
            str(d),         # dimensionality
            str(n_hyperplanes), # number of queries/hyperplanes
            "0",            # orig flag
            input_file,     # input file
            output_folder   # output folder
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running generator: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise RuntimeError("Failed to generate hyperplanes") from e
        
        # read the generated hyperplanes
        hyperplanes_file = f"{output_folder}.q"
        print(f"Reading hyperplanes from {hyperplanes_file}")
        
        if not os.path.exists(hyperplanes_file):
            raise FileNotFoundError(f"Hyperplane file not found: {hyperplanes_file}")
        
        # read the binary file to get normals and biases
        hyperplanes = np.fromfile(hyperplanes_file, dtype=np.float32)
        
        # ensure we got the right num of values
        expected_values = n_hyperplanes * (d + 1)
        if len(hyperplanes) != expected_values:
            print(f"Warning: Expected {expected_values} values but got {len(hyperplanes)}")
        
        hyperplanes = hyperplanes.reshape(-1, d + 1)
        
        # extract now the normals and biases
        normals = hyperplanes[:, :d]
        biases = hyperplanes[:, d]
        
        print(f"Generated {len(normals)} hyperplanes")
        return normals, biases
