import os
import subprocess
from typing import Tuple
import numpy as np
import tempfile

def generate_hyperplanes(points: np.ndarray, n_hyperplanes: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory() as temp_dir:
        # prepare file paths
        input_file = os.path.join(temp_dir, 'dataset.bin')
        output_folder = os.path.join(temp_dir, 'output')
        os.makedirs(output_folder, exist_ok=True)
        
        # get dimensions and write dataset to binary file
        n, d = points.shape
        with open(input_file, 'wb') as f:
            for i in range(n):
                f.write(points[i].astype(np.float32).tobytes())
        
        # get the path to the generate executable and run it
        script_dir = os.path.dirname(os.path.abspath(__file__))
        generate_exe = os.path.join(script_dir, 'generate')
        
        cmd = [
            generate_exe, 
            str(n),
            str(d),
            str(n_hyperplanes),
            "0",
            input_file,
            output_folder
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Failed to generate hyperplanes") from e
        
        # read the generated hyperplanes
        hyperplanes_file = f"{output_folder}.q"
        if not os.path.exists(hyperplanes_file):
            raise FileNotFoundError(f"hyperplane file not found: {hyperplanes_file}")
        
        # read the binary file to get normals and biases
        hyperplanes = np.fromfile(hyperplanes_file, dtype=np.float32)
        hyperplanes = hyperplanes.reshape(-1, d + 1)
        
        # extract now the normals and biases
        normals = hyperplanes[:, :d]
        biases = hyperplanes[:, d]
        
        return normals, biases