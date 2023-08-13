import os
import sys

# Determine the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory (project root) to the sys.path
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

import benchmarx