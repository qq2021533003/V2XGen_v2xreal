import sys
import os

def setup_project_path():
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

setup_project_path()