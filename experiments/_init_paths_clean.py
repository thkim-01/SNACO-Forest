"""Path initialization utility for experiment scripts."""
import os
import sys
from pathlib import Path


def init_paths():
    """Set working directory and sys.path for src module imports."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    os.chdir(str(project_root))
    
    project_root_str = str(project_root)
    sys.path = [p for p in sys.path if p != project_root_str]
    sys.path.insert(0, project_root_str)
    
    return True
