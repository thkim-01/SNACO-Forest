"""Path initialization utility for experiment scripts."""
import os
import sys
from pathlib import Path


def init_paths():
    """Set working directory and sys.path for src module imports.
    
    Returns:
        bool: True if paths set successfully.
    """
    try:
        # Get absolute paths
        script_dir = Path(__file__).resolve().parent  # experiments/
        project_root = script_dir.parent  # Project root
        
        # Change working directory to project root
        os.chdir(str(project_root))
        
        # Clean sys.path and add project root at position 0
        project_root_str = str(project_root)
        # Remove all existing project root entries
        sys.path = [p for p in sys.path if p != project_root_str]
        # Insert at beginning
        sys.path.insert(0, project_root_str)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Path initialization failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False
