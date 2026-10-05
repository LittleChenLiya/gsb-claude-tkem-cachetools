import os
import sys

# Ensure the local source tree takes precedence over any installed copy.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
