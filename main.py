"""矩阵计算器入口"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import cli

if __name__ == "__main__":
    cli()
