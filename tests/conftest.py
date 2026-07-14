"""pytest 配置。"""

import sys
from pathlib import Path

# 确保 src 在路径中
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))
