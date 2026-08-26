"""pytest 配置。"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
# 确保 src 与项目根目录（scope_router 所在）都在路径中
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
