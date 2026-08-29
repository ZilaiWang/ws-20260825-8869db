#!/usr/bin/env python3
"""容器 ENTRYPOINT；业务逻辑位于 rsdet.submission.competition。"""

from rsdet.submission.competition import main

if __name__ == "__main__":
    raise SystemExit(main())
