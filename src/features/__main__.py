#!/usr/bin/env python3
"""
Features module CLI entry point

特徴量エンジニアリングモジュールのコマンドラインインターフェース
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# src.features.pyのCLI実行を呼び出し
if __name__ == "__main__":
    from src.features import main
    main()