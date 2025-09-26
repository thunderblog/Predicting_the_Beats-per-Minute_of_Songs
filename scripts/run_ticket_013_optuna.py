#!/usr/bin/env python3
"""
TICKET-013実行スクリプト

Optuna最適化システムの実行
最高性能TICKET-017-01+02特徴量でのハイパーパラメータ最適化
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.optimization import run_ticket_013_optimization

if __name__ == "__main__":
    print("TICKET-013 Optuna最適化システム起動中...")
    success = run_ticket_013_optimization()

    if success:
        print("\n✅ TICKET-013 実行完了")
    else:
        print("\n❌ TICKET-013 実行失敗")
        sys.exit(1)