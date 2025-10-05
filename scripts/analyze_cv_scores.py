"""CV性能分析スクリプト"""
import pandas as pd
from pathlib import Path

# データ読み込み
results_path = Path(__file__).parent.parent / "experiments" / "experiment_results.csv"
df = pd.read_csv(results_path)

# 有効なCV性能のみ抽出
df_valid = df[df['cv_score'] != 'TBD'].copy()
df_valid['cv_score'] = pd.to_numeric(df_valid['cv_score'], errors='coerce')
df_valid = df_valid.dropna(subset=['cv_score'])

# CV性能でソート
df_sorted = df_valid.sort_values('cv_score')

print("=" * 80)
print("CV性能 トップ10")
print("=" * 80)
print(df_sorted[['exp_id', 'exp_name', 'cv_score', 'cv_std', 'lb_score']].head(10).to_string(index=False))
print("\n")

print("=" * 80)
print("CV性能 ベスト3の詳細")
print("=" * 80)
for idx, row in df_sorted.head(3).iterrows():
    print(f"\n【{row['exp_id']}】{row['exp_name']}")
    print(f"  CV: {row['cv_score']:.6f} ± {row['cv_std']}")
    print(f"  LB: {row['lb_score']}")
    print(f"  CV-LB: {row['cv_lb_diff']}")
    print(f"  Features: {row['n_features']}")
    print(f"  Model: {row['model_type']}")
    print(f"  Note: {row['notes']}")