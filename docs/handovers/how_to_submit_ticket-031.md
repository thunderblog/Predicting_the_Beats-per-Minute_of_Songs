  # ファイル確認
  ls data/processed/submission_ticket031_optimized_*.csv

  # 提出実行（実際のファイル名に置き換え）
  kaggle competitions submit -c playground-series-s5e9 -f
  "data/processed/submission_ticket031_optimized_20250930_190000.csv"    
   -m "TICKET-031 Optimized Ensemble"