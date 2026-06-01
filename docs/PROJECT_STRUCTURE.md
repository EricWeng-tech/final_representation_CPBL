# 專案目錄說明

## 原則

這個 repository 採用「程式、資料、模型分析、輸出、文件」分層。隊友 clone 後可以直接驗證資料與執行 notebook；可重新產生的大型 cache 留在本機，不加入 Git。

## 目錄

```text
期末報告/
├─ scripts/
│  ├─ scrape_all.py
│  ├─ scrape_games.py
│  ├─ validate_data.py
│  └─ build_model_ready.py
├─ notebooks/
│  ├─ 01_random_forest.ipynb
│  ├─ 02_logistic_regression.ipynb
│  ├─ 03_tabpfn.ipynb
│  ├─ 04_model_comparison.ipynb
│  └─ experiments/rf_tuning/05_random_forest_tuning.ipynb
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ cache/
├─ outputs/
│  ├─ metrics/
│  ├─ predictions/
│  ├─ figures/
│  └─ experiments/rf_tuning/
├─ docs/
├─ _archive/
├─ README.md
├─ AGENTS.md
├─ requirements.txt
└─ requirements-tabpfn.txt
```

## 資料分層

| 位置 | 用途 | 是否進 Git |
|------|------|------------|
| `data/raw/` | 四份爬蟲 CSV，供資料檢查與特徵工程重現 | 是 |
| `data/processed/` | notebook 直接讀取的 `model_ready_games.csv` | 是 |
| `data/cache/` | API 回傳 JSON，可由爬蟲重新下載 | 否 |
| `_archive/` | 舊模型、失敗嘗試、暫停中的舊年份爬蟲快照 | 否 |

## 輸出分層

| 位置 | 用途 | 是否進 Git |
|------|------|------------|
| `outputs/metrics/` | 模型比較、係數、整體指標 | 是 |
| `outputs/figures/` | 報告用圖表與可解釋性圖表 | 是 |
| `outputs/experiments/rf_tuning/` | Random Forest 調參結果 | 是，逐場 predictions 除外 |
| `outputs/predictions/` | 每一場的預測結果 | 否 |

## 執行方式

資料檢查：

```powershell
python scripts\validate_data.py
```

重建建模主表：

```powershell
python scripts\build_model_ready.py
```

核心 notebook 執行順序：

```text
01_random_forest.ipynb
02_logistic_regression.ipynb
03_tabpfn.ipynb
04_model_comparison.ipynb
```

Random Forest 參數搜尋是獨立實驗，不覆寫核心結果：

```text
notebooks/experiments/rf_tuning/05_random_forest_tuning.ipynb
```

## 本機環境提醒

原本專案內的 `.venv/` 已存在但不可作為團隊環境依賴。建立新環境時請依照根目錄 README 安裝。Eric 目前可執行 TabPFN 的既有環境為：

```text
C:\Users\Eric\anaconda3\envs\tabpfn312
```
