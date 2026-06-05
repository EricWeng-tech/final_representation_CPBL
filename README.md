# CPBL 勝負預測

使用中華職棒 2018–2025 一軍例行賽資料，建立賽前主隊勝負預測模型。

## 結果

訓練集：2018–2024（1,947 場）｜測試集：2025（358 場）｜特徵數：33

| 模型 | Accuracy | AUC | F1 |
|------|---------:|----:|---:|
| Random Forest baseline | 69.27% | 0.7678 | 0.7277 |
| Random Forest tuned | 70.39% | 0.7647 | 0.7389 |
| TabPFN | 69.27% | 0.7494 | 0.7208 |
| Logistic Regression（11 diff features） | 53.35% | 0.5380 | 0.6640 |

## 資料來源

rebas.tw 公開 JSON API，涵蓋 2018–2026 年一軍例行賽，共 29 個賽季。

## 專案結構

```
scripts/
  scrape_games.py               逐場比賽 box score（主爬蟲）
  scrape_all.py                 球員賽季累計統計
  validate_data.py              資料品質驗證（PASS 58 / WARN 8 / FAIL 0）
  build_model_ready.py          特徵工程 → 建模主表

notebooks/
  01_random_forest.ipynb
  02_logistic_regression.ipynb
  03_tabpfn.ipynb
  04_model_comparison.ipynb
  experiments/rf_tuning/        Random Forest 調參實驗

data/raw/                       四份爬蟲 CSV
data/processed/                 model_ready_games.csv（2,418 場 × 41 欄）
data/cache/                     API JSON cache，不進 Git

outputs/metrics/                模型指標與係數
outputs/figures/                特徵重要性、SHAP 圖
outputs/predictions/            逐場預測結果
outputs/experiments/rf_tuning/  調參輸出

docs/
  DATA_DICTIONARY.md            欄位定義
  model_assumptions.md          模型假設與前處理說明
```

## 快速開始

建議使用 Python 3.12，Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\validate_data.py
jupyter lab
```

TabPFN 另行安裝（僅執行 `03_tabpfn.ipynb` 時需要）：

```powershell
python -m pip install -r requirements-tabpfn.txt
```

## 執行順序

模型分析（直接執行）：

```
notebooks/01_random_forest.ipynb
notebooks/02_logistic_regression.ipynb
notebooks/03_tabpfn.ipynb
notebooks/04_model_comparison.ipynb
```

重新抓資料與重建特徵：

```powershell
python scripts\scrape_games.py
python scripts\validate_data.py
python scripts\build_model_ready.py
```

欄位定義請讀 [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)。
