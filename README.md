# CPBL 勝負預測期末報告

使用中華職棒一軍例行賽資料，建立賽前主隊勝負預測模型。專案重點是可重現、可解釋，以及方便課堂分工協作。

## 目前結果

目前以 2018-2024 年訓練、2025 年測試。33 個特徵的 Random Forest 是目前最佳核心模型：

| 模型 | Accuracy | AUC | F1 |
|------|---------:|----:|---:|
| Random Forest baseline | 69.27% | 0.7678 | 0.7277 |
| Random Forest tuned | 70.39% | 0.7647 | 0.7389 |
| TabPFN（33 features） | 69.27% | 0.7494 | 0.7208 |
| Logistic Regression（11 diff features） | 53.35% | 0.5380 | 0.6640 |

85% 是後續目標，目前尚未達成。已完成的快速調參讓 Random Forest Accuracy 增加 1.12 個百分點。

## 專案結構

```text
scripts/                 爬蟲、特徵工程、資料品質檢查
notebooks/               核心模型 notebook
notebooks/experiments/   獨立實驗 notebook
data/raw/                可重現資料處理的四份原始 CSV
data/processed/          notebook 直接讀取的建模主表
data/cache/              可重新下載的 JSON cache，不進 Git
outputs/metrics/         模型指標與係數
outputs/figures/         可解釋性圖表
outputs/experiments/     實驗結果
docs/                    報告、資料字典與專案文件
_archive/                本機封存資料，不進 Git
```

完整說明請讀 [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)。欄位定義請讀 [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)。

## 快速開始

建議使用 Python 3.12。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\validate_data.py
jupyter lab
```

clone 後可以直接執行資料檢查與模型 notebook。TabPFN 的安裝較大，只有需要執行 `03_tabpfn.ipynb` 時才安裝：

```powershell
python -m pip install -r requirements-tabpfn.txt
```

## 建議執行順序

一般模型分析直接依序執行：

```text
notebooks/01_random_forest.ipynb
notebooks/02_logistic_regression.ipynb
notebooks/03_tabpfn.ipynb
notebooks/04_model_comparison.ipynb
```

重新抓資料與重建特徵時才執行：

```powershell
python scripts\scrape_games.py
python scripts\validate_data.py
python scripts\build_model_ready.py
```

`scripts/scrape_all.py` 用於抓取球員賽季統計快照，不是核心模型 notebook 的必要步驟。

## Git 協作

每個人用自己的 branch 開發，確認資料檢查通過後再合併：

```powershell
git switch -c feature/<your-topic>
git add .
git commit -m "Describe your change"
git push -u origin feature/<your-topic>
```

請勿提交 `.venv/`、`data/cache/`、`_archive/` 或逐場 prediction CSV。原始 CSV 與 `data/processed/model_ready_games.csv` 會保留在 Git，讓隊友不必重新爬資料即可重現分析。

## 舊年份爬蟲

2015-2017 年 CPBL 官網資料的抓取快照已放在本機 `_archive/legacy_cpbl_official_2015_2017/`。這批資料尚未合併到正式模型輸入；細節與官網 endpoint 記錄在 [`AGENTS.md`](AGENTS.md)。
