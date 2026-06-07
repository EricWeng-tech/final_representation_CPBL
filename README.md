# CPBL 勝負預測

使用中華職棒 2018–2025 一軍例行賽資料，建立賽前主隊勝負預測模型。

## 結果

訓練集：2018–2024（1,947 場）｜測試集：2025（358 場）｜特徵數：33

| 模型 | Accuracy | AUC | F1 | Brier |
|------|---------:|----:|---:|------:|
| Random Forest baseline | 69.27% | 0.7679 | 0.7277 | 0.2029 |
| TabPFN（All 33） | 69.27% | 0.7494 | 0.7208 | 0.2059 |
| TabPFN（Diff 11） | 68.72% | 0.7415 | 0.7186 | 0.2085 |
| Logistic Regression（Diff 11） | 53.35% | 0.5380 | 0.6640 | 0.2472 |

### 5-Fold Walk-Forward CV（RF，跨年泛化估計）

| Fold | 驗證年 | 訓練場數 | Accuracy | AUC |
|------|--------|---------|----------|-----|
| 1 | 2021 | 713 | 57.59% | 0.6297 |
| 2 | 2022 | 1,003 | 57.24% | 0.6081 |
| 3 | 2023 | 1,293 | 60.47% | 0.6907 |
| 4 | 2024 | 1,589 | 64.80% | 0.6814 |
| 5 | 2025 | 1,947 | 69.27% | 0.7679 |
| **平均** | | | **61.87% ±0.057** | **0.6756 ±0.059** |

> 跨年平均 61.87% 為更穩健的泛化估計；Walk-Forward 趨勢顯示訓練資料量越大表現越好，持續累積賽季資料可望進一步提升。

### RF Fine-Tuning（RandomizedSearchCV，80 組）

搜尋空間：n_estimators / max_depth / min_samples_leaf / max_features

| 模型 | Accuracy | AUC | F1 | Brier |
|------|---------:|----:|---:|------:|
| RF Baseline | 69.27% | 0.7679 | 0.7277 | 0.2029 |
| RF Tuned | **70.39%** | 0.7647 | **0.7389** | 0.2084 |

最佳參數：`n_estimators=1200, max_depth=4, min_samples_leaf=5, max_features=log2`

### Confidence Threshold 分析（RF）

只預測模型信心度超過 threshold 的場次，準確率隨覆蓋率下降而上升。

| Threshold | Pre-Tune 覆蓋率 | Pre-Tune 準確率 | Tuned 覆蓋率 | Tuned 準確率 |
|-----------|---------------:|---------------:|-------------:|-------------:|
| 0.50 | 100.0% | 69.27% | 100.0% | 70.39% |
| 0.60 | 55.9% | 80.00% | 49.4% | 81.36% |
| 0.65 | 38.6% | 81.88% | 26.8% | 81.25% |
| 0.70 | 21.5% | 84.42% | 12.3% | 86.36% |

## 防資料洩漏策略

所有特徵皆以 `shift(1)` 確保只使用**賽前已知資訊**：

- **球員指標**（ERA / WHIP / FIP / OPS / OBP / SLG）：依 `player_id` 排序後取上一場出賽後的賽季累計值
- **近 10 場滾動統計**（勝率 / 得分 / 失分）：依 `team_id` 排序後 `shift(1).rolling(10)`，排除本場
- 絕不使用本場得分、安打、失誤、WPA、IP/ER/SO 等本場結果欄位

## 資料來源

rebas.tw 公開 JSON API，涵蓋 2018–2026 年一軍例行賽，共 29 個賽季。

## 專案結構

```
scripts/
  scrape_games.py               逐場比賽 box score（主爬蟲）
  scrape_all.py                 球員賽季累計統計
  validate_data.py              資料品質驗證（PASS 57 / WARN 8 / FAIL 0）
  build_model_ready.py          特徵工程 → 建模主表

notebooks/
  01_random_forest.ipynb        RF 訓練、SHAP、Walk-Forward CV
  02_logistic_regression.ipynb  LR 訓練（VIF 篩選 diff features）
  03_tabpfn.ipynb               TabPFN 訓練（需 GPU 環境）
  04_model_comparison.ipynb     三模型彙整比較
  experiments/rf_tuning/        Random Forest 調參實驗

data/raw/                       四份爬蟲 CSV
data/processed/                 model_ready_games.csv（2,418 場 × 41 欄）
data/cache/                     API JSON cache，不進 Git

outputs/metrics/                模型指標（rf / lr / tabpfn / walk-forward / comparison）
outputs/figures/                feature_importance_rf.png、shap_summary.png
outputs/predictions/            逐場預測結果，不進 Git
outputs/experiments/rf_tuning/  調參輸出

docs/
  DATA_DICTIONARY.md            欄位定義
  model_assumptions.md          模型假設與前處理說明
```

## Streamlit 賽前預測介面

### 啟動

```powershell
# 切換到專案根目錄後執行
.\.venv\Scripts\streamlit.exe run scripts\streamlit_app.py
```

瀏覽器開啟 `http://localhost:8501`，關閉按 `Ctrl+C`。

### 操作流程

1. 選擇**比賽日期**、**客隊**、**主隊**
2. 按 **Auto-fill**：自動填入雙方近 10 場滾動數據，並選出 ERA 最低先發投手
3. 展開「牛棚投手」或「打線」可手動指定球員（選填）
4. 確認中間**指標對比表**顯示正確
5. 按 **⚾ 預測勝負**，顯示：
   - 主客隊勝率與信心等級
   - SHAP 特徵貢獻長條圖（紅＝有利主隊、藍＝有利客隊）
   - Gemini AI 白話分析（需設定 API key）

### 串接 Gemini API

在專案根目錄建立 `.env`（**不可推上 GitHub**）：

```
GEMINI_API_KEY=your_key_here
```

API key 取得方式：前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 建立 Free tier key。

> 若未設定 API key 或配額不足，介面自動切換為本地 SHAP 文字分析，不影響預測功能。

---

## 快速開始

建議使用 Python 3.12，Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\validate_data.py
jupyter lab
```

TabPFN 需要獨立 GPU 環境，另行安裝：

```powershell
python -m pip install -r requirements-tabpfn.txt
```

## 執行順序

模型分析（直接執行）：

```
notebooks/01_random_forest.ipynb
notebooks/02_logistic_regression.ipynb
notebooks/03_tabpfn.ipynb        ← 需 GPU 環境
notebooks/04_model_comparison.ipynb
```

重新抓資料與重建特徵：

```powershell
python scripts\scrape_games.py
python scripts\validate_data.py
python scripts\build_model_ready.py
```

欄位定義請讀 [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)。

## 未來展望

- **雲端 MLOps**：每日自動爬取最新賽況，推送至 Streamlit Cloud 或 HuggingFace Spaces
- **擴充高階數據**：納入擊球初速、球場效應（Park Factor）與比賽當天天氣
- **即時更新模型**：每賽季結束後重新 fine-tune，維持預測準確率
