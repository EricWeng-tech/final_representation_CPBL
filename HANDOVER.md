# HANDOVER — CPBL 賽前勝負預測系統

最後更新：2026-06-07。此文件供下一個 agent 或自己繼續任務使用。

---

## 一、專案現況

| 項目 | 狀態 |
|------|------|
| Git repo | 本機，未推 GitHub（Jay 指示暫停推送） |
| 模型 | `outputs/models/tuned_random_forest_gui.joblib`（33 特徵，RF Tuned） |
| Streamlit app | `scripts/streamlit_app.py`（已完成，全面檢查通過） |
| Gemini API key | 存於 `.env`（已加入 `.gitignore`） |
| Python 環境 | 需用 `.venv`（Python 3.x + sklearn 1.8.0），**不可用** 系統 Python 3.14（sklearn DLL 破損） |

---

## 二、已完成任務

1. **GitHub 同步** — 舊 39 特徵 commit 已 force-push 移除，feature/woba-wrcplus 已 merge 入 main
2. **DATA_DICTIONARY.md** — 33 個特徵完整中文說明，含 5 分類（push 至 main，commit 750e8e8）
3. **README.md** — 新增 RF Fine-Tuning 表格、Confidence Threshold 表格、防資料洩漏策略、未來展望
4. **rf_confidence_metrics.csv** — 從 git 歷史 71651c3 復原至 `outputs/experiments/rf_tuning/`
5. **Streamlit web app** — 完整功能：球隊選擇 + VS 展示、先發/牛棚/打線下拉、指標對比表、預測 + SHAP bar chart + Gemini AI 白話分析
6. **球隊 logo** — PNG 已放至 `assets/team_logos/<隊名>.png`（6 隊），無 PNG 自動 fallback 為官方隊色圓形 div
7. **套件安裝** — `.venv` 已有 streamlit / sklearn 1.8.0 / shap / google-generativeai / python-dotenv
8. **Auto-fill bug 修復** — 選擇先發投手後，ERA/WHIP/FIP 正確同步至文字輸入框（`inp_` widget key）
9. **Gemini cascade** — 2.0-flash 失敗自動切 2.5-flash，再失敗切本地 SHAP 文字分析
10. **Caption 修正** — 準確率 70.39%、訓練資料 2018–2024（1,947 場）
11. **亮色主題 + CPBL 背景** — `.streamlit/config.toml` light base，CPBL.png 帶半透明遮罩

---

## 三、待辦事項

- [ ] 簡報 PDF 需更正問題（見下方 §五）
- [ ] 若要推 GitHub：確認 `.env` 在 `.gitignore`，不得帶入 API key

---

## 四、啟動方式

```powershell
# 在 期末報告/ 目錄下
cd "C:\Users\Eric\Desktop\python資料分析與機器學習\期末報告"
.\.venv\Scripts\streamlit.exe run scripts\streamlit_app.py
```

瀏覽器開 `http://localhost:8501`，關閉按 `Ctrl+C`。

---

## 五、簡報需修正問題

| 頁 | 問題 | 正確內容 |
|----|------|---------|
| P.8/9/10 | 頁面空白 | 需補內容（未知是否故意） |
| P.16 | Walk-Forward Fold 4 訓練場數顯示 1583 | 應為 **1,589** |
| P.16 | Walk-Forward 只有 4 Fold | 需加 **Fold 5**（2025 驗證，Train 1,947） |
| P.20 | 列出 K/9 為投手指標 | 模型實際使用 **FIP**，K/9 從未在特徵中 |
| P.21 | 宣稱「SHAP 動態模擬」 | Streamlit 版 **已實作 SHAP**，可改為「已實作」 |

---

## 六、關鍵資料路徑

```
data/processed/model_ready_games.csv      建模主表（2,418 場 × 41 欄）
data/raw/pitchers_box.csv                  投手逐場（含 ERA/WHIP/FIP）
data/raw/lineups.csv                       打線逐場（含 OPS/OBP/SLG）
data/raw/team_game_logs.csv               球隊近 10 場滾動原始資料
outputs/models/tuned_random_forest_gui.joblib   Pipeline(SimpleImputer + RF)
outputs/experiments/rf_tuning/rf_confidence_metrics.csv
outputs/metrics/rf_walkforward_5fold.csv
assets/team_logos/                         6 隊 PNG logo
.streamlit/config.toml                     亮色主題設定
```

---

## 七、核心程式架構

```
scripts/
  gui_predict.py          全部資料載入 / 特徵建構 / 模型推論函式（tkinter 前端，勿更動）
  train_gui_model.py      重新訓練模型（import from gui_predict）
  streamlit_app.py        Web app（import from gui_predict，重用全部函式）
```

`streamlit_app.py` 透過 `sys.path.insert(0, ...)` 直接 import `gui_predict`，
所有 `INPUT_FIELDS / build_manual_prediction_row / add_diff_features` 等共用。

---

## 八、SHAP 實作細節

```python
imputer = model.named_steps["imputer"]
rf      = model.named_steps["rf"]
x_tr    = imputer.transform(x_pred)
explainer = shap.TreeExplorer(rf)
shap_vals = explainer.shap_values(x_tr)

# version-safe 取 class-1 SHAP
if isinstance(shap_vals, list): sv = shap_vals[1][0]
elif shap_vals.ndim == 3:       sv = shap_vals[0, :, 1]
else:                           sv = shap_vals[0]
```

圖表顏色：RdBu_r colormap（紅＝有利主隊，藍＝有利客隊），依 SHAP 絕對值強度決定深淺。

---

## 九、Gemini API

- 套件：`google-generativeai`（FutureWarning 可忽略，仍可用）
- Key：`.env` → `GEMINI_API_KEY`
- **串接順序**：`gemini-2.0-flash` → 失敗 → `gemini-2.5-flash` → 失敗 → 本地 SHAP fallback
- 注意：`gemini-2.0-flash` 在此帳號 free tier 下 limit=0（無法使用），實際由 `gemini-2.5-flash` 處理
- 若 key 未設定，app 靜默跳過，不會 crash
