# 模型假設與前處理對應表

## 一、Random Forest（隨機森林）

樹狀模型為非參數方法，對資料分布幾乎沒有前提假設。

| 假設 / 條件 | 是否需要 | 本專案處理方式 |
|------------|---------|--------------|
| 特徵標準化 | **不需要** | 跳過 |
| 多元共線性（VIF）檢查 | **不需要** | 跳過（樹模型以分裂點選特徵，共線性不影響結果） |
| 常態分布假設 | **不需要** | 跳過 |
| 缺失值處理 | **需要** | `SimpleImputer(strategy='median')`，僅 fit 訓練集，再 transform 測試集（防資料洩漏） |

---

## 二、Logistic Regression（邏輯迴歸）

線性模型對特徵有較多前提假設，需逐項確認。

| 假設 / 條件 | 是否需要 | 本專案處理方式 |
|------------|---------|--------------|
| 特徵標準化 | **需要** | `StandardScaler`（封裝於 Pipeline，fit 訓練集） |
| 多元共線性（VIF）檢查 | **需要** | 詳見下方說明 |
| 缺失值處理 | **需要** | `SimpleImputer(strategy='median')` |
| 常態分布假設 | 不需要 | LR 對輸入分布無假設（假設的是 log-odds 與特徵的線性關係） |
| 二元標籤 | 天然滿足 | `home_win` ∈ {0, 1} |
| 觀測值獨立 | 假設滿足 | 每場比賽視為獨立事件 |

---

## 三、VIF 多元共線性問題說明

### 問題來源

本資料集特徵同時包含：
- **原始特徵**：`home_starter_ERA`、`away_starter_ERA`
- **差值特徵**：`starter_ERA_diff = home_starter_ERA − away_starter_ERA`

三者之間存在**完全線性相依**，VIF → ∞，導致 LR 係數不穩定、預測能力崩潰。

### 兩種常見處理方式

| 方式 | 做法 | 優點 | 缺點 |
|------|------|------|------|
| **方法一：VIF 門檻篩選** | 計算每個特徵的 VIF，刪除 VIF > 10 的特徵，反覆迭代直到全部 VIF < 10 | 客觀、自動化 | 可能刪掉有意義的特徵；迭代次數不確定 |
| **方法二：領域知識選特徵** | 直接只保留差值特徵（11 個 `_diff`），捨棄原始 home/away 欄位 | 直觀、可解釋；每個 diff 已濃縮主客差距資訊 | 需要對特徵意義有了解 |

### 本專案採用：方法二

保留 11 個差值特徵：

```
win_rate_diff       runs_scored_diff    run_diff_10
starter_ERA_diff    starter_WHIP_diff   starter_FIP_diff
lineup_OPS_diff     lineup_OBP_diff     lineup_SLG_diff
bullpen_ERA_diff    bullpen_WHIP_diff
```

篩選後重新計算 VIF，確認無高共線性特徵後才進入 LR 訓練。

---

## 四、前處理流程對比

```
原始資料（model_ready_games.csv）
        │
        ├─ 訓練集 / 測試集切分（依年份，無 shuffle）
        │
        ├─ [共用] SimpleImputer(median) — fit on train only
        │
        ├─ Random Forest 路徑
        │       └─ 直接訓練（無需 VIF / 標準化）
        │
        └─ Logistic Regression 路徑
                ├─ VIF 檢查（全 33 特徵）→ 發現共線性
                ├─ 特徵篩選（只保留 11 個 diff 特徵）
                ├─ VIF 再確認
                ├─ StandardScaler（fit on train diff only）
                └─ LR 訓練
```
