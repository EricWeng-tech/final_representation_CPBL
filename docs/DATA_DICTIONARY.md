# 中職（CPBL）勝負預測系統

> 此文件保留完整欄位定義。專案已改用分層目錄；最新執行方式請以根目錄 `README.md` 與 `docs/PROJECT_STRUCTURE.md` 為準。

**期末報告 — 資料分析與機器學習**

利用中華職棒 2018–2025 年的比賽資料，建立勝負預測模型（Random Forest + Logistic Regression + SHAP 特徵分析）。

> **目前實際流程：** 核心模型已拆分為 `01_random_forest.ipynb`、
> `02_logistic_regression.ipynb`、`03_tabpfn.ipynb` 與
> `04_model_comparison.ipynb`。本文後段仍保留部分舊版 `train_model.py`
> 說明供追溯；現行檔案分類請先閱讀 `docs/PROJECT_STRUCTURE.md`。

---

## 專案架構

```
資料來源（rebas.tw API）
        |
        v
  ┌─────────────────────────────────────────┐
  │  Step 1  scrape_all.py        │  爬球員賽季累計統計
  │  Step 2  scrape_games.py     │  爬逐場比賽 box score
  │  Step 3  validate_data.py    │  資料品質驗證（QC）
  │  Step 4  build_model_ready.py│  特徵工程 → 訓練資料
  │  Step 5  train_model.py      │  建模 + SHAP 分析
  └─────────────────────────────────────────┘
        |
        v
   model_ready_games.csv → 一列一場比賽，包含特徵與標籤
```

---

## 檔案說明

### 腳本

| 檔案 | 功能 |
|------|------|
| `scripts/scrape_all.py` | 爬取全部 29 個賽季的**球員賽季累計統計**（打擊／投球），輸出至 `cpbl_all_batters_*.csv` 與 `cpbl_all_pitchers_*.csv` |
| `scripts/scrape_games.py` | 爬取全部 29 個賽季的**逐場比賽記錄**，包含比分、先發投手、打線成績，輸出四個 CSV 檔 |
| `scripts/build_model_ready.py` | 以嚴格防洩漏策略合併四個 CSV → `data/processed/model_ready_games.csv`（2,418 場 × 41 欄） |
| `train_model.py` | 訓練 Random Forest + Logistic Regression，輸出評估報告、特徵重要性圖、SHAP 圖、預測 CSV |

> `scripts/validate_data.py`：資料品質驗證腳本（57 項檢查，PASS 57 / WARN 8 / FAIL 0），供開發者確認原始資料乾淨度，不屬於主流程。

### 資料檔（執行爬蟲後產生）

| 檔案 | 說明 | 筆數 |
|------|------|------|
| `cpbl_all_batters_*.csv` | 各賽季各球員打擊累計統計 | 3,132 筆 |
| `cpbl_all_pitchers_*.csv` | 各賽季各球員投球累計統計 | 2,638 筆 |
| `games.csv` | 每場比賽基本資料（比分、主客隊、球場、勝敗） | 3,130 場 |
| `pitchers_box.csv` | 每場每位投手出賽成績（含先發/後援標記） | 28,141 筆 |
| `lineups.csv` | 每場打線逐人成績 | 75,191 筆 |
| `team_game_logs.csv` | 每隊每場得分/失分，用於計算近10場滾動勝率 | 6,260 筆 |
| `model_ready_games.csv` | 特徵工程後的建模主表，一列一場比賽 | 2,418 場 × 41 欄 |

### 模型輸出（執行 train_model.py 後產生）

| 檔案 | 說明 |
|------|------|
| `predictions.csv` | 測試集（2025）每場預測結果，含 RF 與 LR 機率與正確與否 |
| `feature_importance_rf.png` | Random Forest Top-20 特徵重要性長條圖 |
| `shap_summary.png` | SHAP Beeswarm 總覽圖（測試集） |

---

## 執行環境

```bash
# 建立虛擬環境（第一次使用）
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # macOS / Linux

# 安裝套件
pip install requests pandas scikit-learn matplotlib shap
```

---

## 如何執行

### Step 1 — 爬球員統計（約 5 分鐘）

```bash
python scripts/scrape_all.py
```

輸出：`cpbl_all_batters_YYYYMMDD_HHMMSS.csv`、`cpbl_all_pitchers_YYYYMMDD_HHMMSS.csv`

### Step 2 — 爬逐場比賽記錄（約 90 分鐘，3130 場 × 0.25s）

```bash
python scripts/scrape_games.py
```

輸出：`games.csv`、`pitchers_box.csv`、`lineups.csv`、`team_game_logs.csv`

> 每場比賽需額外 API 呼叫取得 box score，請確保網路穩定。
> 原始 JSON 備份存於 `data/cache/games_raw/`（已加入 .gitignore）。

### Step 3 — 資料品質驗證（開發用，可略過）

```bash
python scripts/validate_data.py
```

結果：PASS 57 / WARN 8 / FAIL 0（所有 WARN 均為已確認的正常情況）

### Step 4 — 特徵工程

```bash
python scripts/build_model_ready.py
```

輸出：`model_ready_games.csv`（2,418 場 × 41 欄）

### Step 5 — 建模與評估

```bash
python train_model.py
```

輸出：`predictions.csv`、`feature_importance_rf.png`、`shap_summary.png`

---

## 腳本說明

---

### `scrape_all.py` — 爬球員賽季統計

**前提：** 要預測一場比賽的勝負，必須知道上場球員「當下的整體實力」。這份資料提供每位球員在整個賽季的累計成績，是建立球員基礎能力指標的來源。

**做什麼：**
1. 查詢 29 個賽季的所有隊伍清單
2. 對每支球隊呼叫 API，取得該隊所有球員的賽季統計
3. 輸出打者與投手各一份 CSV

**輸入：** 無（直接呼叫 rebas.tw API）
**輸出：** `cpbl_all_batters_*.csv`、`cpbl_all_pitchers_*.csv`

---

### `scrape_games.py` — 爬逐場比賽記錄

**前提：** 光有球員的賽季累計數字還不夠——我們需要知道「這場比賽的當下狀態」：誰先發上場、打線誰出賽、當時的賽季累計 ERA 是多少。這份資料是特徵工程的核心原料。

**做什麼：**
1. 取得每個賽季的比賽列表，篩選出已完賽（`FINISHED`）的場次
2. 對每場比賽額外呼叫一次 API，取得完整的 box score（投手與打者逐人成績）
3. 輸出四個 CSV，分別對應不同粒度的資料（場次、投手、打者、球隊）

**輸入：** 無（直接呼叫 rebas.tw API）
**輸出：** `games.csv`、`pitchers_box.csv`、`lineups.csv`、`team_game_logs.csv`
**耗時：** 約 90 分鐘（3,130 場 × 每場 0.25 秒間隔）

---

---

### `build_model_ready.py` — 特徵工程

**前提：** 預測比賽勝負只能用「開打前已知」的資訊，不能用本場結果。若直接把 ERA、OPS 等賽季累計值作為特徵，因為這些值已包含本場表現，會造成資料洩漏（data leakage），導致模型在訓練時「偷看」了答案。

**防洩漏策略：**
- 各球員的 ERA/WHIP/FIP/OPS：依 `player_id` 排序後做 `shift(1)`，取上一場出賽後的賽季累計值
- 球隊近10場滾動統計：依 `team_id` 排序後做 `shift(1).rolling(10)`，同樣排除本場

**輸入：** `games.csv`、`pitchers_box.csv`、`lineups.csv`、`team_game_logs.csv`
**輸出：** `model_ready_games.csv`（2,418 場 × 41 欄，整體特徵缺失率 1.99%）

---

### `train_model.py` — 建模與評估

**做什麼：**
1. 讀取 `model_ready_games.csv`，以年份切分訓練集（2018–2024）與測試集（2025）
2. 中位數填補缺失值（SimpleImputer），LR 另加 StandardScaler
3. 訓練 Random Forest（n_estimators=300, max_depth=8）與 Logistic Regression（C=0.1, L2）
4. 輸出 Accuracy、AUC、Classification Report、混淆矩陣
5. 產生 RF 特徵重要性圖、SHAP Beeswarm 圖、逐場預測 CSV

**輸入：** `model_ready_games.csv`
**輸出：** `predictions.csv`、`feature_importance_rf.png`、`shap_summary.png`

---

## 資料欄位說明

> **▲ 本場成績**：僅計算這一場的表現，每場比賽結束後歸零重算。
> **★ 賽季累計值**：該球員截至本場比賽結束後的當季總累計，不是單場數字。例如投手本場沒有失分，但 ERA 欄位顯示 3.20，代表他整個賽季到目前為止的累計 ERA 是 3.20。

---

### games.csv — 比賽基本資料

**前提：** 這是整個系統的「主表」。每一列代表一場比賽，包含比賽的時間、地點、雙方隊伍、比分與勝負結果。`home_win` 欄位是模型的**預測目標（標籤）**，其他所有 CSV 的資料最終都會依 `game_id` 對應回這張表。

**一列 = 一場比賽**

#### 識別與時間

| 欄位 | 類型 | 說明 |
|------|------|------|
| `game_id` | 字串 | 比賽唯一識別碼（來自 rebas.tw） |
| `season_id` | 字串 | 賽季識別碼，例如 `CPBL-2025-JO` |
| `year` | 整數 | 年份，例如 `2025` |
| `phase` | 字串 | 賽制，例如 `一軍`（例行賽）、`一軍-2`（季後賽） |
| `group` | 字串 | 賽制細分（上半季／下半季／總冠軍賽等） |
| `date` | 字串 | 比賽日期，格式 `YYYY-MM-DD` |
| `started_at` | 字串 | 實際開賽時間（ISO 8601） |
| `ended_at` | 字串 | 比賽結束時間 |

#### 場地

| 欄位 | 類型 | 說明 |
|------|------|------|
| `location` | 字串 | 球場名稱，例如「天母棒球場」 |
| `innings` | 整數 | 實際比賽局數（正規 9 局；縮短賽 5–8 局） |
| `audience` | 整數 | 觀眾人數 |

#### 比賽結果

| 欄位 | 類型 | 說明 |
|------|------|------|
| `status` | 字串 | 比賽狀態：`FINISHED`（完賽）、`CANCELLED`（取消） |
| `finished_status` | 字串 | 結束方式：正常結束為空值；異常（雨延中止等）會有標記 |
| `home_team` / `away_team` | 字串 | 主場／客場隊伍名稱 |
| `home_team_id` / `away_team_id` | 字串 | 隊伍識別碼 |
| `home_runs` / `away_runs` ▲ | 整數 | 主客隊**最終得分** |
| `home_hits` / `away_hits` ▲ | 整數 | 主客隊總安打數 |
| `home_errors` / `away_errors` ▲ | 整數 | 主客隊失誤數 |
| `home_scores` / `away_scores` ▲ | 字串 | 各局得分（逗號分隔），例如 `0,1,0,2,0,0,0,1,0` |
| `winner_side` | 字串 | 獲勝方：`HOME`、`AWAY`、`''`（平局） |
| `home_win` | 整數 | **模型標籤**：`1`=主隊勝，`0`=客隊勝，`''`=平局或未完賽 |

---

### pitchers_box.csv — 投手逐場成績

**前提：** 投手是影響比賽勝負最關鍵的單一因素。這張表記錄每場比賽每位投手的出賽數據。最重要的用途是：透過 `is_starter=1` 篩出先發投手，取得他當場的賽季累計 ERA、WHIP、FIP，作為「先發投手當下狀態」的特徵。

**一列 = 一位投手在一場比賽的出賽紀錄**（一場比賽通常有 4–8 位投手，主客各算）

#### 識別

| 欄位 | 類型 | 說明 |
|------|------|------|
| `game_id` | 字串 | 對應 games.csv 的比賽識別碼 |
| `side` | 字串 | `home`（主場投手）/ `away`（客場投手） |
| `order` | 整數 | 出場順序（1=先發，2=第一位後援，依此類推） |
| `is_starter` | 整數 | `1`=先發投手，`0`=後援投手 |
| `player_id` | 字串 | 球員識別碼 |
| `name` | 字串 | 球員姓名 |
| `number` | 字串 | 球衣號碼 |

#### 本場投球成績 ▲

| 欄位 | 類型 | 說明 |
|------|------|------|
| `IP_out` | 整數 | 本場**出局數**（投球局數 = IP_out ÷ 3，例如 7 出局 = 2⅓ 局） |
| `NP` | 整數 | 本場**投球數**（Pitch Count）。先發通常 80–120，超過 130 已算高 |
| `BF` | 整數 | 本場**面對打者數**（Batters Faced）。包含所有打席（安打、保送、出局等） |
| `H` ▲ | 整數 | 被安打數 |
| `HR` ▲ | 整數 | 被全壘打數 |
| `BB` ▲ | 整數 | 四壞保送數 |
| `SO` ▲ | 整數 | 三振數 |
| `ER` ▲ | 整數 | **自責分**（Earned Runs）：扣除因失誤造成的得分後，應由投手負責的失分 |
| `R` ▲ | 整數 | 總失分（含非自責分） |
| `HB` ▲ | 整數 | 觸身球（Hit By Pitch）數 |
| `WPA` ▲ | 小數 | **Win Probability Added**：本場投球對球隊勝率的貢獻值（正=有利，負=不利） |

#### 賽季累計投球指標 ★

> 以下數值為該球員**截至本場比賽為止的當季累計**，而非本場單場數字。
> 適合直接用於特徵工程，代表投手「當下狀態」。

| 欄位 | 公式 | 說明 |
|------|------|------|
| `ERA` ★ | `9 × 自責分 / 投球局數` | **自責分率**（Earned Run Average）。數字越低越好。聯盟平均約 3.5–4.5；低於 3.00 屬優秀 |
| `WHIP` ★ | `(被安打 + 四壞球) / 投球局數` | **每局被上壘率**。越低越好；低於 1.20 屬優秀，超過 1.50 代表控球不穩 |
| `FIP` ★ | `(13×HR + 3×(BB+HB) − 2×SO) / IP + 常數` | **防守獨立投球指標**（Fielding Independent Pitching）。排除守備影響，只看投手可控制的結果（三振、保送、全壘打）。比 ERA 更能預測未來表現 |

---

### lineups.csv — 打線逐場成績

**前提：** 知道當天誰上場打擊、他們當時的狀態如何，是預測得分能力的關鍵。這張表記錄每場比賽每位打者的出賽數據。最重要的用途是：計算主客隊「當日實際上場打線」的平均 OPS，代表打線的整體攻擊強度。

**一列 = 一位打者在一場比賽的打擊紀錄**（一場比賽通常有 9–15 位打者，主客各算）

#### 識別

| 欄位 | 類型 | 說明 |
|------|------|------|
| `game_id` | 字串 | 對應 games.csv 的比賽識別碼 |
| `side` | 字串 | `home` / `away` |
| `batting_order` | 整數 | 打序（1–9）；代打者的打序繼承前任打者 |
| `is_PH` | 整數 | `1`=代打（Pinch Hitter），`0`=正規先發打者 |
| `player_id` | 字串 | 球員識別碼 |
| `name` | 字串 | 球員姓名 |
| `number` | 字串 | 球衣號碼 |

#### 本場打擊成績 ▲

| 欄位 | 類型 | 說明 |
|------|------|------|
| `PA` ▲ | 整數 | **打席數**（Plate Appearances）：每次站上打擊區即計一個打席，包含保送、觸身球、犧牲打等 |
| `AB` ▲ | 整數 | **打數**（At Bats）：PA 中扣除四壞球、觸身球、犧牲打、高飛犧牲打。`AB ≤ PA` |
| `H` ▲ | 整數 | **安打數**（Hits）：包含單打、二壘打、三壘打、全壘打 |
| `2B` ▲ | 整數 | **二壘打**（Doubles） |
| `3B` ▲ | 整數 | **三壘打**（Triples） |
| `HR` ▲ | 整數 | **全壘打**（Home Runs）。`2B + 3B + HR ≤ H` |
| `RBI` ▲ | 整數 | **打點**（Runs Batted In）：因該打者打擊而得分的跑者數 |
| `R` ▲ | 整數 | **得分**（Runs Scored）：該打者本人跑回本壘的次數 |
| `BB` ▲ | 整數 | **四壞球保送**（Walks / Base on Balls） |
| `SO` ▲ | 整數 | **三振**（Strikeouts） |
| `HBP` ▲ | 整數 | **觸身球**（Hit By Pitch）：被投手投球擊中而上壘 |
| `SB` ▲ | 整數 | **盜壘成功**（Stolen Bases） |
| `CS` ▲ | 整數 | **盜壘失敗**（Caught Stealing） |
| `WPA` ▲ | 小數 | **Win Probability Added**：本場打擊對球隊勝率的貢獻值 |

#### 賽季累計打擊指標 ★

> 以下數值為該球員**截至本場比賽為止的當季累計**，適合直接作為特徵使用。

| 欄位 | 公式 | 說明 |
|------|------|------|
| `AVG` ★ | `H / AB` | **打擊率**（Batting Average）。介於 0–1；聯盟平均約 .260；.300 以上屬優秀打者 |
| `OBP` ★ | `(H + BB + HBP) / (AB + BB + HBP + SF)` | **上壘率**（On-Base Percentage）。比打擊率更完整，包含保送與觸身球。聯盟平均約 .330；超過 .370 屬優秀 |
| `SLG` ★ | `(1B + 2×2B + 3×3B + 4×HR) / AB` | **長打率**（Slugging Percentage）。衡量打者的長打能力，全壘打貢獻 4 倍。聯盟平均約 .400 |
| `OPS` ★ | `OBP + SLG` | **整體攻擊指數**（On-base Plus Slugging）。最常用的綜合打擊指標。低於 .700 偏弱；.800 屬中上；.900 以上屬明星等級 |

---

### team_game_logs.csv — 球隊逐場得失紀錄

**前提：** 「近期狀態」是預測勝負的重要因素——一支連贏 8 場的球隊，跟連輸 5 場的球隊，面對同一個對手時勝率完全不同。這張表的唯一目的是讓我們能夠快速計算「每支球隊在任意比賽前的近 N 場滾動勝率、平均得分、平均失分」。

**一列 = 一支球隊在一場比賽的結果**（`games.csv` 的每一場比賽在這裡對應兩列：主隊一列、客隊一列）

| 欄位 | 類型 | 說明 |
|------|------|------|
| `game_id` | 字串 | 對應 games.csv 的比賽識別碼 |
| `season_id` | 字串 | 賽季識別碼 |
| `year` | 整數 | 年份 |
| `phase` | 字串 | 賽制 |
| `date` | 字串 | 比賽日期 |
| `group` | 字串 | 賽制細分 |
| `team` | 字串 | 本列代表的隊伍名稱 |
| `team_id` | 字串 | 隊伍識別碼 |
| `side` | 字串 | `home`（主場）/ `away`（客場） |
| `opponent` | 字串 | 對手隊伍名稱 |
| `runs_scored` ▲ | 整數 | 本隊本場得分 |
| `runs_allowed` ▲ | 整數 | 本隊本場失分（= 對手 `runs_scored`） |
| `hits` ▲ | 整數 | 本隊本場安打數 |
| `errors` ▲ | 整數 | 本隊本場失誤數 |
| `win` ▲ | 整數 | `1`=本場勝，`0`=本場敗，`''`=平局或未完賽 |

---

### cpbl_all_batters_*.csv — 球員賽季累計打擊統計

**前提：** 這是球員的「歷史底稿」，記錄每位球員在每個賽季結束時的完整成績。目前用途是背景參考；在特徵工程中，實際用的是 `lineups.csv` 裡的 season-to-date 數值（因為那才代表比賽當天的狀態）。

**一列 = 一位打者在一個賽季的全季累計成績**（3,132 筆 = 29 賽季 × 平均約 108 位打者）

| 欄位 | 說明 |
|------|------|
| `season_id` / `year` / `phase` | 賽季資訊 |
| `team` / `team_id` | 所屬球隊 |
| `player_id` / `name` / `number` | 球員識別碼、姓名、號碼 |
| `games` | 出賽場數 |
| `PA` / `AB` / `H` / `HR` / `RBI` | 打席、打數、安打、全壘打、打點（同上定義） |
| `BB` / `SO` / `HBP` / `SB` / `CS` | 保送、三振、觸身球、盜壘成/敗 |
| `AVG` / `OBP` / `SLG` / `OPS` | 四大打擊指標（同上定義） |
| `OPSplus` | **OPS+**：以 100 為基準的聯盟調整版 OPS，超過 100 代表優於聯盟平均 |
| `BABIP` | **球場安打率**（Batting Average on Balls In Play）：`(H - HR) / (AB - SO - HR + SF)`，衡量打者運氣成分，聯盟平均約 .300 |
| `ISO` | **純長打率**（Isolated Power）：`SLG - AVG`，衡量長打能力 |
| `wOBA` | **加權上壘率**（Weighted On-Base Average）：依安打類型加權，比 OPS 更精確的攻擊指標 |
| `WPA` | **Win Probability Added**：全季打擊對球隊勝率的總貢獻 |
| `BBp` | **保送率**（Walk%）：`BB / PA`，衡量選球能力 |
| `Kp` | **三振率**（Strikeout%）：`SO / PA` |

---

### cpbl_all_pitchers_*.csv — 球員賽季累計投球統計

**前提：** 與打者底稿相同，是投手的歷史背景資料。特徵工程實際使用的是 `pitchers_box.csv` 裡的 season-to-date ERA/WHIP/FIP，因為那代表投手在該場比賽當下的賽季狀態。

**一列 = 一位投手在一個賽季的全季累計成績**（2,638 筆 = 29 賽季 × 平均約 91 位投手）

| 欄位 | 說明 |
|------|------|
| `role` | `starter`（先發）/ `reliever`（後援），由 `SP > 0` 判斷 |
| `games` / `SP` | 出賽場數 / 先發場數 |
| `W` / `L` / `SV` / `HLD` / `BS` | 勝/敗/救援成功/中繼成功/救援失敗 |
| `IP_out` | 總出局數（÷3 = 總投球局數） |
| `ERA` / `WHIP` / `FIP` | 三大投球指標（同上定義） |
| `ERAplus` | **ERA+**：以 100 為基準的聯盟調整版 ERA，超過 100 代表優於聯盟平均 |
| `SO9` (K9) | **每9局三振數**：`SO × 9 / IP`，衡量三振能力 |
| `BB9` | **每9局保送數**：`BB × 9 / IP`，衡量控球能力 |
| `HR9` | **每9局被全壘打數** |
| `LOBp` | **殘壘率**（Left on Base%）：`(H + BB - R) / (H + BB - 1.4×HR)`，數字越高代表危機處理能力越好 |
| `GBp` | **滾地球率**（Ground Ball%）：滾地球出局 / 所有出局，高滾地率投手較不易被打全壘打 |
| `BABIP` | **球場安打率**（投手版）：衡量防守與運氣對投手成績的影響 |
| `WPA` | **Win Probability Added**：全季投球對球隊勝率的總貢獻 |

---

## 賽季涵蓋範圍驗證

### 一軍例行賽場次與隊伍數

| 年份 | 隊數 | 例行賽場次 | 理論場次 | 備註 |
|------|------|-----------|---------|------|
| 2018 | 4 | 240 | 240 | Lamigo、中信兄弟、富邦悍將、統一7-ELEVEn獅 |
| 2019 | 4 | 240 | 240 | 同上（Lamigo 末季） |
| 2020 | 4 | 240 | 240 | Lamigo 更名為**樂天桃猿** |
| 2021 | 5 | 300 | 300 | **味全龍**復隊加入 |
| 2022 | 5 | 300 | 300 | |
| 2023 | 5 | 300 | 300 | ※ rebas.tw 標記為 `一軍-3`，非 `一軍` |
| 2024 | 6 | 360 | 360 | **台鋼雄鷹**加入 |
| 2025 | 6 | 360 | 360 | |
| 2026 | 6 | 進行中 | 364 | 截至 2026-05-25 完賽 115 場 |

> 理論場次公式：`C(隊數, 2) × 每對對戰場數`
> - 4 隊：C(4,2)×40 = 240；5 隊：C(5,2)×30 = 300；6 隊：C(6,2)×24 = 360

### 附屬賽季說明（非例行賽）

| 年份 | phase 標籤 | 場次 | 推測賽事 |
|------|-----------|------|---------|
| 2021 | 一軍-2 | 4 | 台灣大賽（4 場定勝負） |
| 2022 | 一軍-2、一軍-3 | 4、3 | 全明星賽、亞錦賽選拔 |
| 2022 | 一軍-4 | 15 | 季後賽 + 台灣大賽 |
| 2023 | 一軍 | 7 | 開幕賽 |
| 2023 | 一軍-2 | 3 | 全明星賽 |
| 2023 | 一軍-4 | 29 | 季後賽 + 台灣大賽 |
| 2024 | 一軍-2、一軍-3 | 5、3 | 開幕賽、全明星賽 |
| 2024 | 一軍-4 | 30 | 季後賽 + 台灣大賽 |
| 2025 | 一軍-2、一軍-3、一軍-4 | 5、4、2 | 小型邀請賽 |
| 2025 | 一軍-5 | 30 | 季後賽 + 台灣大賽 |
| 2026 | 一軍-2 | 30 | 台灣大賽（已完賽） |

> 建模時只保留**一軍例行賽**（`phase='一軍'`，2023 年例外為 `一軍-3`），附屬賽季不納入訓練。

---

## 建模流程狀態

| 步驟 | 腳本 | 狀態 |
|------|------|------|
| 1 | `scrape_all.py` | 完成 |
| 2 | `scrape_games.py` | 完成 |
| 3 | `validate_data.py` | 完成（PASS 57 / WARN 8 / FAIL 0） |
| 4 | `build_model_ready.py` | 完成（2,418 場 × 41 欄） |
| 5 | `train_model.py` | 腳本完成，尚未執行 |

---

## 資料來源

- **rebas.tw** — 中華職棒統計資料平台，提供公開 JSON API
- 涵蓋範圍：2018–2026 年，一軍 + 二軍，共 29 個賽季
- 所有爬取行為遵循 0.25 秒間隔（`time.sleep(0.25)`），避免對伺服器造成負擔
