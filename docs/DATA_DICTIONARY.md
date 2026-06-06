# 資料欄位說明

## 原始資料（data/raw/）

### games.csv — 每場比賽基本資料（3,130 場）

| 欄位 | 說明 |
|------|------|
| `game_id` | 比賽唯一識別碼 |
| `season_id` / `year` / `phase` | 賽季資訊 |
| `date` | 比賽日期（YYYY-MM-DD） |
| `home_team` / `away_team` | 主客隊名稱 |
| `home_runs` / `away_runs` | 最終得分 |
| `winner_side` | `HOME` / `AWAY` |
| `home_win` | **模型標籤**：1=主隊勝，0=客隊勝 |

### pitchers_box.csv — 投手逐場成績（28,141 筆）

一列 = 一位投手在一場比賽的出賽紀錄。

| 欄位 | 說明 |
|------|------|
| `game_id` / `side` | 比賽識別碼 / `home` 或 `away` |
| `is_starter` | 1=先發，0=後援 |
| `ERA` / `WHIP` / `FIP` | 賽季累計投球指標（season-to-date） |
| `IP_out` | 出局數（÷3 = 投球局數） |
| `ER` / `BB` / `SO` / `HR` | 自責分、保送、三振、被全壘打 |

### lineups.csv — 打線逐場成績（75,191 筆）

一列 = 一位打者在一場比賽的打擊紀錄。

| 欄位 | 說明 |
|------|------|
| `game_id` / `side` | 比賽識別碼 / `home` 或 `away` |
| `is_PH` | 1=代打，0=先發打者 |
| `batting_order` | 打序（1–9） |
| `AVG` / `OBP` / `SLG` / `OPS` | 賽季累計打擊指標（season-to-date） |
| `PA` / `AB` / `H` / `HR` / `RBI` | 打席、打數、安打、全壘打、打點 |

### team_game_logs.csv — 球隊逐場得失分（6,260 筆）

由 `games.csv` 衍生，每場比賽展開為兩列（主隊、客隊各一），用於計算近 10 場滾動統計。

| 欄位 | 說明 |
|------|------|
| `team` / `team_id` / `side` | 隊伍資訊 |
| `runs_scored` / `runs_allowed` | 本場得分 / 失分 |
| `win` | 1=勝，0=敗 |

---

## 建模主表（data/processed/）

### model_ready_games.csv（2,418 場 × 41 欄）

33 個模型特徵 + 8 個識別欄。所有特徵以 `shift(1)` 確保只用賽前資訊。

| 類別 | 特徵數 | 說明 |
|------|--------|------|
| 近 10 場滾動 | 6 | 主客隊勝率、得分、失分 |
| 先發投手 | 6 | 主客隊賽前 ERA / WHIP / FIP |
| 牛棚 | 4 | 主客隊平均賽前 ERA / WHIP |
| 先發打線 | 6 | 主客隊平均 OPS / OBP / SLG |
| 差值特徵 | 11 | 各類指標的 home − away |

---

## 33 個模型特徵完整說明

> 所有特徵皆以 `shift(1)` 處理，確保只使用**賽前已知資訊**，不含本場數據。

### 一、近 10 場滾動統計（6 個）

以球隊為單位，取前 10 場比賽的滾動平均（`shift(1).rolling(10, min_periods=1).mean()`）。

| 特徵名稱 | 來源欄位 | 說明 |
|----------|---------|------|
| `home_win_rate_10` | `team_game_logs.win` | 主隊近 10 場勝率（0–1）。反映近期整體競技狀態 |
| `away_win_rate_10` | `team_game_logs.win` | 客隊近 10 場勝率（0–1） |
| `home_runs_scored_10` | `team_game_logs.runs_scored` | 主隊近 10 場平均每場得分。反映打線近期火力 |
| `away_runs_scored_10` | `team_game_logs.runs_scored` | 客隊近 10 場平均每場得分 |
| `home_runs_allowed_10` | `team_game_logs.runs_allowed` | 主隊近 10 場平均每場失分。反映投手群近期穩定度 |
| `away_runs_allowed_10` | `team_game_logs.runs_allowed` | 客隊近 10 場平均每場失分 |

---

### 二、先發投手賽前指標（6 個）

以球員為單位，取**上一場出賽後**的賽季累計值（`shift(1)` by `player_id`）。僅納入 `is_starter=1` 的先發投手。

| 特徵名稱 | 原始欄位 | 說明 |
|----------|---------|------|
| `home_starter_ERA` | `pitchers_box.ERA` | 主隊先發投手賽季累計自責分率。數字越低越好；低於 3.00 屬優秀 |
| `away_starter_ERA` | `pitchers_box.ERA` | 客隊先發投手賽季累計自責分率 |
| `home_starter_WHIP` | `pitchers_box.WHIP` | 主隊先發每局被上壘率（安打＋保送）÷ 投球局數。低於 1.20 屬優秀 |
| `away_starter_WHIP` | `pitchers_box.WHIP` | 客隊先發每局被上壘率 |
| `home_starter_FIP` | `pitchers_box.FIP` | 主隊先發防守獨立投球指標（排除守備影響，只看三振/保送/全壘打）。比 ERA 更能預測未來表現 |
| `away_starter_FIP` | `pitchers_box.FIP` | 客隊先發防守獨立投球指標 |

---

### 三、牛棚平均賽前指標（4 個）

以球員為單位做 `shift(1)`，再對同場同側所有後援投手（`is_starter=0`）取平均。

| 特徵名稱 | 原始欄位 | 說明 |
|----------|---------|------|
| `home_bullpen_ERA` | `pitchers_box.ERA` | 主隊後援投手群賽季累計 ERA 平均值。牛棚越穩失分越少 |
| `away_bullpen_ERA` | `pitchers_box.ERA` | 客隊後援投手群賽季累計 ERA 平均值 |
| `home_bullpen_WHIP` | `pitchers_box.WHIP` | 主隊牛棚 WHIP 平均值。反映後援投手控球與被打能力 |
| `away_bullpen_WHIP` | `pitchers_box.WHIP` | 客隊牛棚 WHIP 平均值 |

---

### 四、先發打線平均賽前指標（6 個）

以球員為單位做 `shift(1)`，再對同場同側所有先發打者（`is_PH=0`）取平均。

| 特徵名稱 | 原始欄位 | 說明 |
|----------|---------|------|
| `home_lineup_OPS` | `lineups.OPS` | 主隊先發打線賽季累計 OPS 平均（上壘率＋長打率）。綜合打擊能力指標，.800 以上屬中上 |
| `away_lineup_OPS` | `lineups.OPS` | 客隊先發打線 OPS 平均 |
| `home_lineup_OBP` | `lineups.OBP` | 主隊先發打線賽季累計上壘率平均。反映製造上壘機會的能力 |
| `away_lineup_OBP` | `lineups.OBP` | 客隊先發打線上壘率平均 |
| `home_lineup_SLG` | `lineups.SLG` | 主隊先發打線賽季累計長打率平均。反映長打火力 |
| `away_lineup_SLG` | `lineups.SLG` | 客隊先發打線長打率平均 |

---

### 五、差值特徵（11 個）

全部為 `home − away`，**正值代表主隊優勢，負值代表客隊優勢**。
LR 模型只使用這 11 個差值特徵（避免與原始欄位產生完全共線性，VIF → ∞）。

| 特徵名稱 | 計算公式 | 解讀 |
|----------|---------|------|
| `win_rate_diff` | `home_win_rate_10 − away_win_rate_10` | 正值 = 主隊近期狀態較佳 |
| `runs_scored_diff` | `home_runs_scored_10 − away_runs_scored_10` | 正值 = 主隊近期攻擊力較強 |
| `run_diff_10` | `(home得分−home失分) − (away得分−away失分)` | 綜合近期得失分差距，正值 = 主隊淨優勢較大 |
| `starter_ERA_diff` | `home_starter_ERA − away_starter_ERA` | **負值**代表主隊先發更優（ERA 越低越好） |
| `starter_WHIP_diff` | `home_starter_WHIP − away_starter_WHIP` | **負值**代表主隊先發控球更佳 |
| `starter_FIP_diff` | `home_starter_FIP − away_starter_FIP` | **負值**代表主隊先發真實投球能力更強 |
| `lineup_OPS_diff` | `home_lineup_OPS − away_lineup_OPS` | 正值 = 主隊打線整體攻擊力較強 |
| `lineup_OBP_diff` | `home_lineup_OBP − away_lineup_OBP` | 正值 = 主隊打線上壘能力較佳 |
| `lineup_SLG_diff` | `home_lineup_SLG − away_lineup_SLG` | 正值 = 主隊打線長打能力較強 |
| `bullpen_ERA_diff` | `home_bullpen_ERA − away_bullpen_ERA` | **負值**代表主隊牛棚較穩 |
| `bullpen_WHIP_diff` | `home_bullpen_WHIP − away_bullpen_WHIP` | **負值**代表主隊牛棚控球較佳 |
