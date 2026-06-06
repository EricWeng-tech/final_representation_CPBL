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
