"""
從原始 CSV 建立無資料洩漏的模型訓練特徵表 model_ready_games.csv

防洩漏策略：
  rolling stats       → shift(1) 排除本場，取前10場均值
  投手 ERA/WHIP/FIP   → 該投手上一場出賽後的 season-to-date 值
  打者 OPS/OBP/SLG    → 同上
  絕不使用本場得分、安打、失誤、WPA、IP/ER/SO、H/HR/RBI
"""

import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'


# ─────────────────────────── 讀取 ───────────────────────────

def load_csvs():
    games    = pd.read_csv(RAW_DATA_DIR / 'games.csv',          dtype=str)
    logs     = pd.read_csv(RAW_DATA_DIR / 'team_game_logs.csv', dtype=str)
    pitchers = pd.read_csv(RAW_DATA_DIR / 'pitchers_box.csv',   dtype=str)
    lineups  = pd.read_csv(RAW_DATA_DIR / 'lineups.csv',        dtype=str)
    return games, logs, pitchers, lineups


# ─────────────────────────── 篩選比賽 ───────────────────────

def filter_games(games):
    df = games.copy()
    df['home_win'] = pd.to_numeric(df['home_win'], errors='coerce')
    # 2023 例行賽在 rebas.tw 被標記為「一軍-3」，其餘年份為「一軍」
    regular = (df['phase'] == '一軍') | \
              ((df['year'] == '2023') & (df['phase'] == '一軍-3'))
    df = df[
        regular &
        (df['status'] == 'FINISHED') &
        df['home_win'].isin([0.0, 1.0])
    ].copy()
    df['home_win'] = df['home_win'].astype(int)
    return df


# ─────────────────── 近10場滾動特徵（team_game_logs） ───────

def build_rolling_stats(logs):
    df = logs.copy()
    for col in ['win', 'runs_scored', 'runs_allowed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values(['team_id', 'date', 'game_id'])

    for src, dst in [
        ('win',          'rolling_win_rate'),
        ('runs_scored',  'rolling_runs_scored'),
        ('runs_allowed', 'rolling_runs_allowed'),
    ]:
        df[dst] = df.groupby('team_id')[src].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )

    return df[['game_id', 'side',
               'rolling_win_rate', 'rolling_runs_scored', 'rolling_runs_allowed']]


# ─────────────── 共用：對每位球員逐場 shift(1) ──────────────

def _shift_per_player(df, stat_cols):
    df = df.sort_values(['player_id', 'date', 'game_id'])
    for col in stat_cols:
        df[f'pre_{col}'] = df.groupby('player_id')[col].shift(1)
    return df


# ─────────────── 先發投手賽前 ERA/WHIP/FIP ─────────────────

def build_starter_stats(pitchers):
    df = pitchers.copy()
    df['is_starter'] = pd.to_numeric(df['is_starter'], errors='coerce')
    st = df[df['is_starter'] == 1].copy()

    for col in ['ERA', 'WHIP', 'FIP']:
        st[col] = pd.to_numeric(st[col], errors='coerce')

    st = _shift_per_player(st, ['ERA', 'WHIP', 'FIP'])

    # 安全去重：理論上每場每側唯一，防萬一
    st = st.drop_duplicates(subset=['game_id', 'side'], keep='first')
    return st[['game_id', 'side', 'pre_ERA', 'pre_WHIP', 'pre_FIP']]


# ─────────────── 牛棚賽前 ERA/WHIP（聚合） ─────────────────

def build_bullpen_stats(pitchers):
    df = pitchers.copy()
    df['is_starter'] = pd.to_numeric(df['is_starter'], errors='coerce')
    rv = df[df['is_starter'] == 0].copy()

    for col in ['ERA', 'WHIP']:
        rv[col] = pd.to_numeric(rv[col], errors='coerce')

    rv = _shift_per_player(rv, ['ERA', 'WHIP'])

    return (
        rv.groupby(['game_id', 'side'])
        .agg(bullpen_ERA=('pre_ERA', 'mean'),
             bullpen_WHIP=('pre_WHIP', 'mean'))
        .reset_index()
    )


# ─────────────── 先發打線賽前 OPS/OBP/SLG ─────────────────

def build_lineup_stats(lineups):
    df = lineups.copy()
    df['is_PH'] = pd.to_numeric(df['is_PH'], errors='coerce')
    st = df[df['is_PH'] == 0].copy()

    for col in ['OPS', 'OBP', 'SLG']:
        st[col] = pd.to_numeric(st[col], errors='coerce')

    st = _shift_per_player(st, ['OPS', 'OBP', 'SLG'])

    return (
        st.groupby(['game_id', 'side'])
        .agg(lineup_OPS=('pre_OPS', 'mean'),
             lineup_OBP=('pre_OBP', 'mean'),
             lineup_SLG=('pre_SLG', 'mean'))
        .reset_index()
    )


# ─────────────────────────── 合併 ───────────────────────────

def _attach(df, src, side, rename_map):
    sub = (src[src['side'] == side]
           .drop(columns='side')
           .rename(columns=rename_map))
    return df.merge(sub, on='game_id', how='left')


def merge_all(games_f, rolling, starters, bullpen, lineup):
    df = games_f[['game_id', 'season_id', 'year', 'phase', 'date',
                  'home_team', 'away_team', 'home_win']].copy()

    # 近10場滾動
    for side, p in [('home', 'home'), ('away', 'away')]:
        df = _attach(df, rolling, side, {
            'rolling_win_rate':     f'{p}_win_rate_10',
            'rolling_runs_scored':  f'{p}_runs_scored_10',
            'rolling_runs_allowed': f'{p}_runs_allowed_10',
        })

    # 先發投手賽前指標
    for side, p in [('home', 'home'), ('away', 'away')]:
        df = _attach(df, starters, side, {
            'pre_ERA':  f'{p}_starter_ERA',
            'pre_WHIP': f'{p}_starter_WHIP',
            'pre_FIP':  f'{p}_starter_FIP',
        })

    # 牛棚賽前指標
    for side, p in [('home', 'home'), ('away', 'away')]:
        df = _attach(df, bullpen, side, {
            'bullpen_ERA':  f'{p}_bullpen_ERA',
            'bullpen_WHIP': f'{p}_bullpen_WHIP',
        })

    # 打線賽前指標
    for side, p in [('home', 'home'), ('away', 'away')]:
        df = _attach(df, lineup, side, {
            'lineup_OPS': f'{p}_lineup_OPS',
            'lineup_OBP': f'{p}_lineup_OBP',
            'lineup_SLG': f'{p}_lineup_SLG',
        })

    # 差值特徵（home − away，正值代表主隊優勢）
    df['win_rate_diff']     = df['home_win_rate_10']    - df['away_win_rate_10']
    df['runs_scored_diff']  = df['home_runs_scored_10'] - df['away_runs_scored_10']
    df['run_diff_10']       = (df['home_runs_scored_10'] - df['home_runs_allowed_10']) \
                            - (df['away_runs_scored_10'] - df['away_runs_allowed_10'])
    df['starter_ERA_diff']  = df['home_starter_ERA']    - df['away_starter_ERA']
    df['starter_WHIP_diff'] = df['home_starter_WHIP']   - df['away_starter_WHIP']
    df['starter_FIP_diff']  = df['home_starter_FIP']    - df['away_starter_FIP']
    df['lineup_OPS_diff']   = df['home_lineup_OPS']     - df['away_lineup_OPS']
    df['lineup_OBP_diff']   = df['home_lineup_OBP']     - df['away_lineup_OBP']
    df['lineup_SLG_diff']   = df['home_lineup_SLG']     - df['away_lineup_SLG']
    df['bullpen_ERA_diff']  = df['home_bullpen_ERA']     - df['away_bullpen_ERA']
    df['bullpen_WHIP_diff'] = df['home_bullpen_WHIP']    - df['away_bullpen_WHIP']

    return df


# ─────────────────────────── 報告 ───────────────────────────

def print_report(df):
    id_cols = {'game_id', 'season_id', 'year', 'phase', 'date',
               'home_team', 'away_team', 'home_win'}
    feat_cols = [c for c in df.columns if c not in id_cols]

    print(f"\n{'='*50}")
    print(f"樣本數：{len(df)} 場")

    print(f"\nlabel 分布（home_win）：")
    for k in [1, 0]:
        n = (df['home_win'] == k).sum()
        lbl = '主隊勝' if k == 1 else '客隊勝'
        print(f"  {lbl}（{k}）: {n:5d}  ({n/len(df)*100:.1f}%)")
    print(f"  主場勝率: {df['home_win'].mean():.4f}")

    print(f"\n年份分布：")
    yc = df['year'].value_counts().sort_index()
    for yr, cnt in yc.items():
        print(f"  {yr}: {cnt}")

    print(f"\n特徵缺失率（僅列出 > 0 的欄位）：")
    missing = df[feat_cols].isnull().mean().sort_values(ascending=False)
    shown = missing[missing > 0]
    if shown.empty:
        print("  （全部特徵無缺失）")
    else:
        for col, rate in shown.items():
            n = df[col].isnull().sum()
            print(f"  {col}: {rate:.3f}  ({n} 筆)")
    print(f"  整體特徵平均缺失率: {missing.mean():.4f}")
    print(f"\n特徵欄位數：{len(feat_cols)}")
    print(f"{'='*50}")


# ─────────────────────────── 主流程 ─────────────────────────

def main():
    print("讀取 CSV...", flush=True)
    games, logs, pitchers, lineups = load_csvs()

    print("篩選比賽（phase=一軍 / FINISHED / home_win∈{0,1}）...", flush=True)
    games_f = filter_games(games)
    print(f"  通過篩選：{len(games_f)} 場")

    print("計算近10場滾動統計...", flush=True)
    rolling = build_rolling_stats(logs)

    print("計算先發投手賽前 ERA/WHIP/FIP...", flush=True)
    starters = build_starter_stats(pitchers)

    print("計算牛棚賽前 ERA/WHIP...", flush=True)
    bullpen = build_bullpen_stats(pitchers)

    print("計算先發打線賽前 OPS/OBP/SLG...", flush=True)
    lineup = build_lineup_stats(lineups)

    print("合併特徵...", flush=True)
    model_ready = merge_all(games_f, rolling, starters, bullpen, lineup)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_ready.to_csv(PROCESSED_DATA_DIR / 'model_ready_games.csv', index=False, encoding='utf-8-sig')
    print(f"已輸出 model_ready_games.csv  "
          f"（{len(model_ready)} 列 × {len(model_ready.columns)} 欄）")

    print_report(model_ready)


if __name__ == '__main__':
    main()
