"""
爬取 rebas.tw 所有賽季球員統計（2018–2026）
API: GET /api/seasons/{season}/stats  -> 取得該季所有球隊 ID
     GET /api/formal/seasons/{season}/teams/{team_id}/players -> 球員統計
"""

import requests
import json
import csv
import time
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXPORT_DIR = RAW_DATA_DIR / "player_season_exports"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "all_seasons_raw"
BASE = "https://www.rebas.tw"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.rebas.tw/',
}

ALL_SEASONS = [
    # 一軍
    ("CPBL-2026-oB", "2026", "一軍"),
    ("CPBL-2026-4j", "2026", "一軍-2"),
    ("CPBL-2025-JO", "2025", "一軍"),
    ("CPBL-2025-cF", "2025", "一軍-2"),
    ("CPBL-2025-lX", "2025", "一軍-3"),
    ("CPBL-2025-28", "2025", "一軍-4"),
    ("CPBL-2025-L0", "2025", "一軍-5"),
    ("CPBL-2024-xa", "2024", "一軍"),
    ("CPBL-2024-Uz", "2024", "一軍-2"),
    ("CPBL-2024-HE", "2024", "一軍-3"),
    ("CPBL-2024-pJ", "2024", "一軍-4"),
    ("CPBL-2023-Za", "2023", "一軍"),
    ("CPBL-2023-b1", "2023", "一軍-2"),
    ("CPBL-2023-sk", "2023", "一軍-3"),
    ("CPBL-2023-yB", "2023", "一軍-4"),
    ("CPBL-2022-dG", "2022", "一軍"),
    ("CPBL-2022-o7", "2022", "一軍-2"),
    ("CPBL-2022-yt", "2022", "一軍-3"),
    ("CPBL-2022-s6", "2022", "一軍-4"),
    ("CPBL-2021-fi", "2021", "一軍"),
    ("CPBL-2021-53", "2021", "一軍-2"),
    ("CPBL-2020-KS", "2020", "一軍"),
    ("CPBL-2019-Sf", "2019", "一軍"),
    ("CPBL-2018-Fq", "2018", "一軍"),
    # 二軍
    ("CPBLmi-2026-Jn", "2026", "二軍"),
    ("CPBLmi-2025-xS", "2025", "二軍"),
    ("CPBLmi-2024-S4", "2024", "二軍"),
    ("CPBLmi-2023-GP", "2023", "二軍"),
    ("CPBLmi-2022-C2", "2022", "二軍"),
]


def get_season_teams(season_id):
    """用 /api/seasons/{id}/stats 取得該季所有球隊"""
    url = f"{BASE}/api/seasons/{season_id}/stats"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return {}
    d = r.json()
    if d.get('error'):
        return {}
    teams = {}
    for t in d.get('data', {}).get('teams', []):
        team_info = t.get('team', {})
        uid = team_info.get('uniqid', '')
        name = team_info.get('name', '')
        if uid and name:
            teams[uid] = name
    return teams


def fetch_players(season_id, team_id):
    url = f"{BASE}/api/formal/seasons/{season_id}/teams/{team_id}/players"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return None
    d = r.json()
    if d.get('error'):
        return None
    return d['data']


def flatten_batter(row, season_id, year, phase, team_name, team_id):
    p = row.get('player', {})
    return {
        'season_id': season_id, 'year': year, 'phase': phase,
        'team': team_name, 'team_id': team_id,
        'player_id': p.get('uniqid', ''), 'name': p.get('name', ''), 'number': p.get('number', ''),
        'type': 'batter',
        'games': row.get('games', ''), 'PA': row.get('PA', ''), 'AB': row.get('AB', ''),
        'H': row.get('H', ''), 'Double': row.get('Double', ''), 'Triple': row.get('Triple', ''),
        'HR': row.get('HR', ''), 'RBI': row.get('RBI', ''), 'R': row.get('R', ''),
        'BB': row.get('BB', ''), 'SO': row.get('SO', ''), 'HBP': row.get('HBP', ''),
        'SB': row.get('SB', ''), 'CS': row.get('CS', ''),
        'AVG': row.get('AVG', ''), 'OBP': row.get('OBP', ''), 'SLG': row.get('SLG', ''),
        'OPS': row.get('OPS', ''), 'OPSplus': row.get('OPSplus', ''),
        'BABIP': row.get('BABIP', ''), 'ISO': row.get('ISO', ''),
        'wOBA': row.get('wOBA', ''), 'WPA': row.get('WPA', ''), 'RE24': row.get('RE24', ''),
        'BBp': row.get('BBp', ''), 'Kp': row.get('Kp', ''), 'GIDP': row.get('GIDP', ''),
    }


def flatten_pitcher(row, season_id, year, phase, team_name, team_id):
    p = row.get('player', {})
    sp = row.get('SP', 0)
    return {
        'season_id': season_id, 'year': year, 'phase': phase,
        'team': team_name, 'team_id': team_id,
        'player_id': p.get('uniqid', ''), 'name': p.get('name', ''), 'number': p.get('number', ''),
        'type': 'pitcher', 'role': 'starter' if sp and sp > 0 else 'reliever',
        'games': row.get('games', ''), 'SP': sp,
        'W': row.get('R_W', ''), 'L': row.get('R_L', ''), 'SV': row.get('R_SV', ''),
        'HLD': row.get('R_H', ''), 'BS': row.get('R_BS', ''),
        'IPOut': row.get('IPOut', ''),
        'H': row.get('H', ''), 'HR': row.get('HR', ''), 'BB': row.get('BB', ''),
        'SO': row.get('SO', ''), 'ER': row.get('ER', ''), 'R': row.get('R', ''),
        'HBP': row.get('HBP', ''), 'BF': row.get('BF', ''), 'NP': row.get('NP', ''),
        'ERA': row.get('ERA', ''), 'WHIP': row.get('WHIP', ''),
        'FIP': row.get('FIP', ''), 'ERAplus': row.get('ERAplus', ''),
        'SO9': row.get('SO9', ''), 'BB9': row.get('BB9', ''), 'HR9': row.get('HR9', ''),
        'H9': row.get('H9', ''), 'LOBp': row.get('LOBp', ''), 'GBp': row.get('GBp', ''),
        'Kp': row.get('Kp', ''), 'BBp': row.get('BBp', ''),
        'BABIP': row.get('BABIP', ''), 'WPA': row.get('WPA', ''), 'RE24': row.get('RE24', ''),
    }


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    all_batters = []
    all_pitchers = []
    season_meta = {}

    print(f"開始爬取，共 {len(ALL_SEASONS)} 個賽季\n")

    for season_id, year, phase in ALL_SEASONS:
        print(f"[{season_id}] {year} {phase}", end=' ... ', flush=True)

        teams = get_season_teams(season_id)
        if not teams:
            print("無法取得球隊清單，跳過")
            continue

        season_meta[season_id] = {'year': year, 'phase': phase, 'teams': teams}
        season_batters = []
        season_pitchers = []

        for team_id, team_name in teams.items():
            data = fetch_players(season_id, team_id)
            if not data:
                continue
            for row in data.get('batters', []):
                r = flatten_batter(row, season_id, year, phase, team_name, team_id)
                season_batters.append(r)
                all_batters.append(r)
            for row in data.get('pitchers', []):
                r = flatten_pitcher(row, season_id, year, phase, team_name, team_id)
                season_pitchers.append(r)
                all_pitchers.append(r)
            time.sleep(0.2)

        raw_path = CACHE_DIR / f"{season_id}.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump({'batters': season_batters, 'pitchers': season_pitchers}, f, ensure_ascii=False)

        print(f"{len(teams)} 隊 / 打者 {len(season_batters)} / 投手 {len(season_pitchers)}")
        time.sleep(0.3)

    # 儲存合併 CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_batters:
        path = EXPORT_DIR / f"cpbl_all_batters_{ts}.csv"
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(all_batters[0].keys()))
            w.writeheader()
            w.writerows(all_batters)
        print(f"\n打者 CSV: {path}  ({len(all_batters)} 筆)")

    if all_pitchers:
        path = EXPORT_DIR / f"cpbl_all_pitchers_{ts}.csv"
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(all_pitchers[0].keys()))
            w.writeheader()
            w.writerows(all_pitchers)
        print(f"投手 CSV: {path}  ({len(all_pitchers)} 筆)")

    meta_path = EXPORT_DIR / f"cpbl_season_meta_{ts}.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(season_meta, f, ensure_ascii=False, indent=2)
    print(f"賽季清單: {meta_path}")

    print(f"\n完成。打者合計 {len(all_batters)} 筆，投手合計 {len(all_pitchers)} 筆。")


if __name__ == "__main__":
    main()
