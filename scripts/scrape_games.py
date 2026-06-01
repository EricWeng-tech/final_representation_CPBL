"""
爬取所有賽季的比賽記錄：
  games.csv             每場比賽基本資料（比分、勝負、球場）
  starting_pitchers.csv 每場先發投手成績
  lineups.csv           每場先發打線（含當場上場成績）
  team_game_logs.csv    每隊每場得分/失分（用於算近十場勝率）
"""

import requests, json, csv, os, sys, time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "games_raw"
BASE = "https://www.rebas.tw"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.rebas.tw/',
}

ALL_SEASONS = [
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
    ("CPBLmi-2026-Jn", "2026", "二軍"),
    ("CPBLmi-2025-xS", "2025", "二軍"),
    ("CPBLmi-2024-S4", "2024", "二軍"),
    ("CPBLmi-2023-GP", "2023", "二軍"),
    ("CPBLmi-2022-C2", "2022", "二軍"),
]


def get_json(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return None
    if 'json' not in r.headers.get('content-type', ''):
        return None
    d = r.json()
    return None if d.get('error') else d


def get_games_list(season_id):
    d = get_json(f"{BASE}/api/seasons/{season_id}/games")
    return d['data'] if d else []


def get_game_detail(season_id, game_id):
    d = get_json(f"{BASE}/api/seasons/{season_id}/games/{game_id}")
    return d['data'] if d else None


def parse_game_row(g, season_id, year, phase):
    info = g.get('info', {})
    home = g['home']
    away = g['away']
    scores_h = ','.join(str(s) for s in home.get('scores', []))
    scores_a = ','.join(str(s) for s in away.get('scores', []))
    winner = info.get('winner_side', '')
    return {
        'game_id': g['uniqid'],
        'season_id': season_id, 'year': year, 'phase': phase,
        'group': g.get('group', ''),
        'date': info.get('scheduled_start_at', '')[:10],
        'started_at': info.get('started_at', ''),
        'ended_at': info.get('ended_at', ''),
        'location': info.get('location', ''),
        'innings': info.get('innings', 9),
        'audience': info.get('audience', ''),
        'status': info.get('status', ''),
        'finished_status': info.get('finished_status', ''),
        'home_team': home.get('team', ''),
        'home_team_id': home.get('season_team_uniqid', ''),
        'away_team': away.get('team', ''),
        'away_team_id': away.get('season_team_uniqid', ''),
        'home_runs': home.get('runs', ''),
        'away_runs': away.get('runs', ''),
        'home_hits': home.get('hits', ''),
        'away_hits': away.get('hits', ''),
        'home_errors': home.get('errors', ''),
        'away_errors': away.get('errors', ''),
        'home_scores': scores_h,
        'away_scores': scores_a,
        'winner_side': winner,
        'home_win': 1 if winner == 'HOME' else (0 if winner == 'AWAY' else ''),
    }


def parse_pitchers(g, season_id, year, game_id, date, side):
    team_data = g[side]
    pitchers = team_data.get('box', {}).get('pitchings', [])
    rows = []
    for p in pitchers:
        player = p.get('player', {})
        rows.append({
            'game_id': game_id, 'season_id': season_id, 'year': year,
            'date': date, 'side': side,
            'team': team_data.get('team', ''),
            'team_id': team_data.get('season_team_uniqid', ''),
            'order': p.get('order', ''),
            'is_starter': 1 if p.get('order', 0) == 1 else 0,
            'player_id': player.get('uniqid', ''),
            'name': player.get('name', ''),
            'number': player.get('number', ''),
            'IP_out': p.get('IP_out', ''),
            'BF': p.get('BF', ''),
            'NP': p.get('NP', ''),
            'H': p.get('H', ''),
            'HR': p.get('HR', ''),
            'BB': p.get('BB', ''),
            'SO': p.get('SO', ''),
            'ER': p.get('ER', ''),
            'R': p.get('R', ''),
            'HB': p.get('HB', ''),
            'ERA': p.get('ERA', ''),
            'WHIP': p.get('WHIP', ''),
            'FIP': p.get('FIP', ''),
            'WPA': p.get('WPA', ''),
        })
    return rows


def parse_lineups(g, season_id, year, game_id, date, side):
    team_data = g[side]
    batters = team_data.get('box', {}).get('battings', [])
    rows = []
    for b in batters:
        player = b.get('player', {})
        rows.append({
            'game_id': game_id, 'season_id': season_id, 'year': year,
            'date': date, 'side': side,
            'team': team_data.get('team', ''),
            'team_id': team_data.get('season_team_uniqid', ''),
            'batting_order': b.get('order', ''),
            'is_PH': 1 if b.get('is_PH', False) else 0,
            'player_id': player.get('uniqid', ''),
            'name': player.get('name', ''),
            'number': player.get('number', ''),
            'PA': b.get('PA', ''), 'AB': b.get('AB', ''),
            'H': b.get('H', ''), '2B': b.get('2B', ''), '3B': b.get('3B', ''),
            'HR': b.get('HR', ''), 'RBI': b.get('RBI', ''), 'R': b.get('R', ''),
            'BB': b.get('BB', ''), 'SO': b.get('SO', ''), 'HBP': b.get('HBP', ''),
            'SB': b.get('SB', ''), 'CS': b.get('CS', ''),
            'AVG': b.get('AVG', ''), 'OBP': b.get('OBP', ''),
            'SLG': b.get('SLG', ''), 'OPS': b.get('OPS', ''),
            'WPA': b.get('WPA', ''),
        })
    return rows


def write_csv(path, rows, mode='a'):
    if not rows:
        return
    with open(path, mode, newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == 'w':
            w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # 初始化 CSV（寫入標頭）
    all_games, all_pitchers, all_lineups = [], [], []

    total_seasons = len(ALL_SEASONS)
    for si, (season_id, year, phase) in enumerate(ALL_SEASONS, 1):
        print(f"[{si}/{total_seasons}] {season_id} {year} {phase}", flush=True)

        games_list = get_games_list(season_id)
        finished = [g for g in games_list if g['info']['status'] == 'FINISHED']
        print(f"  總場數 {len(games_list)}, 已完賽 {len(finished)}", flush=True)

        season_games, season_pitchers, season_lineups = [], [], []

        for gi, g in enumerate(finished):
            game_id = g['uniqid']
            date = g['info'].get('scheduled_start_at', '')[:10]

            # 基本比賽資料（從列表即可）
            season_games.append(parse_game_row(g, season_id, year, phase))

            # 逐場取得 box score
            detail = get_game_detail(season_id, game_id)
            if detail:
                for side in ('home', 'away'):
                    season_pitchers.extend(parse_pitchers(detail, season_id, year, game_id, date, side))
                    season_lineups.extend(parse_lineups(detail, season_id, year, game_id, date, side))

            if (gi + 1) % 20 == 0:
                print(f"    {gi+1}/{len(finished)} 場...", flush=True)
            time.sleep(0.25)

        # 存每季原始
        with open(CACHE_DIR / f"{season_id}.json", 'w', encoding='utf-8') as f:
            json.dump({'games': season_games, 'pitchers': season_pitchers, 'lineups': season_lineups}, f, ensure_ascii=False)

        # 去重：API 偶爾對同一投手/打者回傳重複 entry，保留 order 最小的
        seen_p = {}
        for r in season_pitchers:
            k = (r['game_id'], r['side'], r['player_id'])
            if k not in seen_p or (r['order'] or 99) < (seen_p[k]['order'] or 99):
                seen_p[k] = r
        season_pitchers = list(seen_p.values())

        seen_l = {}
        for r in season_lineups:
            k = (r['game_id'], r['side'], r['player_id'])
            if k not in seen_l:
                seen_l[k] = r
        season_lineups = list(seen_l.values())

        all_games.extend(season_games)
        all_pitchers.extend(season_pitchers)
        all_lineups.extend(season_lineups)
        print(f"  完成: 比賽 {len(season_games)}, 投手紀錄 {len(season_pitchers)}, 打者紀錄 {len(season_lineups)}")

    # 輸出 CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if all_games:
        write_csv(RAW_DATA_DIR / "games.csv", all_games, 'w')
        print(f"\ngames.csv  ({len(all_games)} 場)")
    if all_pitchers:
        write_csv(RAW_DATA_DIR / "pitchers_box.csv", all_pitchers, 'w')
        print(f"pitchers_box.csv  ({len(all_pitchers)} 筆)")
    if all_lineups:
        write_csv(RAW_DATA_DIR / "lineups.csv", all_lineups, 'w')
        print(f"lineups.csv  ({len(all_lineups)} 筆)")

    # 從 games 衍生 team_game_logs.csv
    team_logs = []
    for g in all_games:
        if g['status'] != 'FINISHED':
            continue
        for side, opp in (('home', 'away'), ('away', 'home')):
            team_logs.append({
                'game_id': g['game_id'], 'season_id': g['season_id'],
                'year': g['year'], 'phase': g['phase'],
                'date': g['date'], 'group': g['group'],
                'team': g[f'{side}_team'], 'team_id': g[f'{side}_team_id'],
                'side': side,
                'opponent': g[f'{opp}_team'],
                'runs_scored': g[f'{side}_runs'],
                'runs_allowed': g[f'{opp}_runs'],
                'hits': g[f'{side}_hits'],
                'errors': g[f'{side}_errors'],
                'win': 1 if g['winner_side'] == side.upper() else (0 if g['winner_side'] in ('HOME','AWAY') else ''),
            })
    if team_logs:
        write_csv(RAW_DATA_DIR / "team_game_logs.csv", team_logs, 'w')
        print(f"team_game_logs.csv  ({len(team_logs)} 筆)")

    print(f"\n全部完成。")


if __name__ == "__main__":
    main()
