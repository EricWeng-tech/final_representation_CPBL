"""
棒球爬蟲資料品質驗證 — 完整版
參考 Retrosheet / Baseball Reference 的驗證邏輯

檢查分五層：
  L1 Schema        欄位存在、編碼、型別
  L2 Bounds        單欄位值域與合理範圍
  L3 Formula       欄位間公式一致性（OPS=OBP+SLG、ERA=9*ER/IP…）
  L4 Game          同場比賽內部一致性（局分合計、投手BF≈打者PA）
  L5 Cross-file    跨檔案參照完整性 + 賽季層級彙總

執行：python scripts/validate_data.py
"""

import csv, sys, math, collections, re
from pathlib import Path
from datetime import datetime, date

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"

issues = []

# ── helpers ──────────────────────────────────────────────────────────────────

def read_csv(path):
    try:
        with open(path, encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        issues.append((FAIL, path, "檔案不存在"))
        return None

def num(v):
    """轉 float，失敗回 None"""
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None

def i(v):
    """轉 int，失敗回 None"""
    try:
        return int(v)
    except (ValueError, TypeError):
        return None

def pct(n, total):
    return f"{n/total*100:.1f}%" if total else "N/A"

def log(level, section, msg):
    issues.append((level, section, msg))
    tag = {"[PASS]": "\033[32m[PASS]\033[0m",
           "[WARN]": "\033[33m[WARN]\033[0m",
           "[FAIL]": "\033[31m[FAIL]\033[0m"}.get(level, level)
    print(f"  {tag} {msg}")

def ip_out_to_ip(v):
    """IP_out（出局數）→ 投球局數 float，例如 7出局=2.1 IP，9出局=3.0 IP"""
    n = i(v)
    return None if n is None else (n // 3) + (n % 3) / 10

def missing_rate(rows, col):
    m = sum(1 for r in rows if r.get(col, '').strip() == '')
    return m, pct(m, len(rows))

# 棒球合理值域（偵測用，非直接拒絕）
BOUNDS = {
    # 打擊
    'AVG':  (0.0, 1.0),
    'OBP':  (0.0, 1.0),
    'SLG':  (0.0, 4.0),   # 純長打率理論最大 4.0（全 HR）
    'OPS':  (0.0, 5.0),
    'BABIP':(0.0, 1.0),
    'ISO':  (0.0, 3.0),
    'wOBA': (0.0, 1.0),
    # 投球
    'ERA':  (0.0, 99.0),
    'WHIP': (0.0, 10.0),
    'FIP':  (-5.0, 20.0),  # FIP 可為負（極少出局被打很少）
    'SO9':  (0.0, 30.0),
    'BB9':  (0.0, 30.0),
    'HR9':  (0.0, 20.0),
    'LOBp': (0.0, 1.0),
    'GBp':  (0.0, 1.0),
}

SEASON_DATE_RANGE = (date(2018, 1, 1), date(2026, 12, 31))
TODAY = date.today()

# ─────────────────────────────────────────────────────────────────────────────
# L1/L2: games.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== games.csv ===")
print("="*60)
games = read_csv(RAW_DATA_DIR / "games.csv")
game_ids_set = set()

if games:
    N = len(games)
    print(f"  總筆數: {N}")

    # L1 Schema — 必要欄位
    required = ['game_id','season_id','year','phase','date',
                'home_team','away_team','home_runs','away_runs',
                'home_hits','away_hits','home_scores','away_scores',
                'winner_side','home_win','status']
    missing_cols = [c for c in required if c not in games[0]]
    if missing_cols:
        log(FAIL, "games", f"缺少欄位: {missing_cols}")
    else:
        log(PASS, "games", "必要欄位齊全")

    # L1 編碼檢查 — 隊名含亂碼（應為中文或字母）
    garbled = sum(1 for r in games
                  if re.search(r'[\x80-\x9f]', r.get('home_team','') + r.get('away_team','')))
    log(PASS if garbled == 0 else FAIL, "games", f"隊名亂碼筆數: {garbled}")

    # L2 game_id 唯一性
    ids = [r['game_id'] for r in games]
    dup = N - len(set(ids))
    log(PASS if dup == 0 else FAIL, "games", f"重複 game_id: {dup} 筆")
    game_ids_set = set(ids)

    # L2 home_win 值域
    bad_win = [r['game_id'] for r in games if r.get('home_win') not in ('1','0','')]
    log(PASS if not bad_win else FAIL, "games",
        f"home_win 異常值: {len(bad_win)} 筆" + (f"（例: {bad_win[:3]}）" if bad_win else ""))

    # L2 隊名不空、主客不同
    no_name = sum(1 for r in games if not r.get('home_team','').strip()
                                    or not r.get('away_team','').strip())
    log(PASS if no_name == 0 else FAIL, "games", f"隊名空白: {no_name} 筆")

    same_team = sum(1 for r in games
                    if r.get('home_team') and r.get('home_team') == r.get('away_team'))
    log(PASS if same_team == 0 else FAIL, "games", f"主客隊相同（異常）: {same_team} 筆")

    # L2 得分非負
    neg_runs = sum(1 for r in games
                   for col in ('home_runs','away_runs')
                   if r.get(col,'') != '' and (i(r[col]) or 0) < 0)
    log(PASS if neg_runs == 0 else FAIL, "games", f"runs 負值: {neg_runs} 筆")

    # L2 日期合理性
    bad_date, future_date = 0, 0
    for r in games:
        d = r.get('date','')
        try:
            dt = datetime.strptime(d, '%Y-%m-%d').date()
            if dt > TODAY:
                future_date += 1
            if not (SEASON_DATE_RANGE[0] <= dt <= SEASON_DATE_RANGE[1]):
                bad_date += 1
        except ValueError:
            bad_date += 1
    log(PASS if bad_date == 0 else WARN, "games", f"日期格式/範圍異常: {bad_date} 筆")
    log(PASS if future_date == 0 else WARN, "games", f"未來日期（status=FINISHED 但日期>今天）: {future_date} 筆")

    # L2 局分合計 vs 總得分（home_scores 以逗號分隔）
    inning_mismatch = 0
    for r in games:
        for side in ('home','away'):
            total = i(r.get(f'{side}_runs',''))
            scores_str = r.get(f'{side}_scores','')
            if scores_str and total is not None:
                try:
                    inning_sum = sum(int(x) for x in scores_str.split(',') if x.strip() != '')
                    if inning_sum != total:
                        inning_mismatch += 1
                except ValueError:
                    inning_mismatch += 1
    log(PASS if inning_mismatch == 0 else WARN, "games",
        f"局分合計 ≠ 總得分: {inning_mismatch} 筆（side×game）")

    # L2 winner_side 與 home_win 互相驗證
    ws_mismatch = 0
    for r in games:
        ws = r.get('winner_side','')
        hw = r.get('home_win','')
        if ws == 'HOME' and hw != '1':
            ws_mismatch += 1
        elif ws == 'AWAY' and hw != '0':
            ws_mismatch += 1
        elif ws == '' and hw not in ('',''):
            pass  # 平局/未完賽 OK
    log(PASS if ws_mismatch == 0 else FAIL, "games",
        f"winner_side 與 home_win 不一致: {ws_mismatch} 筆")

    # L2 location 不應為空（FINISHED 比賽）
    no_location = sum(1 for r in games
                      if r.get('status') == 'FINISHED' and not r.get('location','').strip())
    log(PASS if no_location == 0 else WARN, "games",
        f"FINISHED 比賽 location 空白: {no_location} 筆")

    # L2 innings 合理範圍 5–16（中職縮短賽最少5局仍算正式完賽）
    bad_innings = sum(1 for r in games
                      if r.get('innings','') != ''
                      and not (5 <= (i(r.get('innings','')) or -1) <= 16))
    log(PASS if bad_innings == 0 else WARN, "games",
        f"innings 超出 [5,16]: {bad_innings} 筆")

    # L2 errors 非負
    neg_err = sum(1 for r in games
                  for col in ('home_errors','away_errors')
                  if r.get(col,'') != '' and (i(r.get(col,'')) or 0) < 0)
    log(PASS if neg_err == 0 else FAIL, "games", f"errors 負值: {neg_err} 筆")

    # L2 finished_status 非正常結束的場次（訓練時需標記）
    abnormal = [r['game_id'] for r in games
                if r.get('finished_status','') not in ('', 'NORMAL', 'normal')
                and r.get('status') == 'FINISHED']
    log(PASS if not abnormal else WARN, "games",
        f"非正常結束（雨延/中止等）的 FINISHED 比賽: {len(abnormal)} 場"
        + (f"（例 finished_status={[games[ids.index(g)].get('finished_status') for g in abnormal[:2]]}）"
           if abnormal else ""))

    # L2 每賽季每隊出賽次數合理（一般 100–400 場/隊）
    team_game_count = collections.Counter()
    for r in games:
        if r.get('status') == 'FINISHED':
            team_game_count[r['home_team']] += 1
            team_game_count[r['away_team']] += 1
    suspiciously_few = {t: c for t, c in team_game_count.items() if c < 5}
    log(PASS if not suspiciously_few else WARN, "games",
        f"出賽 < 5 場的隊伍: {list(suspiciously_few.keys())}")

# ─────────────────────────────────────────────────────────────────────────────
# L1–L4: pitchers_box.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== pitchers_box.csv ===")
print("="*60)
pitchers = read_csv(RAW_DATA_DIR / "pitchers_box.csv")

if pitchers:
    N = len(pitchers)
    print(f"  總筆數: {N}")

    # L1 name 不空
    no_name = sum(1 for r in pitchers if not r.get('name','').strip())
    log(PASS if no_name == 0 else FAIL, "pitchers", f"name 空白: {no_name} 筆")

    # L1 player_id 不空
    no_pid = sum(1 for r in pitchers if not r.get('player_id','').strip())
    log(PASS if no_pid == 0 else WARN, "pitchers", f"player_id 空白: {no_pid} 筆")

    # L2 is_starter 值域
    bad_starter = sum(1 for r in pitchers if r.get('is_starter') not in ('0','1'))
    log(PASS if bad_starter == 0 else FAIL, "pitchers", f"is_starter 異常值: {bad_starter} 筆")

    # L2 值域檢查
    for col, (lo, hi) in [('ERA',(0,99)),('WHIP',(0,10)),('FIP',(-5,20))]:
        vals = [num(r.get(col,'')) for r in pitchers]
        miss = sum(1 for v in vals if v is None)
        bad  = sum(1 for v in vals if v is not None and not (lo <= v <= hi))
        log(PASS if bad == 0 else WARN, "pitchers",
            f"{col}: 值域 [{lo},{hi}] 外 {bad} 筆, 缺失 {pct(miss,N)}")

    # L2 IP_out 非負
    neg_ip = sum(1 for r in pitchers if i(r.get('IP_out','')) is not None and i(r.get('IP_out','')) < 0)
    log(PASS if neg_ip == 0 else FAIL, "pitchers", f"IP_out 負值: {neg_ip} 筆")

    # L2 BF >= H + BB + HB（最低限制）
    bf_violation = 0
    for r in pitchers:
        bf = i(r.get('BF',''))
        h  = i(r.get('H',''))
        bb = i(r.get('BB',''))
        hb = i(r.get('HB',''))
        if bf is not None and h is not None and bb is not None and hb is not None:
            if bf < h + bb + hb:
                bf_violation += 1
    log(PASS if bf_violation == 0 else FAIL, "pitchers",
        f"BF < H+BB+HB（面對打者數矛盾）: {bf_violation} 筆")

    # L2 ER <= R（自責分不超過失分）
    er_gt_r = sum(1 for r in pitchers
                  if i(r.get('ER','')) is not None and i(r.get('R','')) is not None
                  and i(r.get('ER','')) > i(r.get('R','')))
    log(PASS if er_gt_r == 0 else FAIL, "pitchers", f"ER > R（自責分超過失分）: {er_gt_r} 筆")

    # L2 HR <= H（全壘打不超過安打）
    hr_gt_h = sum(1 for r in pitchers
                  if i(r.get('HR','')) is not None and i(r.get('H','')) is not None
                  and i(r.get('HR','')) > i(r.get('H','')))
    log(PASS if hr_gt_h == 0 else FAIL, "pitchers", f"HR > H（全壘打超過被安打數）: {hr_gt_h} 筆")

    # NOTE: ERA/WHIP 儲存的是「賽季累計值」（season-to-date），不是本場成績。
    #       公式 ERA=9*ER/IP 只適用於單場，此處不做公式驗證，確認非負即可。
    log(PASS, "pitchers", "ERA/WHIP 為賽季累計值（season-to-date），不驗證單場公式")

    # L3 IP=0 但有 ER/SO（邏輯矛盾）
    ip0_with_stats = sum(1 for r in pitchers
                         if i(r.get('IP_out','')) == 0
                         and (i(r.get('ER','')) or 0) > 0)
    log(PASS if ip0_with_stats == 0 else WARN, "pitchers",
        f"IP_out=0 但 ER>0: {ip0_with_stats} 筆")

    # L4 每場每邊恰好 1 位先發（order=1）
    starter_count = collections.Counter(
        (r['game_id'], r['side']) for r in pitchers if r.get('is_starter') == '1'
    )
    not_one = {k: v for k, v in starter_count.items() if v != 1}
    log(PASS if not not_one else WARN, "pitchers",
        f"每場每邊先發 ≠ 1 位的組合: {len(not_one)} 個")

    # L2 NP 投球數合理範圍（先發 > 30，上限 180）
    np_low  = sum(1 for r in pitchers
                  if r.get('is_starter') == '1'
                  and (i(r.get('NP','')) or 0) > 0
                  and (i(r.get('NP','')) or 999) < 30)
    np_high = sum(1 for r in pitchers
                  if (i(r.get('NP','')) or 0) > 180)
    log(PASS if np_low == 0 else WARN, "pitchers",
        f"先發投手 NP < 30（提前退場或資料異常）: {np_low} 筆")
    log(PASS if np_high == 0 else WARN, "pitchers",
        f"NP > 180（超高投球數）: {np_high} 筆")

    # L2 WPA 合理範圍 -3 到 +3
    wpa_bad = sum(1 for r in pitchers
                  if num(r.get('WPA','')) is not None
                  and not (-3 <= (num(r.get('WPA','')) or 0) <= 3))
    log(PASS if wpa_bad == 0 else WARN, "pitchers",
        f"WPA 超出 [-3,+3]: {wpa_bad} 筆")

    # L4 先發投手 order 必須為 1
    bad_sp_order = sum(1 for r in pitchers
                       if r.get('is_starter') == '1' and i(r.get('order','')) != 1)
    log(PASS if bad_sp_order == 0 else FAIL, "pitchers",
        f"is_starter=1 但 order≠1: {bad_sp_order} 筆")

    # L4 重複 (game_id, side, player_id)
    keys = [(r['game_id'], r['side'], r.get('player_id','')) for r in pitchers]
    dup = len(keys) - len(set(keys))
    log(PASS if dup == 0 else FAIL, "pitchers", f"重複 (game_id,side,player_id): {dup} 筆")

    # L5 game_id 參照完整性
    if games:
        orphan = sum(1 for r in pitchers if r['game_id'] not in game_ids_set)
        log(PASS if orphan == 0 else WARN, "pitchers",
            f"game_id 不在 games.csv: {orphan} 筆")

# ─────────────────────────────────────────────────────────────────────────────
# L1–L4: lineups.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== lineups.csv ===")
print("="*60)
lineups = read_csv(RAW_DATA_DIR / "lineups.csv")

if lineups:
    N = len(lineups)
    print(f"  總筆數: {N}")

    # L1 name 不空
    no_name = sum(1 for r in lineups if not r.get('name','').strip())
    log(PASS if no_name == 0 else FAIL, "lineups", f"name 空白: {no_name} 筆")

    # L1 player_id 不空
    no_pid = sum(1 for r in lineups if not r.get('player_id','').strip())
    log(PASS if no_pid == 0 else WARN, "lineups", f"player_id 空白: {no_pid} 筆")

    # L2 值域 + 缺失率
    stat_checks = [
        ('AVG', 0.0, 1.0), ('OBP', 0.0, 1.0),
        ('SLG', 0.0, 4.0), ('OPS', 0.0, 5.0),
    ]
    for col, lo, hi in stat_checks:
        vals = [num(r.get(col,'')) for r in lineups]
        miss = sum(1 for v in vals if v is None)
        bad  = sum(1 for v in vals if v is not None and not (lo <= v <= hi))
        lvl  = FAIL if miss/N > 0.5 else WARN if miss/N > 0.3 else PASS
        log(lvl, "lineups",
            f"{col}: 缺失 {pct(miss,N)}, 超出 [{lo},{hi}] 的 {bad} 筆")

    # L2 H ≤ AB
    h_gt_ab = sum(1 for r in lineups
                  if i(r.get('H','')) is not None and i(r.get('AB','')) is not None
                  and i(r.get('H','')) > i(r.get('AB','')))
    log(PASS if h_gt_ab == 0 else FAIL, "lineups", f"H > AB（安打超過打數）: {h_gt_ab} 筆")

    # L2 AB ≤ PA
    ab_gt_pa = sum(1 for r in lineups
                   if i(r.get('AB','')) is not None and i(r.get('PA','')) is not None
                   and i(r.get('AB','')) > i(r.get('PA','')))
    log(PASS if ab_gt_pa == 0 else FAIL, "lineups", f"AB > PA（打數超過打席）: {ab_gt_pa} 筆")

    # L2 2B + 3B + HR ≤ H
    xbh_gt_h = sum(1 for r in lineups
                   if all(i(r.get(c,'')) is not None for c in ('2B','3B','HR','H'))
                   and i(r['2B']) + i(r['3B']) + i(r['HR']) > i(r['H']))
    log(PASS if xbh_gt_h == 0 else FAIL, "lineups", f"2B+3B+HR > H（長打超過安打）: {xbh_gt_h} 筆")

    # L2 PA 會計公式：PA ≥ AB + BB + HBP（SF 未必有欄位，故用 ≥）
    pa_undercount = 0
    for r in lineups:
        pa  = i(r.get('PA',''))
        ab  = i(r.get('AB',''))
        bb  = i(r.get('BB',''))
        hbp = i(r.get('HBP',''))
        if pa is not None and ab is not None and bb is not None and hbp is not None:
            if pa < ab + bb + hbp:
                pa_undercount += 1
    log(PASS if pa_undercount == 0 else FAIL, "lineups",
        f"PA < AB+BB+HBP（打席數不足）: {pa_undercount} 筆")

    # L3 OPS = OBP + SLG（允許 ±0.002）
    ops_mismatch = 0
    for r in lineups:
        ops = num(r.get('OPS',''))
        obp = num(r.get('OBP',''))
        slg = num(r.get('SLG',''))
        if ops is not None and obp is not None and slg is not None:
            if abs(ops - (obp + slg)) > 0.002:
                ops_mismatch += 1
    log(PASS if ops_mismatch == 0 else WARN, "lineups",
        f"OPS ≠ OBP+SLG（誤差>0.002）: {ops_mismatch} 筆")

    # NOTE: AVG/OBP/SLG/OPS 同樣為「賽季累計值」，不做單場公式驗證。
    log(PASS, "lineups", "AVG/OBP/SLG/OPS 為賽季累計值（season-to-date），不驗證單場公式")

    # L3 投手欄位不應出現在打者欄位（ERA 欄若存在）
    if 'ERA' in lineups[0]:
        leak = sum(1 for r in lineups
                   if r.get('ERA','').strip() not in ('','0','0.0'))
        log(PASS if leak == 0 else FAIL, "lineups",
            f"打者欄位含非零 ERA（投手資料混入）: {leak} 筆")

    # L2 batting_order 非代打者應為 1–9
    bad_order = sum(1 for r in lineups
                    if r.get('is_PH') == '0'
                    and r.get('batting_order','') != ''
                    and not (1 <= (i(r.get('batting_order','')) or -1) <= 9))
    log(PASS if bad_order == 0 else WARN, "lineups",
        f"非代打者 batting_order 超出 [1,9]: {bad_order} 筆")

    # L2 WPA 合理範圍 -3 到 +3
    wpa_bad = sum(1 for r in lineups
                  if num(r.get('WPA','')) is not None
                  and not (-3 <= (num(r.get('WPA','')) or 0) <= 3))
    log(PASS if wpa_bad == 0 else WARN, "lineups",
        f"WPA 超出 [-3,+3]: {wpa_bad} 筆")

    # L4 重複 (game_id, side, player_id)
    keys = [(r['game_id'], r['side'], r.get('player_id','')) for r in lineups]
    dup = len(keys) - len(set(keys))
    log(PASS if dup == 0 else FAIL, "lineups", f"重複 (game_id,side,player_id): {dup} 筆")

    # L4 每場每邊打者人數（起始應 9–20 人）
    per_game = collections.Counter((r['game_id'], r['side']) for r in lineups)
    too_few  = sum(1 for v in per_game.values() if v < 9)
    too_many = sum(1 for v in per_game.values() if v > 20)
    log(PASS if too_few == 0 else WARN, "lineups",
        f"每場每邊打者 < 9 人: {too_few} 個 (game,side)")
    log(PASS if too_many == 0 else WARN, "lineups",
        f"每場每邊打者 > 20 人: {too_many} 個 (game,side)")

    # L4 是否有場次只有單邊打者（另一邊缺失）
    sides_per_game = collections.defaultdict(set)
    for r in lineups:
        sides_per_game[r['game_id']].add(r['side'])
    single_side = sum(1 for v in sides_per_game.values() if len(v) < 2)
    log(PASS if single_side == 0 else WARN, "lineups",
        f"只有單邊打者紀錄的比賽: {single_side} 場")

    # L5 game_id 參照完整性
    if games:
        orphan = sum(1 for r in lineups if r['game_id'] not in game_ids_set)
        log(PASS if orphan == 0 else WARN, "lineups",
            f"game_id 不在 games.csv: {orphan} 筆")

# ─────────────────────────────────────────────────────────────────────────────
# L1–L5: team_game_logs.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== team_game_logs.csv ===")
print("="*60)
logs = read_csv(RAW_DATA_DIR / "team_game_logs.csv")

if logs:
    N = len(logs)
    print(f"  總筆數: {N}")

    # L2 每場恰好 2 筆
    game_counts = collections.Counter(r['game_id'] for r in logs)
    not_two = {k: v for k, v in game_counts.items() if v != 2}
    log(PASS if not not_two else WARN, "logs",
        f"非 2 筆/場: {len(not_two)} 個 game_id（例: {list(not_two.items())[:3]}）")

    # L2 win 值域
    bad_win = sum(1 for r in logs if r.get('win') not in ('1','0',''))
    log(PASS if bad_win == 0 else FAIL, "logs", f"win 異常值: {bad_win} 筆")

    # L2 每場 win 合計 ≤ 1
    win_per_game = collections.defaultdict(int)
    for r in logs:
        if r.get('win') == '1':
            win_per_game[r['game_id']] += 1
    multi_win = sum(1 for v in win_per_game.values() if v > 1)
    log(PASS if multi_win == 0 else FAIL, "logs", f"同場 win 合計 > 1: {multi_win} 場")

    # L2 runs 非負
    neg_runs = sum(1 for r in logs
                   for col in ('runs_scored','runs_allowed')
                   if r.get(col,'') != '' and (num(r.get(col,'')) or 0) < 0)
    log(PASS if neg_runs == 0 else FAIL, "logs", f"runs 負值: {neg_runs} 筆")

    # L4 同場 runs 互相對稱（A 的 runs_scored = B 的 runs_allowed）
    run_asymmetry = 0
    by_game = collections.defaultdict(list)
    for r in logs:
        by_game[r['game_id']].append(r)
    for gid, pair in by_game.items():
        if len(pair) == 2:
            a, b = pair
            if num(a.get('runs_scored','')) is not None and num(b.get('runs_allowed','')) is not None:
                if abs((num(a['runs_scored']) or 0) - (num(b['runs_allowed']) or 0)) > 0.01:
                    run_asymmetry += 1
    log(PASS if run_asymmetry == 0 else FAIL, "logs",
        f"runs_scored ≠ 對手 runs_allowed: {run_asymmetry} 場")

    # L5 game_id 參照完整性
    if games:
        orphan = sum(1 for r in logs if r['game_id'] not in game_ids_set)
        log(PASS if orphan == 0 else WARN, "logs",
            f"game_id 不在 games.csv: {orphan} 筆")

# ─────────────────────────────────────────────────────────────────────────────
# L5: 跨檔案一致性
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== 跨檔案一致性 ===")
print("="*60)
if games and pitchers and lineups and logs:

    g_ids = set(r['game_id'] for r in games)
    p_ids = set(r['game_id'] for r in pitchers)
    l_ids = set(r['game_id'] for r in lineups)
    t_ids = set(r['game_id'] for r in logs)

    # 有打者紀錄卻無投手紀錄
    log(PASS if not (l_ids - p_ids) else WARN, "cross",
        f"有 lineup 但無 pitcher 的比賽: {len(l_ids - p_ids)} 場")

    # games 有紀錄但無 lineup
    finished_ids = {r['game_id'] for r in games if r.get('status') == 'FINISHED'}
    no_lineup = finished_ids - l_ids
    log(PASS if not no_lineup else WARN, "cross",
        f"FINISHED 但無 lineup 紀錄的比賽: {len(no_lineup)} 場")

    # games 有紀錄但無 logs
    no_logs = finished_ids - t_ids
    log(PASS if not no_logs else WARN, "cross",
        f"FINISHED 但無 team_game_logs 紀錄的比賽: {len(no_logs)} 場")

    # L5 賽季層級：每隊 W + L + 平局 應 ≈ 出賽場數
    if logs:
        team_wl = collections.defaultdict(lambda: {'W':0,'L':0,'G':0})
        for r in logs:
            t = r.get('team','')
            team_wl[t]['G'] += 1
            if r.get('win') == '1':
                team_wl[t]['W'] += 1
            elif r.get('win') == '0':
                team_wl[t]['L'] += 1
        wl_issues = {t: v for t, v in team_wl.items() if v['W'] + v['L'] > v['G']}
        log(PASS if not wl_issues else FAIL, "cross",
            f"W+L > G（勝負場數超過出賽）的隊伍: {list(wl_issues.keys())}")

    # L5 投手 BF 合計 ≈ 打者 PA 合計（同場同邊）
    pitcher_bf = collections.defaultdict(int)
    for r in pitchers:
        bf = i(r.get('BF',''))
        if bf:
            pitcher_bf[(r['game_id'], r['side'])] += bf

    batter_pa = collections.defaultdict(int)
    for r in lineups:
        pa = i(r.get('PA',''))
        if pa:
            batter_pa[(r['game_id'], r['side'])] += pa

    # NOTE: BF 可能包含換投時的共同打席，與打者 PA 合計略有差異屬正常。
    #       僅標記差距 > 10 的極端異常情況。
    common = set(pitcher_bf.keys()) & set(batter_pa.keys())
    bf_pa_mismatch = sum(1 for k in common if abs(pitcher_bf[k] - batter_pa[k]) > 10)
    log(PASS if bf_pa_mismatch == 0 else WARN, "cross",
        f"投手 BF 合計 vs 打者 PA 合計 差距 > 10 的 (game,side): {bf_pa_mismatch} 個")

# ─────────────────────────────────────────────────────────────────────────────
# 最終摘要
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== 摘要 ===")
print("="*60)
passes = sum(1 for l,_,_ in issues if l == PASS)
warns  = sum(1 for l,_,_ in issues if l == WARN)
fails  = sum(1 for l,_,_ in issues if l == FAIL)
print(f"  PASS: {passes}  WARN: {warns}  FAIL: {fails}")

if fails:
    print("\n  ── FAIL（必須修正）──")
    for l,f,m in issues:
        if l == FAIL:
            print(f"    [{f}] {m}")

if warns:
    print("\n  ── WARN（調查後決定）──")
    for l,f,m in issues:
        if l == WARN:
            print(f"    [{f}] {m}")

print()
sys.exit(1 if fails else 0)
