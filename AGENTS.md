# CPBL Data Collection Notes

## Active Layout

Read `docs/PROJECT_STRUCTURE.md` before moving files or running archived code.

```text
scripts/                  active Python scripts
notebooks/                model analysis notebooks
data/raw/                 reproducible CSV inputs
data/processed/           model-ready table
data/cache/               re-downloadable JSON cache
outputs/                  metrics, figures, and generated predictions
_archive/                 local-only historical artifacts
```

## Modeling Scope

The modeling dataset should contain CPBL first-team regular-season games only.

- Include: `LEAGUE_MATCHES`
- Exclude: postseason, challenge series, spring training, all-star games, minor league games
- Current rebas.tw cache coverage: 2018-2026
- Legacy CPBL official-site collection target: 2015-2017

Raw cache may contain additional competition types. The processed modeling table must continue to filter them out.

## Official Sources

### rebas.tw: 2018 onward

The league endpoint lists CPBL seasons and competition type:

```text
GET https://www.rebas.tw/api/leagues/CPBL
```

Keep seasons where:

```text
type == "LEAGUE_MATCHES"
```

Important endpoints:

```text
GET https://www.rebas.tw/api/seasons/{season_id}/games
GET https://www.rebas.tw/api/seasons/{season_id}/games/{game_id}
GET https://www.rebas.tw/api/seasons/{season_id}/stats
GET https://www.rebas.tw/api/formal/seasons/{season_id}/teams/{team_id}/players
```

The rebas.tw league endpoint currently starts at 2018. It does not expose 2015-2017 season IDs.

### CPBL official site: 2015-2017

Use the official English CPBL site:

```text
GET  https://en.cpbl.com.tw/box?year={year}&kindCode=A&gameSno={game_sno}
POST https://en.cpbl.com.tw/box/getlive
```

The initial `GET` provides the hidden `__RequestVerificationToken`. Submit it with:

```text
GameSno={game_sno}
KindCode=A
Year={year}
PrevOrNext=
PresentStatus=
```

`KindCode=A` means the first-team regular season. The JSON response includes:

```text
CurtGameDetailJson
ScoreboardJson
PitchingJson
BattingJson
FirstSnoJson
```

Use `https://en.cpbl.com.tw`, not `https://www.cpbl.com.tw`, for legacy batch collection. The Chinese site redirected repeated requests to its home page after sustained access. The English endpoint returns equivalent box-score fields and still includes Chinese player names.

## Archived Legacy Work

The paused 2015-2017 importer and downloaded data are archived locally:

```text
_archive/legacy_cpbl_official_2015_2017/
```

Snapshot:

| Year | Status | Games |
|------|--------|------:|
| 2015 | complete | 240 |
| 2016 | complete | 240 |
| 2017 | partial | 200 / 240 |

Archived files:

```text
CPBL-official-2015-A.json
CPBL-official-2016-A.json
CPBL-official-2017-A.partial.json
scrape_cpbl_legacy.py
```

The archived scraper is historical code. Before a fresh run, adapt its root paths so JSON cache writes to `data/cache/games_raw/` and CSV output writes to `data/raw/`. Do not copy it into the active pipeline unchanged.

## Suspended Legacy Process

On 2026-06-01, the original legacy scraper process was suspended:

```text
PID 17424
```

The process was loaded before the directory reorganization. Its in-memory paths are stale. Keep it suspended; do not resume it after this layout migration. The archived checkpoint is the source of truth for future continuation.

The four active raw CSV inputs were not rewritten by that process:

```text
data/raw/games.csv
data/raw/pitchers_box.csv
data/raw/lineups.csv
data/raw/team_game_logs.csv
```

## Fresh Continuation Checklist

When continuing 2015-2017 collection:

1. Copy `scrape_cpbl_legacy.py` from the archive into `scripts/`.
2. Adapt it to the active `data/cache/games_raw/` and `data/raw/` paths.
3. Copy the three archived JSON files into `data/cache/games_raw/`.
4. Run the adapted script with the Python 3.12 environment.
5. Run `python scripts/validate_data.py`.
6. Run `python scripts/build_model_ready.py`.

The legacy scraper intentionally throttles requests and saves a checkpoint every 20 games. Do not increase its request rate aggressively.
