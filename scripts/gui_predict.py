"""
Simple CPBL future matchup predictor GUI.

User flow:
1. Enter a date.
2. Choose away and home teams.
3. Enter or auto-fill starter, bullpen, lineup, and recent team stats.
4. Click Predict.

The app trains a Random Forest on completed games before the requested date,
then predicts from the user-entered matchup stats. Auto-fill is only a
convenience for starting values; the editable fields are the prediction input.
"""

from __future__ import annotations

import sys
import tkinter as tk
from datetime import date, datetime, timedelta
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_games.csv"
LINEUPS_PATH = PROJECT_ROOT / "data" / "raw" / "lineups.csv"
PITCHERS_PATH = PROJECT_ROOT / "data" / "raw" / "pitchers_box.csv"

ID_COLS = {
    "game_id",
    "season_id",
    "year",
    "phase",
    "date",
    "home_team",
    "away_team",
}
LABEL = "home_win"

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "random_state": 42,
    "n_jobs": -1,
}

DIFF_FORMULAS = {
    "win_rate_diff": ("home_win_rate_10", "away_win_rate_10"),
    "runs_scored_diff": ("home_runs_scored_10", "away_runs_scored_10"),
    "run_diff_10": (None, None),
    "starter_ERA_diff": ("home_starter_ERA", "away_starter_ERA"),
    "starter_WHIP_diff": ("home_starter_WHIP", "away_starter_WHIP"),
    "starter_FIP_diff": ("home_starter_FIP", "away_starter_FIP"),
    "lineup_OPS_diff": ("home_lineup_OPS", "away_lineup_OPS"),
    "lineup_OBP_diff": ("home_lineup_OBP", "away_lineup_OBP"),
    "lineup_SLG_diff": ("home_lineup_SLG", "away_lineup_SLG"),
    "bullpen_ERA_diff": ("home_bullpen_ERA", "away_bullpen_ERA"),
    "bullpen_WHIP_diff": ("home_bullpen_WHIP", "away_bullpen_WHIP"),
}

EXPLANATION_FEATURES = [
    ("win_rate_diff", "Recent win-rate edge", "higher favors home"),
    ("run_diff_10", "Recent run-differential edge", "higher favors home"),
    ("starter_ERA_diff", "Starter ERA edge", "lower favors home"),
    ("starter_WHIP_diff", "Starter WHIP edge", "lower favors home"),
    ("starter_FIP_diff", "Starter FIP edge", "lower favors home"),
    ("lineup_OPS_diff", "Lineup OPS edge", "higher favors home"),
    ("bullpen_ERA_diff", "Bullpen ERA edge", "lower favors home"),
    ("bullpen_WHIP_diff", "Bullpen WHIP edge", "lower favors home"),
]

INPUT_FIELDS = [
    ("win_rate_10", "Win rate last 10", "0.000-1.000"),
    ("runs_scored_10", "Runs scored last 10", "runs/game"),
    ("runs_allowed_10", "Runs allowed last 10", "runs/game"),
    ("starter_ERA", "Starter ERA", "season/current"),
    ("starter_WHIP", "Starter WHIP", "season/current"),
    ("starter_FIP", "Starter FIP", "season/current"),
    ("bullpen_ERA", "Bullpen ERA", "current"),
    ("bullpen_WHIP", "Bullpen WHIP", "current"),
    ("lineup_OPS", "Lineup OPS", "expected lineup"),
    ("lineup_OBP", "Lineup OBP", "expected lineup"),
    ("lineup_SLG", "Lineup SLG", "expected lineup"),
]


def parse_date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.strptime(value.strip(), "%Y-%m-%d").date())
    except ValueError as exc:
        raise ValueError("Date must use YYYY-MM-DD, for example 2026-06-01.") from exc


def load_data() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run python scripts/build_model_ready.py first."
        )

    df = pd.read_csv(DATA_PATH, dtype={"year": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
    df = df[df[LABEL].isin([0, 1])].copy()
    df[LABEL] = df[LABEL].astype(int)

    feature_cols = [c for c in df.columns if c not in ID_COLS and c != LABEL]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["date", "game_id"]).reset_index(drop=True), feature_cols


def load_lineup_data() -> pd.DataFrame:
    if not LINEUPS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(LINEUPS_PATH, dtype=str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["OPS", "OBP", "SLG", "batting_order", "is_PH"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["date", "game_id", "side", "batting_order"]).reset_index(drop=True)


def load_pitcher_data() -> pd.DataFrame:
    if not PITCHERS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(PITCHERS_PATH, dtype=str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["ERA", "WHIP", "FIP", "order", "is_starter"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["date", "game_id", "side", "order"]).reset_index(drop=True)


def train_random_forest(
    df: pd.DataFrame, feature_cols: list[str], prediction_date: pd.Timestamp
) -> tuple[Pipeline, int, str]:
    train = df[df["date"] < prediction_date].copy()
    if len(train) < 100:
        raise ValueError(
            f"Only {len(train)} prior games are available before {prediction_date.date()}; "
            "choose a later date."
        )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(**RF_PARAMS)),
        ]
    )
    model.fit(train[feature_cols], train[LABEL])

    years = sorted(train["year"].dropna().unique(), key=int)
    year_text = f"{years[0]}-{years[-1]}" if years else "unknown years"
    return model, len(train), year_text


def side_for_row(row: pd.Series, team: str) -> str | None:
    if row.get("home_team") == team:
        return "home"
    if row.get("away_team") == team:
        return "away"
    return None


def latest_team_row(
    df: pd.DataFrame, team: str, target_side: str, prediction_date: pd.Timestamp
) -> tuple[pd.Series, str]:
    before = df[df["date"] < prediction_date].copy()
    same_side = before[before[f"{target_side}_team"] == team]
    if not same_side.empty:
        return same_side.iloc[-1], target_side

    any_side = before[(before["home_team"] == team) | (before["away_team"] == team)]
    if any_side.empty:
        raise ValueError(f"No prior games found for team: {team}")

    row = any_side.iloc[-1]
    found_side = side_for_row(row, team)
    if found_side is None:
        raise ValueError(f"Could not locate team side for: {team}")
    return row, found_side


def copy_side_features(
    feature_row: dict[str, float],
    feature_cols: list[str],
    source_row: pd.Series,
    source_side: str,
    target_side: str,
) -> None:
    source_prefix = f"{source_side}_"
    target_prefix = f"{target_side}_"

    for col in feature_cols:
        if not col.startswith(target_prefix):
            continue
        source_col = source_prefix + col[len(target_prefix) :]
        if source_col in source_row.index:
            feature_row[col] = source_row[source_col]


def add_diff_features(feature_row: dict[str, float]) -> None:
    for diff_col, (home_col, away_col) in DIFF_FORMULAS.items():
        if diff_col == "run_diff_10":
            h_scored = feature_row.get("home_runs_scored_10")
            h_allowed = feature_row.get("home_runs_allowed_10")
            a_scored = feature_row.get("away_runs_scored_10")
            a_allowed = feature_row.get("away_runs_allowed_10")
            feature_row[diff_col] = (h_scored - h_allowed) - (a_scored - a_allowed)
            continue

        feature_row[diff_col] = feature_row.get(home_col) - feature_row.get(away_col)


def build_prediction_row(
    df: pd.DataFrame,
    feature_cols: list[str],
    prediction_date: pd.Timestamp,
    away_team: str,
    home_team: str,
) -> tuple[pd.DataFrame, str]:
    feature_row = {col: pd.NA for col in feature_cols}

    home_source, home_source_side = latest_team_row(df, home_team, "home", prediction_date)
    away_source, away_source_side = latest_team_row(df, away_team, "away", prediction_date)

    copy_side_features(feature_row, feature_cols, home_source, home_source_side, "home")
    copy_side_features(feature_row, feature_cols, away_source, away_source_side, "away")
    add_diff_features(feature_row)

    source_text = (
        "Estimated from latest prior team profiles: "
        f"{home_team} from {home_source['date'].date()} as {home_source_side}, "
        f"{away_team} from {away_source['date'].date()} as {away_source_side}."
    )
    return pd.DataFrame([feature_row], columns=feature_cols), source_text


def profile_values(source_row: pd.Series, source_side: str) -> dict[str, float]:
    values = {}
    prefix = f"{source_side}_"
    for suffix, _label, _hint in INPUT_FIELDS:
        values[suffix] = source_row.get(prefix + suffix, pd.NA)
    return values


def value_to_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def parse_number(label: str, value: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a number. Got: {value!r}") from exc


def build_manual_prediction_row(
    feature_cols: list[str],
    manual_values: dict[str, dict[str, float]],
) -> pd.DataFrame:
    feature_row = {col: pd.NA for col in feature_cols}

    for side in ("home", "away"):
        for suffix, _label, _hint in INPUT_FIELDS:
            col = f"{side}_{suffix}"
            if col in feature_row:
                feature_row[col] = manual_values[side][suffix]

    add_diff_features(feature_row)
    return pd.DataFrame([feature_row], columns=feature_cols)


def player_options(
    lineups: pd.DataFrame, team: str, prediction_date: pd.Timestamp
) -> list[str]:
    if lineups.empty:
        return []

    rows = lineups[
        (lineups["team"] == team)
        & (lineups["date"] < prediction_date)
        & (lineups.get("is_PH", 0).fillna(0) == 0)
    ].copy()
    if rows.empty:
        return []

    latest_year = rows["year"].dropna().astype(int).max()
    rows = rows[rows["year"].astype(int) == latest_year]
    latest = rows.sort_values("date").drop_duplicates("name", keep="last")
    names = latest["name"].dropna().astype(str).tolist()
    return sorted(set(names))


def latest_batter_stats(
    lineups: pd.DataFrame, team: str, player_name: str, prediction_date: pd.Timestamp
) -> pd.Series | None:
    rows = lineups[
        (lineups["team"] == team)
        & (lineups["name"] == player_name)
        & (lineups["date"] < prediction_date)
    ].copy()
    if rows.empty:
        return None
    return rows.iloc[-1]


def lineup_average_from_players(
    lineups: pd.DataFrame,
    team: str,
    player_names: list[str],
    prediction_date: pd.Timestamp,
) -> tuple[dict[str, float], list[str]]:
    stats = []
    missing = []
    for name in player_names:
        clean = name.strip()
        if not clean:
            continue
        row = latest_batter_stats(lineups, team, clean, prediction_date)
        if row is None:
            missing.append(clean)
            continue
        stats.append(row)

    if not stats:
        raise ValueError(f"No prior batting stats found for selected {team} lineup players.")

    df = pd.DataFrame(stats)
    return {
        "lineup_OPS": float(df["OPS"].mean()),
        "lineup_OBP": float(df["OBP"].mean()),
        "lineup_SLG": float(df["SLG"].mean()),
    }, missing


def pitcher_options(
    pitchers: pd.DataFrame,
    team: str,
    prediction_date: pd.Timestamp,
    starter_only: bool | None = None,
) -> list[str]:
    if pitchers.empty:
        return []

    rows = pitchers[(pitchers["team"] == team) & (pitchers["date"] < prediction_date)].copy()
    if starter_only is True:
        rows = rows[rows.get("is_starter", 0).fillna(0) == 1]
    elif starter_only is False:
        rows = rows[rows.get("is_starter", 0).fillna(0) == 0]
    if rows.empty:
        return []

    latest_year = rows["year"].dropna().astype(int).max()
    rows = rows[rows["year"].astype(int) == latest_year]
    latest = rows.sort_values("date").drop_duplicates("name", keep="last")
    names = latest["name"].dropna().astype(str).tolist()
    return sorted(set(names))


def latest_pitcher_stats(
    pitchers: pd.DataFrame, team: str, player_name: str, prediction_date: pd.Timestamp
) -> pd.Series | None:
    rows = pitchers[
        (pitchers["team"] == team)
        & (pitchers["name"] == player_name)
        & (pitchers["date"] < prediction_date)
    ].copy()
    if rows.empty:
        return None
    return rows.iloc[-1]


def starter_stats_from_player(
    pitchers: pd.DataFrame, team: str, player_name: str, prediction_date: pd.Timestamp
) -> dict[str, float]:
    row = latest_pitcher_stats(pitchers, team, player_name.strip(), prediction_date)
    if row is None:
        raise ValueError(f"No prior pitcher stats found for starter: {team} {player_name}")
    return {
        "starter_ERA": float(row["ERA"]),
        "starter_WHIP": float(row["WHIP"]),
        "starter_FIP": float(row["FIP"]),
    }


def bullpen_average_from_players(
    pitchers: pd.DataFrame,
    team: str,
    player_names: list[str],
    prediction_date: pd.Timestamp,
) -> tuple[dict[str, float], list[str]]:
    stats = []
    missing = []
    for name in player_names:
        clean = name.strip()
        if not clean:
            continue
        row = latest_pitcher_stats(pitchers, team, clean, prediction_date)
        if row is None:
            missing.append(clean)
            continue
        stats.append(row)

    if not stats:
        raise ValueError(f"No prior pitching stats found for selected {team} bullpen pitchers.")

    df = pd.DataFrame(stats)
    return {
        "bullpen_ERA": float(df["ERA"].mean()),
        "bullpen_WHIP": float(df["WHIP"].mean()),
    }, missing


def confidence_label(prob_home: float) -> str:
    if prob_home >= 0.65 or prob_home <= 0.35:
        return "Strong"
    if prob_home >= 0.58 or prob_home <= 0.42:
        return "Medium"
    return "Low"


def edge_text(col: str, value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "missing"
    if pd.isna(val):
        return "missing"
    if abs(val) < 0.0005:
        return "even"

    lower_favors_home = col in {
        "starter_ERA_diff",
        "starter_WHIP_diff",
        "starter_FIP_diff",
        "bullpen_ERA_diff",
        "bullpen_WHIP_diff",
    }
    home_edge = val < 0 if lower_favors_home else val > 0
    side = "home edge" if home_edge else "away edge"
    return f"{val:.3f}, {side}"


class MatchupPredictor(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CPBL Future Matchup Predictor")
        self.geometry("1180x940")
        self.minsize(1040, 760)

        try:
            self.df, self.feature_cols = load_data()
            self.lineups = load_lineup_data()
            self.pitchers = load_pitcher_data()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Startup error", str(exc))
            raise SystemExit(1) from exc

        self.date_var = tk.StringVar()
        self.away_var = tk.StringVar()
        self.home_var = tk.StringVar()
        self.result_var = tk.StringVar(value="Enter a future matchup and click Predict.")
        self.input_vars: dict[str, dict[str, tk.StringVar]] = {
            "away": {},
            "home": {},
        }
        self.lineup_vars: dict[str, list[tk.StringVar]] = {
            "away": [],
            "home": [],
        }
        self.lineup_combos: dict[str, list[ttk.Combobox]] = {
            "away": [],
            "home": [],
        }
        self.starter_vars: dict[str, tk.StringVar] = {
            "away": tk.StringVar(),
            "home": tk.StringVar(),
        }
        self.starter_combos: dict[str, ttk.Combobox | None] = {
            "away": None,
            "home": None,
        }
        self.bullpen_vars: dict[str, list[tk.StringVar]] = {
            "away": [],
            "home": [],
        }
        self.bullpen_combos: dict[str, list[ttk.Combobox]] = {
            "away": [],
            "home": [],
        }
        self.source_var = tk.StringVar(value="Use Auto-fill, then edit the stats you know.")

        self._build_ui()
        self._set_defaults()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        form = ttk.Frame(self, padding=16)
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        form.columnconfigure(5, weight=1)

        ttk.Label(form, text="Game date").grid(row=0, column=0, padx=(0, 6), sticky="w")
        ttk.Entry(form, textvariable=self.date_var, width=14).grid(
            row=0, column=1, padx=(0, 14), sticky="ew"
        )

        ttk.Label(form, text="Away team").grid(row=0, column=2, padx=(0, 6), sticky="w")
        self.away_combo = ttk.Combobox(form, textvariable=self.away_var, state="readonly")
        self.away_combo.grid(row=0, column=3, padx=(0, 14), sticky="ew")

        ttk.Label(form, text="Home team").grid(row=0, column=4, padx=(0, 6), sticky="w")
        self.home_combo = ttk.Combobox(form, textvariable=self.home_var, state="readonly")
        self.home_combo.grid(row=0, column=5, padx=(0, 14), sticky="ew")

        ttk.Button(form, text="Auto-fill latest", command=self.autofill_latest).grid(
            row=0, column=6, padx=(0, 8)
        )
        ttk.Button(form, text="Predict", command=self.predict).grid(row=0, column=7)

        content = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

        self.inputs_frame = ttk.Frame(content)
        self.inputs_frame.columnconfigure(0, weight=1)
        content.add(self.inputs_frame, weight=3)

        self.result_frame = ttk.LabelFrame(content, text="Prediction Result", padding=12)
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(2, weight=1)
        content.add(self.result_frame, weight=2)

        self._build_manual_inputs()
        self._build_pitcher_inputs()
        self._build_lineup_inputs()

        main = self.result_frame
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        ttk.Label(main, text="Future Matchup Prediction", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            main,
            textvariable=self.result_var,
            wraplength=820,
            justify="left",
            font=("Segoe UI", 12),
        ).grid(row=1, column=0, sticky="ew", pady=(0, 12))

        self.detail = tk.Text(
            main,
            wrap="word",
            relief="solid",
            borderwidth=1,
            font=("Consolas", 10),
        )
        self.detail.grid(row=2, column=0, sticky="nsew")
        self.detail.configure(state="disabled")

    def _build_manual_inputs(self) -> None:
        panel = ttk.Frame(self.inputs_frame, padding=(0, 0, 8, 12))
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        away_box = ttk.LabelFrame(panel, text="Away Team Manual Inputs", padding=10)
        home_box = ttk.LabelFrame(panel, text="Home Team Manual Inputs", padding=10)
        away_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        home_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_side_inputs(away_box, "away")
        self._build_side_inputs(home_box, "home")

        ttk.Label(
            panel,
            textvariable=self.source_var,
            wraplength=820,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_side_inputs(self, parent: ttk.LabelFrame, side: str) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)

        for idx, (suffix, label, hint) in enumerate(INPUT_FIELDS):
            row = idx // 2
            col = (idx % 2) * 2
            var = tk.StringVar()
            self.input_vars[side][suffix] = var

            ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 6))
            entry = ttk.Entry(parent, textvariable=var, width=10)
            entry.grid(row=row, column=col + 1, sticky="ew", padx=(0, 12), pady=2)
            entry.insert(0, "")
            entry.configure()

    def _build_pitcher_inputs(self) -> None:
        panel = ttk.Frame(self.inputs_frame, padding=(0, 0, 8, 12))
        panel.grid(row=1, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        away_box = ttk.LabelFrame(panel, text="Away Pitchers", padding=10)
        home_box = ttk.LabelFrame(panel, text="Home Pitchers", padding=10)
        away_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        home_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_pitcher_side(away_box, "away")
        self._build_pitcher_side(home_box, "home")

        ttk.Button(
            panel,
            text="Fill starter and bullpen stats from selected pitchers",
            command=self.fill_pitcher_stats_from_players,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_pitcher_side(self, parent: ttk.LabelFrame, side: str) -> None:
        parent.columnconfigure(1, weight=1)
        for col in range(6):
            parent.columnconfigure(col, weight=1)

        ttk.Label(parent, text="Starter").grid(row=0, column=0, sticky="w", padx=(0, 6))
        starter = ttk.Combobox(parent, textvariable=self.starter_vars[side])
        starter.grid(row=0, column=1, columnspan=4, sticky="ew", padx=(0, 4), pady=2)
        starter.configure(postcommand=lambda s=side: self.refresh_pitcher_options(s))
        self.starter_combos[side] = starter

        ttk.Label(parent, text="Bullpen").grid(row=1, column=0, sticky="w", padx=(0, 6))
        for idx in range(5):
            var = tk.StringVar()
            combo = ttk.Combobox(parent, textvariable=var)
            combo.grid(row=1, column=idx + 1, sticky="ew", padx=4, pady=2)
            combo.configure(postcommand=lambda s=side: self.refresh_pitcher_options(s))
            self.bullpen_vars[side].append(var)
            self.bullpen_combos[side].append(combo)

    def _build_lineup_inputs(self) -> None:
        panel = ttk.Frame(self.inputs_frame, padding=(0, 0, 8, 0))
        panel.grid(row=2, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        away_box = ttk.LabelFrame(panel, text="Away Lineup Players", padding=10)
        home_box = ttk.LabelFrame(panel, text="Home Lineup Players", padding=10)
        away_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        home_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_lineup_side(away_box, "away")
        self._build_lineup_side(home_box, "home")

        ttk.Button(
            panel,
            text="Fill lineup OPS/OBP/SLG from selected players",
            command=self.fill_lineup_stats_from_players,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_lineup_side(self, parent: ttk.LabelFrame, side: str) -> None:
        for col in range(3):
            parent.columnconfigure(col, weight=1)

        for idx in range(9):
            var = tk.StringVar()
            combo = ttk.Combobox(parent, textvariable=var)
            combo.grid(row=idx // 3, column=idx % 3, sticky="ew", padx=4, pady=2)
            combo.configure(postcommand=lambda s=side: self.refresh_player_options(s))
            self.lineup_vars[side].append(var)
            self.lineup_combos[side].append(combo)

    def _set_defaults(self) -> None:
        latest_year = self.df["year"].dropna().astype(int).max()
        current = self.df[self.df["year"].astype(int) == latest_year]
        teams = sorted(set(current["home_team"].dropna()) | set(current["away_team"].dropna()))
        self.away_combo["values"] = teams
        self.home_combo["values"] = teams
        if len(teams) >= 2:
            self.away_var.set(teams[0])
            self.home_var.set(teams[1])
        self.away_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_team_changed())
        self.home_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_team_changed())

        latest = self.df["date"].max()
        default_date = latest.date() + timedelta(days=1) if pd.notna(latest) else date.today()
        self.date_var.set(default_date.strftime("%Y-%m-%d"))
        self.autofill_latest(show_errors=False)
        self.refresh_player_options("away")
        self.refresh_player_options("home")
        self.refresh_pitcher_options("away")
        self.refresh_pitcher_options("home")

    def on_team_changed(self) -> None:
        self.refresh_player_options("away")
        self.refresh_player_options("home")
        self.refresh_pitcher_options("away")
        self.refresh_pitcher_options("home")

    def refresh_player_options(self, side: str) -> None:
        try:
            prediction_date = parse_date(self.date_var.get())
        except ValueError:
            prediction_date = self.df["date"].max() + pd.Timedelta(days=1)

        team = self.away_var.get() if side == "away" else self.home_var.get()
        options = player_options(self.lineups, team, prediction_date)
        for combo in self.lineup_combos[side]:
            combo["values"] = options

    def refresh_pitcher_options(self, side: str) -> None:
        try:
            prediction_date = parse_date(self.date_var.get())
        except ValueError:
            prediction_date = self.df["date"].max() + pd.Timedelta(days=1)

        team = self.away_var.get() if side == "away" else self.home_var.get()
        starter_opts = pitcher_options(self.pitchers, team, prediction_date, starter_only=True)
        bullpen_opts = pitcher_options(self.pitchers, team, prediction_date, starter_only=False)
        if self.starter_combos[side] is not None:
            self.starter_combos[side]["values"] = starter_opts
        for combo in self.bullpen_combos[side]:
            combo["values"] = bullpen_opts

    def autofill_latest(self, show_errors: bool = True) -> None:
        try:
            prediction_date = parse_date(self.date_var.get())
            away_team = self.away_var.get().strip()
            home_team = self.home_var.get().strip()
            if not away_team or not home_team:
                raise ValueError("Choose both away and home teams.")
            if away_team == home_team:
                raise ValueError("Away and home team must be different.")

            home_source, home_source_side = latest_team_row(
                self.df, home_team, "home", prediction_date
            )
            away_source, away_source_side = latest_team_row(
                self.df, away_team, "away", prediction_date
            )

            for suffix, value in profile_values(home_source, home_source_side).items():
                self.input_vars["home"][suffix].set(value_to_text(value))
            for suffix, value in profile_values(away_source, away_source_side).items():
                self.input_vars["away"][suffix].set(value_to_text(value))

            self.autofill_lineup_players("home", home_team, prediction_date)
            self.autofill_lineup_players("away", away_team, prediction_date)
            self.autofill_pitcher_players("home", home_team, prediction_date)
            self.autofill_pitcher_players("away", away_team, prediction_date)

            self.source_var.set(
                "Auto-filled from latest prior profiles. Edit these fields for the actual "
                f"expected starters/lineups: {home_team} from {home_source['date'].date()} "
                f"as {home_source_side}; {away_team} from {away_source['date'].date()} "
                f"as {away_source_side}."
            )
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                messagebox.showerror("Auto-fill error", str(exc))

    def autofill_lineup_players(
        self, side: str, team: str, prediction_date: pd.Timestamp
    ) -> None:
        if self.lineups.empty:
            return

        rows = self.lineups[
            (self.lineups["team"] == team)
            & (self.lineups["date"] < prediction_date)
            & (self.lineups.get("is_PH", 0).fillna(0) == 0)
        ].copy()
        if rows.empty:
            return

        latest_game_id = rows.iloc[-1]["game_id"]
        latest_game = rows[rows["game_id"] == latest_game_id].sort_values("batting_order")
        names = latest_game["name"].dropna().astype(str).head(9).tolist()
        for idx, var in enumerate(self.lineup_vars[side]):
            var.set(names[idx] if idx < len(names) else "")

    def autofill_pitcher_players(
        self, side: str, team: str, prediction_date: pd.Timestamp
    ) -> None:
        if self.pitchers.empty:
            return

        rows = self.pitchers[
            (self.pitchers["team"] == team) & (self.pitchers["date"] < prediction_date)
        ].copy()
        if rows.empty:
            return

        latest_game_id = rows.iloc[-1]["game_id"]
        latest_game = rows[rows["game_id"] == latest_game_id].sort_values("order")
        starters = latest_game[latest_game.get("is_starter", 0).fillna(0) == 1]
        relievers = latest_game[latest_game.get("is_starter", 0).fillna(0) == 0]

        self.starter_vars[side].set(
            str(starters.iloc[0]["name"]) if not starters.empty else ""
        )
        names = relievers["name"].dropna().astype(str).head(5).tolist()
        for idx, var in enumerate(self.bullpen_vars[side]):
            var.set(names[idx] if idx < len(names) else "")

    def fill_lineup_stats_from_players(self) -> None:
        try:
            prediction_date = parse_date(self.date_var.get())
            missing_messages = []
            for side, team in [("away", self.away_var.get()), ("home", self.home_var.get())]:
                names = [var.get() for var in self.lineup_vars[side]]
                stats, missing = lineup_average_from_players(
                    self.lineups, team, names, prediction_date
                )
                for suffix in ["lineup_OPS", "lineup_OBP", "lineup_SLG"]:
                    field = suffix.replace("lineup_", "lineup_")
                    self.input_vars[side][field].set(value_to_text(stats[suffix]))
                if missing:
                    missing_messages.append(f"{team}: missing {', '.join(missing)}")

            msg = "Lineup OPS/OBP/SLG filled from selected players."
            if missing_messages:
                msg += " Some players were skipped: " + " | ".join(missing_messages)
            self.source_var.set(msg)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Lineup fill error", str(exc))

    def fill_pitcher_stats_from_players(self) -> None:
        try:
            prediction_date = parse_date(self.date_var.get())
            missing_messages = []
            for side, team in [("away", self.away_var.get()), ("home", self.home_var.get())]:
                starter_name = self.starter_vars[side].get().strip()
                if starter_name:
                    starter_stats = starter_stats_from_player(
                        self.pitchers, team, starter_name, prediction_date
                    )
                    for suffix in ["starter_ERA", "starter_WHIP", "starter_FIP"]:
                        self.input_vars[side][suffix].set(value_to_text(starter_stats[suffix]))

                bullpen_names = [var.get() for var in self.bullpen_vars[side]]
                if any(name.strip() for name in bullpen_names):
                    bullpen_stats, missing = bullpen_average_from_players(
                        self.pitchers, team, bullpen_names, prediction_date
                    )
                    for suffix in ["bullpen_ERA", "bullpen_WHIP"]:
                        self.input_vars[side][suffix].set(value_to_text(bullpen_stats[suffix]))
                    if missing:
                        missing_messages.append(f"{team}: missing {', '.join(missing)}")

            msg = "Starter and bullpen stats filled from selected pitchers."
            if missing_messages:
                msg += " Some pitchers were skipped: " + " | ".join(missing_messages)
            self.source_var.set(msg)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Pitcher fill error", str(exc))

    def collect_manual_values(self) -> dict[str, dict[str, float]]:
        values = {"home": {}, "away": {}}
        for side in ("home", "away"):
            for suffix, label, _hint in INPUT_FIELDS:
                raw = self.input_vars[side][suffix].get()
                values[side][suffix] = parse_number(f"{side} {label}", raw)
        return values

    def predict(self) -> None:
        try:
            prediction_date = parse_date(self.date_var.get())
            away_team = self.away_var.get().strip()
            home_team = self.home_var.get().strip()
            if not away_team or not home_team:
                raise ValueError("Choose both away and home teams.")
            if away_team == home_team:
                raise ValueError("Away and home team must be different.")

            model, train_n, train_years = train_random_forest(
                self.df, self.feature_cols, prediction_date
            )
            manual_values = self.collect_manual_values()
            x_pred = build_manual_prediction_row(self.feature_cols, manual_values)
            prob_home = float(model.predict_proba(x_pred)[0, 1])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Prediction error", str(exc))
            return

        prob_away = 1.0 - prob_home
        pick = home_team if prob_home >= 0.5 else away_team
        confidence = confidence_label(prob_home)
        self.result_var.set(
            f"{away_team} at {home_team} on {prediction_date.date()}\n"
            f"Random Forest pick: {pick} | Confidence: {confidence}\n"
            f"Home win probability: {prob_home * 100:.1f}% | "
            f"Away win probability: {prob_away * 100:.1f}%"
        )

        lines = [
            "MODEL",
            "  Type: Random Forest",
            f"  Training games: {train_n}",
            f"  Training seasons: {train_years}",
            "",
            "INPUT",
            f"  Date: {prediction_date.date()}",
            f"  Away: {away_team}",
            f"  Home: {home_team}",
            "",
            "RESULT",
            f"  Pick: {pick}",
            f"  Home win probability: {prob_home * 100:.1f}%",
            f"  Away win probability: {prob_away * 100:.1f}%",
            f"  Confidence: {confidence}",
            "",
            "FEATURE SOURCE",
            "  User-entered manual stats from the form.",
            f"  Auto-fill note: {self.source_var.get()}",
            "",
            "KEY EDGES",
        ]

        row = x_pred.iloc[0]
        for col, label, note in EXPLANATION_FEATURES:
            if col in row.index:
                lines.append(f"  {label}: {edge_text(col, row[col])} ({note})")

        lines.extend(
            [
                "",
                "IMPORTANT",
                "  This is for unknown future matchups. The model prediction uses",
                "  the editable stats in the form, not an exact historical game row.",
                "  Auto-fill is only a starting point. Replace starter, bullpen,",
                "  and lineup values with the real expected pre-game information.",
            ]
        )
        self._set_detail("\n".join(lines))

    def _set_detail(self, text: str) -> None:
        self.detail.configure(state="normal")
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")


def main() -> int:
    app = MatchupPredictor()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
