"""
CPBL 賽前勝負預測系統 — Streamlit Web App

Run:
    streamlit run scripts/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd
import streamlit as st

# ── Chinese font for matplotlib ───────────────────────────────────────────────
_CJK_PRIORITY = ["Microsoft YaHei", "SimHei", "Noto Sans TC", "Noto Sans CJK TC"]
_cjk_font: str | None = None
for _fn in _CJK_PRIORITY:
    if any(f.name == _fn for f in _fm.fontManager.ttflist):
        _cjk_font = _fn
        break
if _cjk_font:
    plt.rcParams["font.family"] = _cjk_font
plt.rcParams["axes.unicode_minus"] = False

# Allow importing gui_predict from same directory
sys.path.insert(0, str(Path(__file__).parent))

from gui_predict import (
    INPUT_FIELDS,
    add_diff_features,
    build_manual_prediction_row,
    bullpen_average_from_players,
    confidence_label,
    latest_pitcher_stats,
    latest_team_row,
    lineup_average_from_players,
    load_data,
    load_lineup_data,
    load_pitcher_data,
    load_saved_random_forest,
    parse_number,
    pitcher_options,
    player_options,
    profile_values,
    starter_stats_from_player,
    value_to_text,
)

# ── Team identity ─────────────────────────────────────────────────────────────
TEAM_INFO: dict[str, dict] = {
    "統一7-ELEVEn獅": {"abbr": "獅", "color": "#E3001B", "text": "#ffffff"},
    "中信兄弟":        {"abbr": "兄", "color": "#F5C518", "text": "#000000"},
    "樂天桃猿":        {"abbr": "猿", "color": "#003DA5", "text": "#ffffff"},
    "富邦悍將":        {"abbr": "將", "color": "#002868", "text": "#ffffff"},
    "台鋼雄鷹":        {"abbr": "鷹", "color": "#C8102E", "text": "#ffffff"},
    "味全龍":          {"abbr": "龍", "color": "#006400", "text": "#ffffff"},
}

_ASSETS = Path(__file__).parent.parent / "assets" / "team_logos"

# ── SHAP feature name translations ────────────────────────────────────────────
FEAT_ZH: dict[str, str] = {
    "home_win_rate_10":    "主隊近10場勝率",
    "away_win_rate_10":    "客隊近10場勝率",
    "home_runs_scored_10": "主隊近10場平均得分",
    "away_runs_scored_10": "客隊近10場平均得分",
    "home_runs_allowed_10":"主隊近10場平均失分",
    "away_runs_allowed_10":"客隊近10場平均失分",
    "home_starter_ERA":    "主隊先發 ERA",
    "away_starter_ERA":    "客隊先發 ERA",
    "home_starter_WHIP":   "主隊先發 WHIP",
    "away_starter_WHIP":   "客隊先發 WHIP",
    "home_starter_FIP":    "主隊先發 FIP",
    "away_starter_FIP":    "客隊先發 FIP",
    "home_bullpen_ERA":    "主隊牛棚 ERA",
    "away_bullpen_ERA":    "客隊牛棚 ERA",
    "home_bullpen_WHIP":   "主隊牛棚 WHIP",
    "away_bullpen_WHIP":   "客隊牛棚 WHIP",
    "home_lineup_OPS":     "主隊打線 OPS",
    "away_lineup_OPS":     "客隊打線 OPS",
    "home_lineup_OBP":     "主隊打線 OBP",
    "away_lineup_OBP":     "客隊打線 OBP",
    "home_lineup_SLG":     "主隊打線 SLG",
    "away_lineup_SLG":     "客隊打線 SLG",
    "win_rate_diff":       "近期勝率差距",
    "runs_scored_diff":    "近期得分差距",
    "run_diff_10":         "近期得失分差距",
    "starter_ERA_diff":    "先發 ERA 差值",
    "starter_WHIP_diff":   "先發 WHIP 差值",
    "starter_FIP_diff":    "先發 FIP 差值",
    "lineup_OPS_diff":     "打線 OPS 差值",
    "lineup_OBP_diff":     "打線 OBP 差值",
    "lineup_SLG_diff":     "打線 SLG 差值",
    "bullpen_ERA_diff":    "牛棚 ERA 差值",
    "bullpen_WHIP_diff":   "牛棚 WHIP 差值",
}

def _feat_label(name: str) -> str:
    return FEAT_ZH.get(name, name)

# ── SHAP-based fallback explanation (no LLM needed) ──────────────────────────
def _shap_fallback_explanation(
    sv_series: pd.Series,  # type: ignore[type-arg]
    home_team: str,
    away_team: str,
    prob_home: float,
    conf: str,
) -> str:
    top3: list[str] = sv_series.abs().sort_values(ascending=False).head(3).index.tolist()
    winner = home_team if prob_home >= 0.5 else away_team
    pct = f"{max(prob_home, 1-prob_home)*100:.0f}%"
    lines: list[str] = [f"模型預測 **{winner}** 獲勝，信心 {conf}（{pct}）。"]
    for feat in top3:
        val = float(sv_series.loc[feat])  # type: ignore[arg-type]
        label = _feat_label(str(feat))
        if val > 0:
            lines.append(f"**{label}** 有利於主隊 {home_team}。")
        else:
            lines.append(f"**{label}** 有利於客隊 {away_team}。")
    lines.append("以上分析來自隨機森林模型的 SHAP 特徵貢獻值。")
    return "\n\n".join(lines)

def _team_logo_html(team: str | None, size: int = 72) -> str:
    if not team:
        return f'<svg width="{size}" height="{size}"></svg>'
    png = _ASSETS / f"{team}.png"
    if png.exists():
        import base64
        data = base64.b64encode(png.read_bytes()).decode()
        return (
            f'<img src="data:image/png;base64,{data}" '
            f'width="{size}" height="{size}" '
            f'style="border-radius:50%;object-fit:cover;display:inline-block;" />'
        )
    info = TEAM_INFO.get(team, {"abbr": team[:1] if team else "?", "color": "#888888", "text": "#ffffff"})
    abbr = info["abbr"]
    bg   = info["color"]
    fg   = info["text"]
    font = int(size * 0.38)
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:{bg};display:inline-flex;align-items:center;'
        f'justify-content:center;font-weight:bold;color:{fg};'
        f'font-size:{font}px;font-family:sans-serif;vertical-align:middle;">'
        f'{abbr}</div>'
    )

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CPBL 賽前勝負預測",
    page_icon="⚾",
    layout="wide",
)

# ── Background image ──────────────────────────────────────────────────────────
_bg_path = Path(__file__).parent.parent / "CPBL.png"
if _bg_path.exists():
    import base64 as _b64
    _bg_b64 = _b64.b64encode(_bg_path.read_bytes()).decode()
    st.markdown(f"""
<style>
.stApp {{
    background-image:
        linear-gradient(rgba(245,247,250,0.88), rgba(245,247,250,0.88)),
        url("data:image/png;base64,{_bg_b64}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}}
</style>
""", unsafe_allow_html=True)

# ── Load resources (cached across reruns) ─────────────────────────────────────
@st.cache_resource(show_spinner="載入模型與資料…")
def get_resources():
    df, feature_cols = load_data()
    lineups = load_lineup_data()
    pitchers = load_pitcher_data()
    model, train_n, train_years, _ = load_saved_random_forest(feature_cols)
    return df, feature_cols, lineups, pitchers, model, train_n, train_years


df, feature_cols, lineups, pitchers, model, train_n, train_years = get_resources()

latest_year = df["year"].dropna().astype(int).max()
current_year_df = df[df["year"].astype(int) == latest_year]
teams = sorted(
    set(current_year_df["home_team"].dropna())
    | set(current_year_df["away_team"].dropna())
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚾ CPBL 賽前勝負預測系統")
st.caption(
    f"預測模型訓練資料：2018–2026（2,418 場）｜"
    f"參考準確率：70.39%（Notebook 實驗，2025 測試集）｜"
    f"特徵數：33｜模型：Tuned Random Forest"
)

# Team logo strip
logo_html = " &nbsp; ".join(
    f'<span title="{t}">{_team_logo_html(t, 48)}</span>' for t in TEAM_INFO
)
st.markdown(
    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">{logo_html}</div>',
    unsafe_allow_html=True,
)

# ── Top row: date + teams + autofill ─────────────────────────────────────────
c_date, c_away, c_home, c_btn = st.columns([2, 2, 2, 1])

with c_date:
    latest_date = df["date"].max()
    default_date = (
        (latest_date + pd.Timedelta(days=1)).date()
        if pd.notna(latest_date)
        else date.today()
    )
    game_date = st.date_input("比賽日期", value=default_date)

with c_away:
    away_team = st.selectbox("客隊", teams, index=0, key="away_team_sel")

with c_home:
    home_opts = [t for t in teams if t != away_team]
    home_team = st.selectbox("主隊", home_opts, index=0, key="home_team_sel")

prediction_date = pd.Timestamp(game_date)

# ── Session state initialisation ──────────────────────────────────────────────
def _init_side(side: str) -> None:
    if f"{side}_stats" not in st.session_state:
        st.session_state[f"{side}_stats"] = {f: "" for f, _, _ in INPUT_FIELDS}
    if f"{side}_starter" not in st.session_state:
        st.session_state[f"{side}_starter"] = ""
    if f"{side}_bullpen" not in st.session_state:
        st.session_state[f"{side}_bullpen"] = [""] * 5
    if f"{side}_lineup" not in st.session_state:
        st.session_state[f"{side}_lineup"] = [""] * 9


_init_side("away")
_init_side("home")

# ── Auto-fill button ──────────────────────────────────────────────────────────
with c_btn:
    st.write("")
    st.write("")
    do_autofill = st.button("Auto-fill", use_container_width=True)

if do_autofill:
    for side, team in [("away", away_team), ("home", home_team)]:
        try:
            source_row, source_side = latest_team_row(df, team, side, prediction_date)
            vals = profile_values(source_row, source_side)
            new_stats = {k: value_to_text(v) for k, v in vals.items()}
            st.session_state[f"{side}_stats"] = new_stats
            # Fix: also push values into the widget keys so text_input refreshes
            for suffix, text_val in new_stats.items():
                st.session_state[f"{side}_inp_{suffix}"] = text_val
        except Exception:
            pass

        # Auto-select starter with lowest ERA
        try:
            team_str: str = team or ""
            starter_opts = pitcher_options(pitchers, team_str, prediction_date, starter_only=True)
            if starter_opts:
                best = starter_opts[0]
                best_era = float("inf")
                for name in starter_opts:
                    prow = latest_pitcher_stats(pitchers, team_str, name, prediction_date)
                    if prow is not None:
                        try:
                            era = float(prow["ERA"])  # type: ignore[arg-type]
                            if era < best_era:
                                best_era = era
                                best = name
                        except (TypeError, ValueError):
                            pass
                st.session_state[f"{side}_starter"] = best
                prow2 = latest_pitcher_stats(pitchers, team_str, best, prediction_date)
                if prow2 is not None:
                    for k, col in [("starter_ERA", "ERA"), ("starter_WHIP", "WHIP"), ("starter_FIP", "FIP")]:
                        try:
                            val = f"{float(prow2[col]):.3f}"  # type: ignore[arg-type]
                        except (TypeError, ValueError):
                            val = ""
                        st.session_state[f"{side}_stats"][k] = val
                        st.session_state[f"{side}_inp_{k}"] = val
        except Exception:
            pass

    st.rerun()

# ── Team panel helper ─────────────────────────────────────────────────────────
def team_panel(side: str, team: str) -> None:
    label = "客隊" if side == "away" else "主隊"
    st.subheader(f"{label} ｜ {team}")

    # Starter
    _OTHER = "其他（新人／手動輸入）"
    starter_opts = pitcher_options(pitchers, team, prediction_date, starter_only=True)
    cur_starter = st.session_state[f"{side}_starter"]
    starter_list = ["（未選）"] + starter_opts + [_OTHER]
    starter_idx = starter_list.index(cur_starter) if cur_starter in starter_list else 0
    starter = st.selectbox("先發投手", starter_list, index=starter_idx, key=f"{side}_starter_sel")
    st.session_state[f"{side}_starter"] = starter

    if starter == _OTHER:
        # Show league-average defaults as starting point for manual entry
        _LEAGUE_AVG = {"starter_ERA": "4.50", "starter_WHIP": "1.35", "starter_FIP": "4.60"}
        st.caption("新人／無歷史紀錄：已填入聯盟平均值，請自行調整下方數值。")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("ERA（預設）", "4.50")
        mc2.metric("WHIP（預設）", "1.35")
        mc3.metric("FIP（預設）", "4.60")
        for k, v in _LEAGUE_AVG.items():
            if not st.session_state[f"{side}_stats"].get(k):
                st.session_state[f"{side}_stats"][k] = v
                st.session_state[f"{side}_inp_{k}"] = v
    elif starter and starter != "（未選）":
        row = latest_pitcher_stats(pitchers, team, starter, prediction_date)
        if row is not None:
            s_era = float(row["ERA"]) if pd.notna(row["ERA"]) else float("nan")
            s_whip = float(row["WHIP"]) if pd.notna(row["WHIP"]) else float("nan")
            s_fip = float(row["FIP"]) if pd.notna(row["FIP"]) else float("nan")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("ERA", f"{s_era:.2f}")
            mc2.metric("WHIP", f"{s_whip:.2f}")
            mc3.metric("FIP", f"{s_fip:.2f}")
            for _k, _v in [
                ("starter_ERA", f"{s_era:.3f}"),
                ("starter_WHIP", f"{s_whip:.3f}"),
                ("starter_FIP", f"{s_fip:.3f}"),
            ]:
                st.session_state[f"{side}_stats"][_k] = _v
                st.session_state[f"{side}_inp_{_k}"] = _v

    # Bullpen
    with st.expander("牛棚投手（選填）"):
        bullpen_opts = pitcher_options(pitchers, team, prediction_date, starter_only=False)
        new_bullpen: list[str] = []
        bp_cols = st.columns(5)
        for i, col in enumerate(bp_cols):
            cur = st.session_state[f"{side}_bullpen"][i]
            bp_list = [""] + bullpen_opts
            bp_idx = bp_list.index(cur) if cur in bp_list else 0
            val = col.selectbox(f"後援{i+1}", bp_list, index=bp_idx, key=f"{side}_bp_{i}")
            new_bullpen.append(val)
        st.session_state[f"{side}_bullpen"] = new_bullpen

        selected_bp = [n for n in new_bullpen if n]
        if selected_bp:
            try:
                bp_stats, _ = bullpen_average_from_players(
                    pitchers, team, selected_bp, prediction_date
                )
                bc1, bc2 = st.columns(2)
                bc1.metric("牛棚平均 ERA", f"{bp_stats['bullpen_ERA']:.2f}")
                bc2.metric("牛棚平均 WHIP", f"{bp_stats['bullpen_WHIP']:.2f}")
                st.session_state[f"{side}_stats"]["bullpen_ERA"] = f"{bp_stats['bullpen_ERA']:.3f}"
                st.session_state[f"{side}_stats"]["bullpen_WHIP"] = f"{bp_stats['bullpen_WHIP']:.3f}"
            except Exception:
                pass

    # Lineup
    with st.expander("打線（選填）"):
        lineup_opts = player_options(lineups, team, prediction_date)
        new_lineup: list[str] = []
        lu_cols = st.columns(3)
        for i in range(9):
            cur = st.session_state[f"{side}_lineup"][i]
            lu_list = [""] + lineup_opts
            lu_idx = lu_list.index(cur) if cur in lu_list else 0
            val = lu_cols[i % 3].selectbox(
                f"{i+1}棒", lu_list, index=lu_idx, key=f"{side}_lu_{i}"
            )
            new_lineup.append(val)
        st.session_state[f"{side}_lineup"] = new_lineup

        selected_lu = [n for n in new_lineup if n]
        if selected_lu:
            try:
                lu_stats, _ = lineup_average_from_players(
                    lineups, team, selected_lu, prediction_date
                )
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("打線平均 OPS", f"{lu_stats['lineup_OPS']:.3f}")
                lc2.metric("打線平均 OBP", f"{lu_stats['lineup_OBP']:.3f}")
                lc3.metric("打線平均 SLG", f"{lu_stats['lineup_SLG']:.3f}")
                for k, v in lu_stats.items():
                    st.session_state[f"{side}_stats"][k] = f"{v:.3f}"
            except Exception:
                pass

    # Manual stats
    st.markdown("**統計數值（可手動調整）**")
    stats = st.session_state[f"{side}_stats"]
    new_stats: dict[str, str] = {}
    sc1, sc2 = st.columns(2)
    for idx, (suffix, label, hint) in enumerate(INPUT_FIELDS):
        col = sc1 if idx % 2 == 0 else sc2
        new_stats[suffix] = col.text_input(
            label,
            value=stats.get(suffix, ""),
            placeholder=hint,
            key=f"{side}_inp_{suffix}",
        )
    st.session_state[f"{side}_stats"] = new_stats


# ── Two-column team panels ────────────────────────────────────────────────────
st.divider()

# Big team logos above panels
logo_col_away, logo_vs, logo_col_home = st.columns([3, 1, 3])
with logo_col_away:
    st.markdown(
        f'<div style="text-align:center;">{_team_logo_html(away_team, 96)}'
        f'<br><b style="font-size:1.1em;">{away_team}（客）</b></div>',
        unsafe_allow_html=True,
    )
with logo_vs:
    st.markdown(
        '<div style="text-align:center;padding-top:28px;font-size:2em;font-weight:bold;">VS</div>',
        unsafe_allow_html=True,
    )
with logo_col_home:
    st.markdown(
        f'<div style="text-align:center;">{_team_logo_html(home_team, 96)}'
        f'<br><b style="font-size:1.1em;">{home_team}（主）</b></div>',
        unsafe_allow_html=True,
    )

st.write("")
col_away_panel, col_home_panel = st.columns(2)
with col_away_panel:
    team_panel("away", away_team)
with col_home_panel:
    team_panel("home", home_team)

# ── Metrics comparison table ──────────────────────────────────────────────────
st.divider()
st.subheader("主客隊指標對比")

compare_rows = []
for suffix, label, _ in INPUT_FIELDS:
    try:
        h = float(st.session_state["home_stats"].get(suffix, ""))
        a = float(st.session_state["away_stats"].get(suffix, ""))
        lower_better = any(x in suffix for x in ("ERA", "WHIP", "FIP"))
        home_better = (h < a) if lower_better else (h > a)
        edge = f"← {home_team}" if home_better else f"{away_team} →"
        if abs(h - a) < 0.0005:
            edge = "—"
        compare_rows.append({
            "指標": label,
            f"{away_team}（客）": f"{a:.3f}",
            "優勢": edge,
            f"{home_team}（主）": f"{h:.3f}",
        })
    except (ValueError, TypeError):
        compare_rows.append({
            "指標": label,
            f"{away_team}（客）": st.session_state["away_stats"].get(suffix, "—"),
            "優勢": "—",
            f"{home_team}（主）": st.session_state["home_stats"].get(suffix, "—"),
        })

if compare_rows:
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

# ── Predict button ────────────────────────────────────────────────────────────
st.divider()
predict_clicked = st.button("⚾ 預測勝負", type="primary", use_container_width=True)

if predict_clicked:
    # Parse inputs
    try:
        manual_values: dict[str, dict[str, float]] = {"home": {}, "away": {}}
        for side in ("home", "away"):
            for suffix, label, _ in INPUT_FIELDS:
                raw = st.session_state[f"{side}_stats"].get(suffix, "")
                manual_values[side][suffix] = parse_number(f"{side} {label}", raw)
    except ValueError as e:
        st.error(f"輸入錯誤：{e}")
        st.stop()

    x_pred = build_manual_prediction_row(feature_cols, manual_values)
    prob_home = float(model.predict_proba(x_pred)[0, 1])
    prob_away = 1.0 - prob_home
    pick = home_team if prob_home >= 0.5 else away_team
    conf = confidence_label(prob_home)

    # ── Result metrics ────────────────────────────────────────────────────────
    st.subheader("預測結果")
    r1, r2, r3 = st.columns(3)
    r1.metric(f"主隊 {home_team} 勝率", f"{prob_home*100:.1f}%")
    r2.metric(f"客隊 {away_team} 勝率", f"{prob_away*100:.1f}%")
    r3.metric("預測勝隊", pick, delta=f"信心：{conf}")

    st.progress(
        prob_home,
        text=f"← 主隊 {home_team} {prob_home*100:.1f}%  |  {prob_away*100:.1f}% {away_team} 客隊 →",
    )

    # ── SHAP ─────────────────────────────────────────────────────────────────
    try:
        import shap as _shap

        imputer = model.named_steps["imputer"]
        rf = model.named_steps["rf"]
        x_transformed = imputer.transform(x_pred)
        explainer = _shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(x_transformed)

        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        sv_series = pd.Series(sv, index=feature_cols)
        top10_idx = sv_series.abs().sort_values(ascending=False).head(10).index
        sv_top = sv_series[top10_idx].sort_values()

        vals = np.array(sv_top.values, dtype=float)
        zh_labels = [_feat_label(str(f)) for f in sv_top.index]
        max_abs = float(np.abs(vals).max()) or 1.0

        # Map each value to a position in RdBu_r colormap: -max→deep blue, +max→deep red
        cmap = plt.get_cmap("RdBu_r")
        bar_colors = [cmap(0.5 + 0.48 * v / max_abs) for v in vals]

        BG = "#ffffff"
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
        ax.set_facecolor("#f8f9fa")

        bars = ax.barh(
            zh_labels, vals, color=bar_colors,
            edgecolor="none", height=0.62
        )

        # Value labels at end of each bar
        for bar, v in zip(bars, vals):
            x_pos = v + (max_abs * 0.02 if v >= 0 else -max_abs * 0.02)
            ha = "left" if v >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{v:+.3f}", va="center", ha=ha,
                    color="#333333", fontsize=8.5, fontweight="bold")

        # Zero line
        ax.axvline(0, color="#aaaaaa", linewidth=1.2, zorder=0)

        # Subtle x-grid
        ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.6, linestyle="--")
        ax.set_axisbelow(True)

        # Axis labels & ticks
        ax.set_xlabel("← 有利客隊　　　SHAP 值　　　有利主隊 →",
                      color="#555555", fontsize=9.5, labelpad=8)
        ax.tick_params(axis="y", colors="#222222", labelsize=9.5, pad=6)
        ax.tick_params(axis="x", colors="#888888", labelsize=8)

        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colour legend chips
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0.78, 0.08, 0.18, 0.85), label=f"有利主隊 {home_team or ''}"),
            Patch(facecolor=(0.13, 0.47, 0.71, 0.85), label=f"有利客隊 {away_team or ''}"),
        ]
        ax.legend(handles=legend_elements, loc="lower right",
                  framealpha=0.7, fontsize=8.5,
                  labelcolor="#222222", edgecolor="#cccccc")

        plt.tight_layout(pad=1.5)

        st.subheader("特徵貢獻分析（SHAP）")
        st.pyplot(fig)
        plt.close()

        # ── AI explanation: Gemini first, fallback to rule-based ─────────────
        st.subheader("AI 特徵解讀")
        gemini_ok = False
        import warnings
        try:
            from dotenv import load_dotenv
            load_dotenv()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                top3_idx = sv_series.abs().sort_values(ascending=False).head(3).index
                feat_detail_lines = "\n".join(
                    f"- {_feat_label(str(f))}：SHAP={float(sv_series.loc[f]):+.4f}，"  # type: ignore[arg-type]
                    f"{'有利主隊' if float(sv_series.loc[f]) > 0 else '有利客隊'}"  # type: ignore[arg-type]
                    for f in top3_idx
                )
                prompt = f"""你是職棒數據分析師，擅長用繁體中文解說機器學習預測。

比賽資訊：
- 客隊：{away_team}
- 主隊：{home_team}
- 模型預測主隊勝率：{prob_home*100:.1f}%（信心：{conf}）

SHAP Top 3 關鍵特徵（來自隨機森林模型）：
{feat_detail_lines}

請用 Markdown 格式輸出以下結構（繁體中文，語氣口語清晰）：

## 預測摘要
一句話說明預測結果與信心。

## 關鍵特徵解讀
針對上方每個特徵，各寫一段：
### [特徵名稱]
- **是什麼**：這個指標代表什麼意思
- **為什麼重要**：為何選用此特徵預測勝負
- **本場影響**：這場比賽中此特徵如何影響預測

## 結論
一句話總結為何模型選擇這個預測結果。

注意：不要加開場白，直接從「## 預測摘要」開始。"""

                _models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash"]
                for _m in _models_to_try:
                    try:
                        with st.spinner(f"Gemini 分析中…"):
                            response = genai.GenerativeModel(_m).generate_content(prompt)
                        st.markdown(response.text)
                        gemini_ok = True
                        break
                    except Exception as _me:
                        if _m == _models_to_try[-1]:
                            raise _me
                        continue
        except Exception as _ge:
            err_str = str(_ge)
            if "429" in err_str or "quota" in err_str.lower():
                st.caption("（Gemini 配額不足，改用本地 SHAP 分析）")

        if not gemini_ok:
            fallback = _shap_fallback_explanation(sv_series, home_team or "", away_team or "", prob_home, conf)
            st.markdown(fallback)

    except Exception as e:
        st.warning(f"SHAP 計算失敗：{e}")
