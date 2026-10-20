from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="PULSE | News Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "OnlineNewsPopularity.csv"

# --------------------------------------------------
# PREMIUM DARK THEME
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #090D16;
    --panel: #111827;
    --border: #263247;
    --text: #F0F4FC;
    --muted: #91A0B8;
    --cyan: #4DEEEA;
    --violet: #A78BFA;
    --orange: #FFB86B;
}

.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, #14233B 0%, transparent 36%),
        var(--bg);
    color: var(--text);
    font-family: 'Manrope', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #0C1220;
    border-right: 1px solid var(--border);
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #151F31, #101725);
    border: 1px solid var(--border);
    padding: 19px;
    border-radius: 16px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--cyan);
}

h1, h2, h3 {
    letter-spacing: -0.035em;
}

.pulse-brand {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.20em;
    color: var(--cyan);
    margin-bottom: 12px;
}

.hero {
    padding: 25px 0 20px 0;
}

.hero h1 {
    font-size: clamp(38px, 5vw, 65px);
    line-height: 1.05;
    font-weight: 800;
    margin: 0;
    color: #F7F9FF;
}

.hero p {
    color: var(--muted);
    font-size: 16px;
    margin-top: 14px;
    max-width: 700px;
}

.section-label {
    color: var(--cyan);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: .15em;
    text-transform: uppercase;
    margin-top: 12px;
}

div[data-testid="stTabs"] button {
    font-weight: 700;
}

hr {
    border-color: var(--border);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    category_columns = {
        "data_channel_is_lifestyle": "Lifestyle",
        "data_channel_is_entertainment": "Entertainment",
        "data_channel_is_bus": "Business",
        "data_channel_is_socmed": "Social Media",
        "data_channel_is_tech": "Technology",
        "data_channel_is_world": "World",
    }

    flags = data[list(category_columns)].fillna(0).eq(1)
    data["category"] = flags.idxmax(axis=1).map(category_columns)
    data.loc[~flags.any(axis=1), "category"] = "Other / Unclassified"

    weekday_columns = {
        "weekday_is_monday": "Monday",
        "weekday_is_tuesday": "Tuesday",
        "weekday_is_wednesday": "Wednesday",
        "weekday_is_thursday": "Thursday",
        "weekday_is_friday": "Friday",
        "weekday_is_saturday": "Saturday",
        "weekday_is_sunday": "Sunday",
    }

    day_flags = data[list(weekday_columns)].fillna(0).eq(1)
    data["weekday"] = day_flags.idxmax(axis=1).map(weekday_columns)
    data.loc[~day_flags.any(axis=1), "weekday"] = "Unknown"

    data["is_weekend_label"] = np.where(
        data["is_weekend"].eq(1), "Weekend", "Weekday"
    )

    return data


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Place OnlineNewsPopularity.csv in data/raw/.")
    st.stop()

df = load_data(DATA_PATH)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ◈ PULSE")
    st.caption("NEWS INTELLIGENCE PLATFORM")
    st.divider()

    st.markdown("### Explore the dataset")

    categories = sorted(df["category"].unique())
    selected_categories = st.multiselect(
        "News categories",
        categories,
        default=categories
    )

    days = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    selected_days = st.multiselect(
        "Publication days",
        days,
        default=days
    )

    st.markdown("### Popularity range")

    max_shares = int(df["shares"].max())
    share_limit = st.slider(
        "Maximum shares",
        min_value=1000,
        max_value=max(1000, max_shares),
        value=max(1000, max_shares),
        step=1000
    )

    st.divider()
    st.caption("SOURCE · UCI ONLINE NEWS POPULARITY")
    st.caption("Historical Mashable article data")

filtered = df[
    df["category"].isin(selected_categories)
    & df["weekday"].isin(selected_days)
    & (df["shares"] <= share_limit)
].copy()

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="pulse-brand">◈ PULSE / NEWS INTELLIGENCE</div>
    <h1>What makes news<br>travel?</h1>
    <p>
        Explore the patterns behind online article popularity.
        Investigate categories, publication timing, and the
        characteristics associated with social sharing.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

if filtered.empty:
    st.warning("No articles match these filters. Adjust your selections.")
    st.stop()

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
st.markdown('<div class="section-label">01 / DATA SNAPSHOT</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

m1.metric("ARTICLES IN VIEW", f"{len(filtered):,}")
m2.metric("MEDIAN SHARES", f"{filtered['shares'].median():,.0f}")
m3.metric("MEAN SHARES", f"{filtered['shares'].mean():,.0f}")
m4.metric("CATEGORIES", f"{filtered['category'].nunique()}")

# --------------------------------------------------
# TABS
# --------------------------------------------------
overview, categories_tab, timing_tab, features_tab,findings_tab = st.tabs([
    "◈ Overview",
    "▥ Categories",
    "◷ Publishing",
    "⌁ Article Features",
    "✦ Findings",
])

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#DCE5F5", family="Manrope"),
    margin=dict(l=20, r=20, t=55, b=20),
    height=390
)

with overview:
    st.markdown("### Popularity distribution")
    left, right = st.columns([1.4, 1])

    with left:
        hist = px.histogram(
            filtered,
            x="shares",
            nbins=55,
            title="Article Shares · Raw Scale",
            color_discrete_sequence=["#4DEEEA"]
        )
        hist.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(hist, use_container_width=True)

    with right:
        log_hist = px.histogram(
            filtered.assign(log_shares=np.log1p(filtered["shares"])),
            x="log_shares",
            nbins=45,
            title="Popularity · Log Scale",
            color_discrete_sequence=["#A78BFA"]
        )
        log_hist.update_layout(**PLOT_LAYOUT)
        log_hist.update_xaxes(title="log(1 + shares)")
        st.plotly_chart(log_hist, use_container_width=True)

    st.markdown("### Most-shared articles in the filtered dataset")

    top_articles = filtered.nlargest(10, "shares")[
        ["url", "category", "shares", "n_tokens_title", "num_imgs", "num_videos"]
    ].copy()

    top_articles.columns = [
        "Article URL", "Category", "Shares",
        "Headline Words", "Images", "Videos"
    ]

    st.dataframe(
        top_articles,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Article URL": st.column_config.LinkColumn("Article URL")
        }
    )

with categories_tab:
    st.markdown("### Category performance")

    category_summary = (
        filtered.groupby("category")
        .agg(
            Articles=("shares", "size"),
            Median_Shares=("shares", "median"),
            Mean_Shares=("shares", "mean")
        )
        .reset_index()
        .sort_values("Median_Shares", ascending=False)
    )

    bar = px.bar(
        category_summary,
        x="category",
        y="Median_Shares",
        color="Median_Shares",
        color_continuous_scale=["#35415F", "#4DEEEA"],
        hover_data=["Articles", "Mean_Shares"],
        title="Median Shares by News Category"
    )
    bar.update_layout(**PLOT_LAYOUT)
    bar.update_layout(coloraxis_showscale=False)
    st.plotly_chart(bar, use_container_width=True)

    st.dataframe(
        category_summary.round(2),
        use_container_width=True,
        hide_index=True
    )

with timing_tab:
    st.markdown("### Does publication timing matter?")

    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    day_summary = (
        filtered[filtered["weekday"] != "Unknown"]
        .groupby("weekday")
        .agg(
            Articles=("shares", "size"),
            Median_Shares=("shares", "median"),
            Mean_Shares=("shares", "mean")
        )
        .reindex(day_order)
        .dropna()
        .reset_index()
    )

    day_chart = px.bar(
        day_summary,
        x="weekday",
        y="Median_Shares",
        color="Median_Shares",
        color_continuous_scale=["#35415F", "#FFB86B"],
        hover_data=["Articles", "Mean_Shares"],
        title="Median Shares by Publication Day"
    )
    day_chart.update_layout(**PLOT_LAYOUT)
    day_chart.update_layout(coloraxis_showscale=False)
    st.plotly_chart(day_chart, use_container_width=True)

    weekend_summary = (
        filtered.groupby("is_weekend_label")["shares"]
        .agg(["count", "median", "mean"])
        .reset_index()
    )
    st.dataframe(
        weekend_summary.round(2),
        use_container_width=True,
        hide_index=True
    )

with features_tab:
    st.markdown("### Explore article characteristics")

    feature_options = {
        "Headline length": "n_tokens_title",
        "Article length": "n_tokens_content",
        "Number of images": "num_imgs",
        "Number of videos": "num_videos",
        "Number of links": "num_hrefs",
        "Title sentiment": "title_sentiment_polarity",
        "Global sentiment": "global_sentiment_polarity",
        "Subjectivity": "global_subjectivity"
    }

    feature_label = st.selectbox(
        "Select a feature",
        list(feature_options.keys())
    )

    feature_column = feature_options[feature_label]

    scatter = px.scatter(
        filtered,
        x=feature_column,
        y="shares",
        color="category",
        opacity=0.55,
        hover_data=["url", "weekday"],
        title=f"{feature_label} vs Article Shares"
    )
    scatter.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(scatter, use_container_width=True)

    spearman_r = filtered[[feature_column, "shares"]].corr(
        method="spearman"
    ).iloc[0, 1]

    st.metric(
        "Spearman rank correlation",
        f"{spearman_r:.3f}" if pd.notna(spearman_r) else "N/A"
    )

    st.caption(
        "Correlation describes association, not causation. "
        "The filtered dataset determines this statistic."
    )

# --------------------------------------------------
# FINDINGS & STATISTICAL EVIDENCE
# --------------------------------------------------
with findings_tab:
    st.markdown("### Findings from the selected data")
    st.caption(
        "Results update with the sidebar filters. "
        "These are observational associations, not causal conclusions."
    )

    # Category comparison
    category_findings = (
        filtered.groupby("category")["shares"]
        .agg(Articles="size", Median_Shares="median", Mean_Shares="mean")
        .sort_values("Median_Shares", ascending=False)
    )

    if not category_findings.empty:
        leader = category_findings.index[0]
        leader_median = category_findings.iloc[0]["Median_Shares"]

        c1, c2 = st.columns(2)
        c1.metric("Highest median-share category", leader)
        c2.metric("Its median shares", f"{leader_median:,.0f}")

        st.markdown("#### Category evidence")
        st.dataframe(
            category_findings.round(2),
            use_container_width=True
        )

    # Weekday versus weekend comparison
    weekday_values = filtered.loc[
        filtered["is_weekend"] == 0, "shares"
    ].dropna()

    weekend_values = filtered.loc[
        filtered["is_weekend"] == 1, "shares"
    ].dropna()

    st.markdown("#### Weekday vs weekend")

    if len(weekday_values) > 0 and len(weekend_values) > 0:
        u_stat, p_value = mannwhitneyu(
            weekday_values,
            weekend_values,
            alternative="two-sided"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Weekday median", f"{weekday_values.median():,.0f}")
        c2.metric("Weekend median", f"{weekend_values.median():,.0f}")
        c3.metric("Mann–Whitney p-value", f"{p_value:.4g}")

        if p_value < 0.05:
            st.info(
                "The two groups show a statistically significant "
                "difference at the 5% level. This does not establish causation."
            )
        else:
            st.info(
                "The test does not provide sufficient evidence of a "
                "difference at the 5% significance level."
            )

        st.caption(
            f"Mann–Whitney U statistic: {u_stat:,.0f}. "
            "The test compares rank distributions, not medians alone."
        )
    else:
        st.warning(
            "Both weekday and weekend articles must be present "
            "in the filtered data to perform this test."
        )

    # Feature associations
    st.markdown("#### Feature associations with popularity")

    candidate_features = [
        "n_tokens_title",
        "n_tokens_content",
        "num_imgs",
        "num_videos",
        "num_hrefs",
        "global_sentiment_polarity",
        "title_sentiment_polarity"
    ]

    association_rows = []

    for col in candidate_features:
        pair = filtered[[col, "shares"]].dropna()

        if len(pair) >= 3 and pair[col].nunique() > 1:
            rho, p = spearmanr(pair[col], pair["shares"])

            association_rows.append({
                "Feature": col,
                "Spearman correlation": rho,
                "P-value": p,
                "Sample size": len(pair)
            })

    if association_rows:
        association_table = pd.DataFrame(association_rows)
        association_table["Absolute correlation"] = (
            association_table["Spearman correlation"].abs()
        )
        association_table = association_table.sort_values(
            "Absolute correlation", ascending=False
        )

        st.dataframe(
            association_table.drop(columns="Absolute correlation").round(4),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Correlations describe associations. P-values are not adjusted "
        "for multiple comparisons, so interpret them cautiously."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "PULSE · EDAV COURSE-END PROJECT | "
    "Historical observational data; findings are not causal claims."
)
