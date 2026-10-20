# PULSE — News Article Popularity Analysis

**EDAV Course-End Project | Python · Exploratory Data Analysis · Interactive Visualization**

PULSE is a data analytics project exploring the characteristics associated with online news article popularity. It combines statistical analysis, data visualization, and an interactive dashboard to investigate how article features relate to social sharing.

## Project Overview

The project uses the Online News Popularity dataset from the UCI Machine Learning Repository, containing 39,644 articles and 61 original columns.

### Research Questions

- How does article popularity vary across news categories?
- Are shares distributed differently across publication days?
- Are headline length, article length, sentiment, images, videos, and links associated with popularity?
- Which article characteristics have the strongest rank-based associations with shares?

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- SciPy
- Plotly
- Streamlit
- Jupyter Notebook

## Repository Structure

```text
News-Article-Popularity-Analysis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── news_popularity_analysis.ipynb
├── src/
├── dashboard/
│   └── app.py
├── outputs/
│   └── charts/
├── report/
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

Download the Online News Popularity dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/332/online+news+popularity).

Place the extracted CSV file at:

`data/raw/OnlineNewsPopularity.csv`

The raw dataset is excluded from Git tracking by default. Obtain it from UCI before running the project.

## Setup and Usage

Clone the repository and enter its directory:

```bash
git clone https://github.com/kailashmaganti/News-Article-Popularity-Analysis.git
cd News-Article-Popularity-Analysis
```

Create and activate a virtual environment on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the analysis notebook:

```bash
jupyter notebook notebooks/news_popularity_analysis.ipynb
```

Run the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

## Methodology

1. Inspect dataset structure, missing values, duplicates, and descriptive statistics.
2. Examine the distribution of social shares, including a log-transformed view.
3. Compare popularity across categories and publication days.
4. Explore relationships between article characteristics and shares.
5. Use rank correlations and nonparametric statistical testing.
6. Present the results through an interactive Streamlit dashboard.

## Important Limitations

This dataset contains historical observational data from Mashable. Statistical associations do not establish causation, and findings may not generalize to current news platforms or audiences.

## Results

Detailed findings, numerical results, and visualizations will be documented after validating the completed analysis.

