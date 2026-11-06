"""
Statistics and trends: life expectancy and income across countries,
1952-2007 (Gapminder data, 142 countries, 12 survey years).

Produces three figures for the report:
    categorical_plot.png  - histogram of life expectancy, 1952 vs 2007
    relational_plot.png   - scatter of life expectancy against GDP per capita (2007)
    statistical_plot.png  - heatmap of median life expectancy by continent and year

Usage:
    python statistics_and_trends.py [path/to/gapminder.csv]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', [
    '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
CONTINENT_STYLE = {            # colour + marker, so identity is not colour-only
    'Africa': (ORANGE, 'o'),
    'Americas': (BLUE, 's'),
    'Asia': (AQUA, '^'),
    'Europe': (YELLOW, 'D'),
    'Oceania': (MAGENTA, 'v'),
}


def set_style():
    """Shared, quiet plotting style sized for a two-page A4 report."""
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 300, 'figure.dpi': 120,
    })


def plot_categorical_plot(df):
    """Histogram of life expectancy in 1952 and 2007 on common bins."""
    fig, ax = plt.subplots(figsize=(6.85, 2.45))
    bins = np.arange(27.5, 85.1, 2.5)
    top = max(np.histogram(df.loc[df['year'] == y, 'lifeExp'], bins)[0].max()
              for y in (1952, 2007))
    for year, colour in [(1952, BLUE), (2007, ORANGE)]:
        values = df.loc[df['year'] == year, 'lifeExp']
        ax.hist(values, bins=bins, color=colour, alpha=0.45,
                edgecolor='white', linewidth=0.8, label=f'{year}')
        ax.hist(values, bins=bins, histtype='step', color=colour,
                linewidth=1.4)
        med = values.median()
        ax.axvline(med, color=colour, linestyle='--', linewidth=1.2)
        right = year == 1952
        ax.text(med + (0.5 if right else -0.5), top * 1.15,
                f'{year} median\n{med:.1f} yrs', color=INK, fontsize=7,
                va='top', ha='left' if right else 'right')
    ax.set_xlabel('Life expectancy at birth (years)')
    ax.set_ylabel('Number of countries')
    ax.set_ylim(0, top * 1.2)
    ax.set_title('Distribution of life expectancy across 142 countries',
                 loc='left')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig('categorical_plot.png')
    plt.close(fig)


def plot_relational_plot(df):
    """Scatter of life expectancy against log GDP per capita in 2007."""
    d = df[df['year'] == 2007]
    fig, ax = plt.subplots(figsize=(6.85, 2.75))
    for continent, (colour, marker) in CONTINENT_STYLE.items():
        part = d[d['continent'] == continent]
        ax.scatter(part['gdpPercap'], part['lifeExp'], s=np.sqrt(part['pop']) / 90 + 8,
                   c=colour, marker=marker, alpha=0.8, edgecolors='white',
                   linewidths=0.6, label=continent)
    slope, intercept = np.polyfit(np.log10(d['gdpPercap']), d['lifeExp'], 1)
    xs = np.logspace(np.log10(250), np.log10(55000), 50)
    ax.plot(xs, slope * np.log10(xs) + intercept, color=INK2, linewidth=1.2,
            linestyle='--', label='Least-squares fit')
    for name, dx, dy in [('Swaziland', 1.15, -1.5), ('South Africa', 1.15, -2.5),
                         ('Botswana', 1.12, 1.5), ('Vietnam', 0.55, 3.2),
                         ('Japan', 0.52, 1.6)]:
        row = d[d['country'] == name].iloc[0]
        ax.annotate(name, (row['gdpPercap'], row['lifeExp']),
                    xytext=(row['gdpPercap'] * dx, row['lifeExp'] + dy),
                    fontsize=6.8, color=INK2,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    ax.set_xscale('log')
    ax.set_xticks([300, 1000, 3000, 10000, 30000])
    ax.set_xticklabels(['$300', '$1k', '$3k', '$10k', '$30k'])
    ax.set_xlabel('GDP per capita (inflation-adjusted US$, log scale)')
    ax.set_ylabel('Life expectancy (years)')
    ax.set_title('Life expectancy against income, 2007 (marker area ~ population)',
                 loc='left')
    ax.legend(frameon=False, ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5),
              markerscale=0.8)
    fig.tight_layout()
    fig.savefig('relational_plot.png')
    plt.close(fig)


def plot_statistical_plot(df):
    """Heatmap of median life expectancy by continent and survey year."""
    table = df.pivot_table(index='continent', columns='year',
                           values='lifeExp', aggfunc='median')
    counts = df[df['year'] == 2007]['continent'].value_counts()
    table.index = [f'{c} (n={counts[c]})' for c in table.index]
    fig, ax = plt.subplots(figsize=(6.85, 2.25))
    sns.heatmap(table, annot=True, fmt='.1f', cmap=SEQ_BLUE, vmin=35, vmax=85,
                linewidths=1.5, linecolor='white', annot_kws={'fontsize': 7},
                cbar_kws={'label': 'Median life exp. (yrs)', 'pad': 0.015},
                ax=ax)
    ax.set_xlabel('Survey year')
    ax.set_ylabel('')
    ax.tick_params(axis='both', length=0)
    ax.grid(False)
    ax.set_title('Median life expectancy by continent, 1952-2007', loc='left')
    fig.tight_layout()
    fig.savefig('statistical_plot.png')
    plt.close(fig)


def statistical_analysis(df, col):
    """Return the first four moments of a column (excess kurtosis)."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Light checks: types, missing values, duplicates, and a quick summary."""
    df = df.drop_duplicates().dropna(subset=['lifeExp', 'gdpPercap', 'pop'])
    df['year'] = df['year'].astype(int)
    print(df.describe().round(2))
    print(df[['lifeExp', 'gdpPercap', 'pop']].corr().round(3))
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    skew = 'right' if moments[2] > 0.5 else 'left' if moments[2] < -0.5 else 'not'
    kurt = ('leptokurtic' if moments[3] > 1 else
            'platykurtic' if moments[3] < -1 else 'mesokurtic')
    print(f'The data was {skew} skewed and {kurt}.')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'gapminder.csv'
    set_style()
    df = preprocessing(pd.read_csv(path))
    col = 'lifeExp'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    for year in (1952, 2007):
        print(f'\n--- {year} ---')
        writing(statistical_analysis(df[df['year'] == year], col), col)
    d = df[df['year'] == 2007]
    r, p = ss.pearsonr(np.log10(d['gdpPercap']), d['lifeExp'])
    rho, _ = ss.spearmanr(d['gdpPercap'], d['lifeExp'])
    print(f'\nPearson r (log GDP, lifeExp, 2007) = {r:.3f} (p = {p:.1e}); '
          f'Spearman rho = {rho:.3f}')


if __name__ == '__main__':
    main()
