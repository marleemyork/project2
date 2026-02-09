'''
This script includes functions for repeated visualization techniques that I use.
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Boxplot of one column across the categories of another column
def boxplot_by_category(df, value_col, category_col, title, figsize=(10,6)):
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=category_col, y=value_col)
    sns.despine()
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Boxplot of multiple columns alongside eachother for comparison
def multi_boxplots(df, value_cols, category_col, title, figsize=(12,6),showfliers=True):
    # reshape to long format
    long_df = df.melt(id_vars=category_col, value_vars=value_cols,
                      var_name="variable", value_name="value")

    plt.figure(figsize=figsize)
    sns.boxplot(data=long_df, x="variable", y="value",hue=category_col,showfliers=showfliers)
    sns.despine()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Boxplots of multiple columns and grouped by a category, organized by all columns
# of one group next to eachother
def multi_boxplots_grouped(df, value_cols, category_col, title, figsize=(12,6),showfliers=True):
    # reshape to long format
    long_df = df.melt(
        id_vars=category_col,
        value_vars=value_cols,
        var_name="variable",
        value_name="value"
    )

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=long_df,
        x=category_col,   # categories on x-axis
        y="value",
        hue="variable",    # variables grouped within each category
        showfliers=showfliers
    )
    sns.despine()
    plt.title(title)
    plt.tight_layout()
    plt.show()
