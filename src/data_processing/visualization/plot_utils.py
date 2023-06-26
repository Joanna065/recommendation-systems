from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud


def plot_counts(data: pd.DataFrame, count_col: str, title: str, x_label=None, y_label=None,
                ann_format="{}", hue=None, figsize=(12, 6), ax_style="darkgrid", palette="Blues_d",
                rotate=False, annotate=True, order=None):
    plt.figure(figsize=figsize)
    plt.tight_layout()

    with sns.axes_style(ax_style):
        ax = sns.countplot(x=count_col, data=data, hue=hue, palette=palette, order=order)
        ax.set_title(title, fontsize=14.0)

        if annotate:
            ax = _annotate_bars(ax, ann_format)
        if rotate:
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=50,
                               horizontalalignment='right')
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        return ax


def plot_bar(data: pd.DataFrame, x: str, y: str, title: str, ann_format="{}", figsize=(12, 6),
             ax_style="darkgrid", palette="Blues_d", rotate=False, annotate=False):
    plt.figure(figsize=figsize)
    plt.tight_layout()

    with sns.axes_style(ax_style):
        ax = sns.barplot(x=x, y=y, data=data, palette=palette)
        ax.set_title(title, fontsize=14.0)

        if annotate:
            ax = _annotate_bars(ax, ann_format)
        if rotate:
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=50,
                               horizontalalignment='right')
        return ax


def plot_violin(data: pd.DataFrame, x: str, y: str, title: str, figsize=(12, 6),
                ax_style="darkgrid", palette="Blues_d"):
    plt.figure(figsize=figsize)
    plt.tight_layout()
    with sns.axes_style(ax_style):
        ax = sns.violinplot(x=x, y=y, data=data, palette=palette)
        ax.set_title(title, fontsize=14.0)

    return ax


def plot_line(data: pd.DataFrame, x: str = None, y: str = None, title: str = None, xlabel=None,
              ylabel=None, figsize=(12, 6), ax_style="darkgrid", palette=None,
              linewidth: float = 1.0):
    plt.figure(figsize=figsize)
    plt.tight_layout()
    with sns.axes_style(ax_style):
        ax = sns.lineplot(x=x, y=y, data=data, palette=palette)
        ax.set_title(title, fontsize=14.0)
        plt.setp(ax.lines, linewidth=linewidth)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    return ax


def plot_distribution(values: List[float], title: str, xlabel: str = None, hist: bool = True,
                      figsize=(12, 6), ax_style="darkgrid"):
    plt.figure(figsize=figsize)
    plt.tight_layout()
    with sns.axes_style(ax_style):
        ax = sns.distplot(values, hist=hist)
        ax.set_title(title, fontsize=14.0)
    if xlabel is not None:
        plt.xlabel(xlabel)

    return ax


def _annotate_bars(ax, ann_format: str):
    for p in ax.patches:
        ax.annotate(ann_format.format(p.get_height()),
                    xy=(p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    return ax


def change_bars_width(ax, new_value):
    for i, patch in enumerate(ax.patches):
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)
    return ax


def make_wordcloud(freq_dict, max_words=1000, mask_img_path=None):
    if mask_img_path is not None:
        movie_mask = np.array(Image.open(mask_img_path))
        wc = WordCloud(background_color="black", max_words=max_words, height=400, width=1200,
                       margin=0, mask=movie_mask)
    else:
        wc = WordCloud(background_color="black", max_words=max_words, height=400, width=1200,
                       margin=0)

    wc.generate_from_frequencies(freq_dict)
    return wc
