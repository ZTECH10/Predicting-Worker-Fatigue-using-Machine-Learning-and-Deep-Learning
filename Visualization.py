# Based on reading the book  "Machine Learning for the Quantified Self: On the Art of Learning from Sensory Data"

import matplotlib.colors as cl
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram
import itertools
from scipy.optimize import curve_fit
import re
import math
import sys
from pathlib import Path
import dateutil


class Visualization:

    point_displays = ['+', 'x']  # '*', 'd', 'o', 's', '<', '>']
    line_displays = ['-']  # , '--', ':', '-.']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # Set some initial attributes to define and create a save location for the images.
    def __init__(self, module_path='.py'):
        subdir = Path(module_path).name.split('.')[0]

        self.plot_number = 1
        self.figures_dir = Path('figures') / subdir
        self.figures_dir.mkdir(exist_ok=True, parents=True)

    def save(self, plot_obj, file_name, formats=('png',)):  # 'svg'

        # fig_name = f'figure_{self.plot_number}'
        fig_name = file_name
        for format in formats:
            save_path = self.figures_dir / \
                f'{self.plot_number}__{fig_name}.{format}'
            plot_obj.savefig(save_path)
            print(f'Figure saved to {save_path}')
        self.plot_number += 1

    def plot_dataset(self, data_table, columns, match, display, file_name):
        names = list(data_table.columns)

        # Create subplots if more columns are specified.
        if len(columns) > 1:
            f, xar = plt.subplots(len(columns), sharex=True,
                                  sharey=False, figsize=(40, 40))
        else:
            f, xar = plt.subplots()
            xar = [xar]

        f.subplots_adjust(hspace=0.4)

        xfmt = md.DateFormatter('%H:%M')

        # Pass through the columns specified.
        for i in range(0, len(columns)):
            xar[i].xaxis.set_major_formatter(xfmt)
            xar[i].set_prop_cycle(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
            # if a column match is specified as 'exact', select the column name(s) with an exact match.
            # If it's specified as 'like', select columns containing the name.

            # We can match exact (i.e. a columns name is an exact name of a columns or 'like' for
            # which we need to find columns names in the dataset that contain the name.
            if match[i] == 'exact':
                relevant_cols = [columns[i]]
            elif match[i] == 'like':
                relevant_cols = [
                    name for name in names if columns[i] == name[0:len(columns[i])]]
            else:
                raise ValueError(
                    "Match should be 'exact' or 'like' for " + str(i) + ".")

            max_values = []
            min_values = []

            # Pass through the relevant columns.
            for j in range(0, len(relevant_cols)):
                # Create a mask to ignore the NaN and Inf values when plotting:
                mask = data_table[relevant_cols[j]].replace(
                    [np.inf, -np.inf], np.nan).notnull()
                max_values.append(data_table[relevant_cols[j]][mask].max())
                min_values.append(data_table[relevant_cols[j]][mask].min())

                # Display point, or as a line
                if display[i] == 'points':
                    xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                                self.point_displays[j % len(self.point_displays)])
                else:
                    xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                                self.line_displays[j % len(self.line_displays)])

            xar[i].tick_params(axis='y', labelsize=10)
            xar[i].legend(relevant_cols, fontsize='xx-small', numpoints=1, loc='upper center',
                          bbox_to_anchor=(0.5, 1.3), ncol=len(relevant_cols), fancybox=True, shadow=True, prop={'size': 25})

            xar[i].set_ylim([min(min_values) - 0.1*(max(max_values) - min(min_values)),
                             max(max_values) + 0.1*(max(max_values) - min(min_values))])

        # Make sure we get a nice figure with only a single x-axis and labels there.
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel('Time')
        self.save(plt, file_name)
        plt.show()

    def plot_xy(self, x, y, method='plot', xlabel=None, ylabel=None, xlim=None, ylim=None, names=None, line_styles=None, loc=None, title=None, file_name='test'):
        for input in x, y:
            if not hasattr(input[0], '__iter__'):
                raise TypeError(
                    'x/y should be given as a list of lists of coordinates')

        plot_method = getattr(plt, method)
        for i, (x_line, y_line) in enumerate(zip(x, y)):

            plot_method(x_line, y_line, line_styles[i]) if line_styles is not None else plt.plot(
                x_line, y_line)

            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            if title is not None:
                plt.title(title)
            if names is not None:
                plt.legend(names)

        self.save(plt, file_name)
        plt.show()
