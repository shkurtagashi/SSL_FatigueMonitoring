from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class Explorer:
    def __init__(self, files: List[str | PathLike], *, groups_of=8, resolution='minutes', do_fix=True):
        assert len(files) % groups_of == 0

        self.resolution = resolution
        self.df = None
        self.groups = [tuple(files[i:i + groups_of]) for i in range(0, len(files), groups_of)]
        self.stats = []
        self.aggregate_stats = None
        self.do_fix = do_fix

    def load(self, files: Tuple[str | PathLike]):
        df_list = [pd.read_parquet(file) for file in files]
        self.df = pd.concat(df_list, ignore_index=True)

        def get_length_or_none(item):
            try:
                return len(item)
            except TypeError:
                return None

        def arrays_have_expected_length(row):
            lengths = [get_length_or_none(row[col]) for col in self.df.columns if isinstance(row[col], np.ndarray)]
            expected = [length == self.expected_length for length in lengths if length is not None]
            return all(expected)

        expected_lengths = self.df.apply(arrays_have_expected_length, axis=1)
        assert expected_lengths.all()

    @property
    def expected_length(self):
        """For now, the number of minutes in a week"""
        if self.resolution in 'minutes':
            return 7 * 24 * 60
        if self.resolution in 'hours':
            return 7 * 24
        if self.resolution in 'days':
            return 7
        raise ValueError(f"Invalid resolution: {self.resolution}")

    def do(self):
        for group in tqdm(self.groups):
            self.load(group)
            if self.do_fix: self.fix()
            else: print("Not fixing")
            self.do_current(axis=1)
        self.visualize()

    def fix(self):
        if "steps" in self.df.columns:
            steps_means = 6
            steps_std = 20  # it's around 17, actually, but let's overestimate it
            allowed = 20  # 20 STD is a lot,
            # and 400 steps per minute is higher than Usain Bolt and other sprinters' cadence

            def process_arrays(row):
                steps = row['steps'].copy()
                missing_steps = row['missing_steps'].copy()

                # TODO steps also above 0
                mask = ~((steps_means - allowed * steps_std < steps) & (steps < steps_means + allowed * steps_std))

                steps[mask] = 0
                missing_steps[mask] = True

                return steps, missing_steps

            # .apply() will apply the function to each row, axis=1 indicates row-wise operation.
            missing_old = np.sum(np.array(self.df['missing_steps'].values.tolist()))
            mean_old = np.mean(
                np.array(self.df['steps'].values.tolist())[~np.array(self.df['missing_steps'].values.tolist())])

            self.df['steps'] = self.df.apply(lambda row: process_arrays(row)[0], axis=1)
            self.df['missing_steps'] = self.df.apply(lambda row: process_arrays(row)[1], axis=1)

            missing_new = np.sum(np.array(self.df['missing_steps'].values.tolist()))
            mean_new = np.mean(
                np.array(self.df['steps'].values.tolist())[~np.array(self.df['missing_steps'].values.tolist())])
            print(
                f"Set 'steps' which are more than {allowed} STDs away from the mean to missing. "
                f"{missing_new - missing_old} added as missing, mean {mean_old:.4f} -> {mean_new:.4f}")

    def visualize(self):
        print(self.df.columns.values.tolist())
        self.calculate_aggregate_stats()

        # Iterate over the outer dictionary and the axes
        for outer_key, sub_dict in self.aggregate_stats.items():
            # Create a new figure for each sub-dictionary
            fig, ax = plt.subplots()

            data_for_plot = []
            category_names = []
            for inner_key, list_values in sub_dict.items():
                data_for_plot.extend(list_values)
                category_names.extend([inner_key] * len(list_values))

            # Create a DataFrame
            df = pd.DataFrame({
                'value': data_for_plot,
                'category': category_names
            })

            # Create a box plot
            sns.boxplot(x='category', y='value', data=df, ax=ax)

            mean_values = df.groupby('category')['value'].mean()
            for i, category in enumerate(df['category'].unique()):
                ax.annotate(
                    f"{mean_values[category]:.4f}", (i, mean_values[category]),
                    ha='center', size='x-small', color='black', weight='semibold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

            if all('percentages' in name for name in category_names):
                ax.set_ylim(0 - 1e-2, 1 + 1e-2)

            ax.set_title(outer_key)

            for i, label in enumerate(ax.get_xticklabels()):
                x, y = label.get_position()
                if i % 2 == 1:
                    label.set_y(y - 0.04)

            fig.tight_layout()
        plt.show()

    def do_current(self, *, axis):
        assert self.df is not None
        # axis = None is aggregated by groups, axis = 1 is aggregated by users
        stats = {}

        # "['participant_id' 'start' 'end' 'steps' 'heart_rate' 'missing_heart_rate'
        #  'missing_steps' 'sleep_classic_0' 'sleep_classic_1' 'sleep_classic_2'
        #  'sleep_classic_3' '__index_level_0__' 'id']"

        try:
            assert all(
                [x in self.df.columns for x in ["steps", "heart_rate",
                                                *[f"sleep_classic_{i}" for i in range(0, 3 + 1)]
                                                ]
                 ]
            )
        except AssertionError:
            print(self.df.columns.values.tolist())
            quit()

        for col in ["steps", "heart_rate"]:
            stats[col] = self.get_column_stats(col, col_mask=f"missing_{col}", axis=axis)

        sc0 = self.df["sleep_classic_0"]
        sc1 = self.df["sleep_classic_1"]
        sc2 = self.df["sleep_classic_2"]
        sc3 = self.df["sleep_classic_3"]
        s = np.array([sc0.values.tolist(), sc1.values.tolist(), sc2.values.tolist(), sc3.values.tolist()])
        s0 = s[0]
        s1 = s[1]
        s2 = s[2]
        s3 = s[3]
        or_sleep = (s0 | s1 | s2 | s3).mean()
        total_sleep = s0.mean() + s1.mean() + s2.mean() + s3.mean()
        if total_sleep > or_sleep:
            try:
                assert self.resolution == "hours"  # minutes should have no overlap
            except AssertionError as ae:
                print(ae)

        stats["sleep"] = dict()
        for col in [f"sleep_classic_{i}" for i in range(0, 3 + 1)]:
            stats["sleep"][col + "_percentage"] = np.mean(np.array(self.df[col].values.tolist()), axis=axis)

        self.stats.append(stats)

    def get_column_stats(self, col: str, col_mask: str, axis=None):
        data = np.array(self.df[col].values.tolist())
        mask = np.array(self.df[col_mask].values.tolist())
        data = data.astype(int)
        mask = mask.astype(bool)

        # if col == 'steps':
        #     self.display_counts_scattered(data.flatten())

        masked = np.ma.masked_array(data, mask)

        # filled(np.nan)
        masked_min = np.min(masked, axis=axis).compressed()
        masked_mean = np.mean(masked, axis=axis).compressed()
        masked_max = np.max(masked, axis=axis).compressed()
        masked_std = np.std(masked, axis=axis).compressed()

        return dict(min=masked_min, mean=masked_mean, max=masked_max, std=masked_std,
                    available=(1 - mask.mean(axis=axis)))

    def display_counts_scattered(self, data):
        values, counts = np.unique(data, return_counts=True)
        plt.scatter(values, counts, alpha=0.5)
        plt.yscale('log')
        plt.show()

    def display_scatter(self, what, where):
        plt.scatter(x=where, y=what, alpha=0.5)
        plt.ylim(0, 240)
        plt.show()

    def display_line(self, data, w=None):
        plt.plot(data)
        if w is not None:
            plt.plot([0]*w + [np.mean(data[i-w:i+w]) for i in range(w, len(data) - w)] + [0]*w)
        # plt.yscale('log')
        plt.ylim(0, 240)
        plt.show()

    def calculate_aggregate_stats(self):
        keys = ['steps', 'heart_rate']
        subkeys = [
            ('mins', 'min'),
            ('means', 'mean'),
            ('maxes', 'max'),
            ('stds', 'std'),
            ('availables', 'available'),
        ]

        special_keys = ["sleep"]
        special_subkeys = [
            (f'sleep_classic_{i}_percentages', f'sleep_classic_{i}_percentage') for i in range(0, 3 + 1)
        ]

        self.aggregate_stats = dict()
        for key in keys:
            self.aggregate_stats[key] = {subkey_plural: [] for subkey_plural, subkey_singular in subkeys}
        for key in special_keys:
            self.aggregate_stats[key] = {subkey_plural: [] for subkey_plural, subkey_singular in special_subkeys}

        for stat in self.stats:
            for key in keys:
                for subkey_plural, subkey_singular in subkeys:
                    v = stat[key][subkey_singular]
                    if isinstance(v, (list, np.ndarray)):
                        self.aggregate_stats[key][subkey_plural].extend(v)
                    else:
                        self.aggregate_stats[key][subkey_plural].append(v)

            for key in special_keys:
                for subkey_plural, subkey_singular in special_subkeys:
                    v = stat[key][subkey_singular]
                    if isinstance(v, (list, np.ndarray)):
                        self.aggregate_stats[key][subkey_plural].extend(v)
                    else:
                        self.aggregate_stats[key][subkey_plural].append(v)


dirname = Path("data/processed/split_2020_02_10/train_7_day")
filenames = [path for path in os.listdir(dirname) if os.path.splitext(path)[1] == '.parquet']
print(dirname)
Explorer([dirname / file for file in filenames], groups_of=8, resolution='minutes', do_fix=False).do()
