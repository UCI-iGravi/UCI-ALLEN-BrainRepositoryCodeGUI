import sys
import os
import csv
from collections import defaultdict

import numpy as np
import pandas as pd


def load_summary_structures(filepath: str) -> (pd.core.frame.DataFrame, set):
    """
    Loads from the summary structures file and 
    returns the table and all of its unique labels.
    """
    summary_structures = pd.read_csv(summary_structures_filepath)
    structures = set([x.lower() for x in summary_structures['name']])
    return summary_structures, structures


def load_region_counts(filepath: str) -> (pd.core.frame.DataFrame, set):
    """ 
    Loads from the region counts file and 
    returns the table and all of its unique labels.
    """
    region_counts = pd.read_csv(filepath)
    # Process the region_counts.csv
    level_1 = set(region_counts['level_1'])
    level_2 = set(region_counts['level_2'])
    level_3 = set(region_counts['level_3'])
    level_4 = set(region_counts['level_4'])
    level_5 = set(region_counts['level_5'])
    level_6 = set(region_counts['level_6'])
    level_7 = set(region_counts['level_7'])
    level_8 = set(region_counts['level_8'])
    level_9 = set(region_counts['level_9'])
    level_10 = set(region_counts['level_10'])
    region_name = set(region_counts['region name'])
    levels = level_1 | level_2 | level_3 | level_4 | level_5 | \
             level_6 | level_7 | level_8 | level_9 | level_10 | region_name
    levels = set(levels)
    levels = {x.lower() for x in levels if x==x}
    return region_counts, levels


def check_columns(keyword: str, row: pd.core.series.Series) -> bool:
    """
    For a given row, check if the keyword is within any of the level columns.
    """
    keyword = keyword.lower()
    row_labels = [str(row['level_0']).lower(), str(row['level_1']).lower(), 
                  str(row['level_2']).lower(), str(row['level_3']).lower(), \
                  str(row['level_4']).lower(), str(row['level_5']).lower(), \
                  str(row['level_6']).lower(), str(row['level_7']).lower(), \
                  str(row['level_8']).lower(), str(row['level_9']).lower(), \
                  str(row['level_10']).lower()]
    if keyword in row_labels:
        return True
    return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please specify input parameters. Example shown below:")
        print("python count_labels.py SUMMARY_STRUCTURES_PATH REGION_COUNTS_PATH OUTPUT_PATH")
        sys.exit()
    summary_structures_filepath = sys.argv[1]
    region_counts_filepath = sys.argv[2]
    output_path = os.path.dirname(region_counts_filepath) + "/summary_structures_counts.csv"
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]
    # Print input information
    print("Summary structures filepath: " + summary_structures_filepath)
    print("Region counts filepath: " + region_counts_filepath)
    print("Output csv filepath: " + output_path)
    # Process the input csv files
    summary_structures, structures = load_summary_structures(summary_structures_filepath)
    structure_count = defaultdict(int)
    region_counts, _ = load_region_counts(region_counts_filepath)
    region_counts = region_counts.fillna(0)  # Fill in NaN values with 0

    # For each label in summary_structures, check through every region_counts row
    for label in structures:
        for _, row in region_counts.iterrows():
            # If there is a match, return True, else return False.
            is_match = check_columns(label, row)
            # If a match is found, add the counts together
            if is_match:
                structure_count[label] += int(row['count'])
    structure_count = dict(structure_count)
    
    # Add counts to summary_structures
    summary_names = [x.lower() for x in summary_structures['name']]
    summary_counts = [structure_count[name] if name in structure_count.keys() else 0 for name in summary_names]
    summary_structures['count'] = summary_counts
    summary_structures.to_csv(output_path, index=False)