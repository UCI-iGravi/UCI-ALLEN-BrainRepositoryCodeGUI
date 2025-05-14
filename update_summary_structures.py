import numpy as np
import pandas as pd
import sys


def get_row_at_region(df, key):
    """ For use with count_df and region_df (cell_region_count.csv and region_counts.csv) """
    return df[df['region'] == key]


def get_row_at_parcellation_idx(df, idx):
    """ For use only with CCF_all_regions / 1_adult_mouse_brain_graph_mapping.csv """
    return df[df['parcellation_index'] == idx]


def get_row_at_identifier(df, id):
    """ For use only with summary_structures.csv """
    if "MBA:" not in str(id):
        id = "MBA:" + str(id)
    return df[df['identifier'] == id]


def get_rows_containing_value_in_path(df, value, column_name='structure_id_path', delimiter='/'):
    """ Function to get all rows where a value is contained in a column with delimited values """
    #pattern = f'(^|{delimiter}){value}({delimiter}|$)'
    pattern = f'{delimiter}{value}{delimiter}'
    return df.loc[df[column_name].str.contains(pattern, regex=True)]


def update_summary_structures(region_df, ccf_all_regions_df, ccf_summary_structures_df):
    """ Updates the dataframe for summary_structures.csv with the counts of all of the regions in the cell counting output.

    Args:
        region_df (_type_): _description_
        ccf_all_regions_df (_type_): _description_
        ccf_summary_structures_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Make copies of the dataframes so we don't modify the originals
    region_df = region_df.copy()
    ccf_all_regions_df = ccf_all_regions_df.copy()
    ccf_summary_structures_df = ccf_summary_structures_df.copy()

    # Add an extra column to region_df for keeping track of ids
    region_df['id'] = np.nan

    # Fill in default values of CCF counts
    ccf_summary_structures_df['count'] = 0

    # However, some regions in region_count cannot be found in summary structures 
    # because the region level is smaller than the 300 structures we want. 
    # We need to figure out for this region (e.g. dentate gyrus, granule cell layer, region number/parcellation_index 622, id=632 
    # is the lower level region of which regions inside 300 
    # we need to first go over each row inside 300 summary regions.csv, based on the id number, check the column 
    # structure_id_path in 1_adult_mouse_brain_graph_mapping.csv
    
    # Go through each row of summary structures
    for index, row in ccf_summary_structures_df.iterrows():
        # Grab the identifier if one exists, otherwise ignore that row, it is unassigned. 
        if row['identifier'] is not np.nan:
            curr_identifier = int(row['identifier'].replace("MBA:", ""))  # Remove the MBA: string
        
            # Get all of the rows in 1_adult_mouse_brain_graph_mapping.csv that contain the current identifier
            # in the structure_id_path column
            all_region_rows = get_rows_containing_value_in_path(ccf_all_regions_df, curr_identifier)
            parcellation_indices = all_region_rows['parcellation_index'].tolist()
            
            # Get the counts of all of the matching parcellation indices and combine them in total_count
            total_count = 0
            for p in parcellation_indices:
                region_df_row = get_row_at_region(region_df, p)
                if not region_df_row.empty:
                    total_count += int(region_df_row['count'].iloc[0])

            # Now save the combined counts to their corresponding row in summary structures
            ccf_summary_structures_df.at[index, 'count'] = total_count
    return ccf_summary_structures_df


if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) != 5:
        HELP_STR = "This script updates the summary structures file with the counts from the cell region count file. Newer cell counts done on the GUI have incorporated this function, " + \
                   "there is no need to run this script unless you have older data.\n\n" + \
                   "Usage: python update_summary_structures.py <region_df_path> <mouse_graph_map_path> <summary_structures_path> <output_path>\n" + \
                   "<region_df_path>: the outputted csv cell counts file path\n" + \
                   "<mouse_graph_map_path>: the CCF adult mouse brain graph mapping CSV file path\n" + \
                   "<summary_structures_path>: the CCF summary structures file path\n" + \
                   "<output_path>: where to save the outputted updated summary structures file\n\n" + \
                   "Example: python update_summary_structures.py cell_counts/cell_region_count.csv CCF_DATA/1_adult_mouse_brain_graph_mapping.csv CCF_DATA/300_summary_structures.csv updated_summary_structures.csv"
        print(HELP_STR)
        sys.exit(1)
    
    region_df_path = sys.argv[1]
    ccf_all_regions_df_path = sys.argv[2]
    ccf_summary_structures_df_path = sys.argv[3]
    output_path = sys.argv[4]
    
    # Read in the data
    region_df = pd.read_csv(region_df_path)
    ccf_all_regions_df = pd.read_csv(ccf_all_regions_df_path)
    ccf_summary_structures_df = pd.read_csv(ccf_summary_structures_df_path)
    
    # Update the summary structures dataframe
    ccf_summary_structures_df = update_summary_structures(region_df, ccf_all_regions_df, ccf_summary_structures_df)
    ccf_summary_structures_df.to_csv(output_path, index=False)