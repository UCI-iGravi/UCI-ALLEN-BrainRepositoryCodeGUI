import sys
import numpy as np
from cellAnalysis.cell_detection import createShardedPointAnnotation

if __name__ == '__main__':
    if len(sys.argv) != 3:
        HELP_STR = "Converts counted cells (inputpoints.txt) to Neuroglancer sharded points." + \
                   "Convert the raw cell counts (unscaled) if you want to use this with the raw tiffs.\n\n" + \
                   "Usage: python convert_points_to_ng.py <input_points_path> <out_dir>\n\n" + \
                   "Arguments:\n" + \
                   "<input_points_path>: Path to inputpoints.txt file\n" + \
                   "<out_dir>: Path to output the converted points to. It will create a new folder at that path called neuroglancer_sharded_annotations.\n" + \
                   "Example: python convert_points_to_ng.py inputpoints.txt output_folder."
        print(HELP_STR)
        sys.exit(1)
        
    # Read in the arguments
    input_points_path = sys.argv[1]
    out_dir = sys.argv[2]
    
    cell_locations = np.loadtxt(input_points_path, skiprows=2)
    createShardedPointAnnotation(cell_locations, out_dir) # Create sharded points for Neuroglancer layer
    sys.exit(0)