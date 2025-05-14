""" 
Code to visualize various image formats in Neuroglancer locally.
Maintained by Andy Thai (andy.thai@uci.edu)
"""
# Standard library
import webbrowser
import glob 
import sys
import copy
import inspect

# 3rd party libraries
import numpy as np
from natsort import natsorted
import neuroglancer
import zarr
from joblib import Parallel, delayed
import nibabel as nib

# Our libraries
from tif_to_ome import readTifWrapper, readTifSection


def image_shader():
    shader_code = """
    #uicontrol invlerp normalized(range=[0, 3500], window=[0, 65535])
    void main() {
        emitGrayscale(normalized());
    }
    """
    return inspect.cleandoc(shader_code)


def volume_shader(registered: bool=False):
    if not registered:  # If not using registered data, use the following shader code
        shader_code = """
        // Controls the strength of the volume rendering opacity
        #uicontrol float brightness slider(min=-10, max=10, default=0, step=0.1)

        // Settings for brain tissue
        #uicontrol float tissueMinValue slider(min=0, max=65535, default=0, step=1)
        #uicontrol float tissueMaxValue slider(min=0, max=65535, default=400, step=1)
        #uicontrol vec3 tissueColor color(default=\"#FFFFFF\")

        // Settings for cell values
        #uicontrol float cellMinValue slider(min=0, max=65535, default=500, step=1)
        #uicontrol vec3 cellColor color(default=\"#FFFF00\")

        void main() {
            // Get data and normalize it
            float x = float(getInterpolatedDataValue().value);
            float norm = (x - tissueMinValue) / (tissueMaxValue - tissueMinValue);
            
            // Default color
            vec4 voxelColor = vec4(tissueColor, norm*exp(brightness));
            
            // Color yellow if cell activation detected
            if (x > cellMinValue) {
                voxelColor = vec4(cellColor, norm*exp(brightness));
            }
            emitRGBA(voxelColor);
        }
        """
    else:  # If using registered data, use the following shader code. Only changes are in float x line.
        shader_code = """
        // Controls the strength of the volume rendering opacity
        #uicontrol float brightness slider(min=-10, max=10, default=0, step=0.1)

        // Settings for brain tissue
        #uicontrol float tissueMinValue slider(min=0, max=65535, default=0, step=1)
        #uicontrol float tissueMaxValue slider(min=0, max=65535, default=400, step=1)
        #uicontrol vec3 tissueColor color(default=\"#FFFFFF\")

        // Settings for cell values
        #uicontrol float cellMinValue slider(min=0, max=65535, default=500, step=1)
        #uicontrol vec3 cellColor color(default=\"#FFFF00\")

        void main() {
            // Get data and normalize it
            float x = float(getInterpolatedDataValue());
            float norm = (x - tissueMinValue) / (tissueMaxValue - tissueMinValue);
            
            // Default color
            vec4 voxelColor = vec4(tissueColor, norm*exp(brightness));
            
            // Color yellow if cell activation detected
            if (x > cellMinValue) {
                voxelColor = vec4(cellColor, norm*exp(brightness));
            }
            emitRGBA(voxelColor);
        }
        """
    return inspect.cleandoc(shader_code)


def rgba_shader():
    shader_code = """
    #uicontrol float brightness slider(min=0, max=100, default=1, step=0.1)
    #uicontrol float red_channel slider(min=0, max=1, default=1, step=0.01)
    #uicontrol float green_channel slider(min=0, max=1, default=1, step=0.01)
    #uicontrol float blue_channel slider(min=0, max=1, default=1, step=0.01)

    void main() {
        float r = toNormalized(getDataValue(0)) * red_channel;
        float g = toNormalized(getDataValue(1)) * blue_channel;
        float b = toNormalized(getDataValue(2)) * green_channel;
        vec3 color = vec3(r, g, b);
        color *= brightness;
        vec4 rgba = vec4(color, 1.0);
        emitRGBA(rgba);
    }
    """
    return inspect.cleandoc(shader_code)


def segmentation_shader():
    shader_code = """
    #uicontrol float brightness slider(min=-1, max=1, default=0)
    #uicontrol float contrast slider(min=0, max=2, default=1)
    
    void main() {
    // Get the integer value from the segmentation volume
    uint value = uint(getDataValue());

    // Generate a random color based on the value
    float r = fract(sin(float(value) * 12.9898) * 43758.5453);
    float g = fract(sin(float(value) * 78.233) * 43758.5453);
    float b = fract(sin(float(value) * 39.346) * 43758.5453);

    // Apply brightness and contrast adjustments
    vec3 color = vec3(r, g, b);
    color = (color - 0.5) * contrast + 0.5 + brightness;

    emitRGB(color);
    }
    """
    return inspect.cleandoc(shader_code)


def create_source(url, 
                  xyz_dim_in: list, dim_unit_in: list, 
                  xyz_dim_out: list, dim_unit_out: list, 
                  transform_matrix: list):
    """Generates a source dictionary for Neuroglancer.

    Args:
        url (_type_): URL or image layer 
        xyz_dim (list): Unit dimensions of the image
        dim_unit (str): Unit of the dimensions
        xyz_mat (list): Diagonal scaling elements for the affine matrix for the image

    Returns:
        dict: source dictionary for Neuroglancer
    """
    x_dim_in, y_dim_in, z_dim_in = xyz_dim_in
    x_unit_in, y_unit_in, z_unit_in = dim_unit_in
    x_dim_out, y_dim_out, z_dim_out = xyz_dim_out
    x_unit_out, y_unit_out, z_unit_out = dim_unit_out
    #x_mat, y_mat, z_mat = xyz_mat
    #transform_matrix = [[x_mat, 0, 0, 0],
    #                    [0, y_mat, 0, 0],
    #                    [0, 0, z_mat, 0]]
    # Generate source dictionary
    src = {
        "url": url,
        "transform": {
            "outputDimensions": {
                "x": [x_dim_out, x_unit_out],
                "y": [y_dim_out, y_unit_out],
                "z": [z_dim_out, z_unit_out]
            },
            "inputDimensions": {
                "x": [x_dim_in, x_unit_in],
                "y": [y_dim_in, y_unit_in],
                "z": [z_dim_in, z_unit_in]
            },
            "matrix": transform_matrix,
        },
    }
    return src


def create_image_layer(input_type: str, input_path: str, dimensions, xyz_dim_in, dim_unit_in, xyz_dim_out, dim_unit_out, xyz_mat):
    url = input_path  # Used to generate the layers later. Will be an actual URL if a URL is given. 
                      # Otherwise, it will be a LocalVolume object.

    # Check if input path is a URL
    if input_type == "OME-Zarr (URL)":
        print(f"Loading OME-Zarr image from URL: {url}")
        url = "zarr://" + url
        # Create image layer from URL
        image_layer = neuroglancer.ImageLayer( # ["m", "m", "m"] should be dim_unit or xyz_unit
            source=create_source(url, xyz_dim_in, dim_unit_in, xyz_dim_out, dim_unit_out, xyz_mat),
            opacity=1.0,
            shader=image_shader(),
        )
        
    # Otherwise, load up the image locally
    else:
        # Case: input path is a nii file, so load up nii file
        if input_type == "nii":
            print(f"Loading nii file from {input_path}")
            input_volume = nib.load(input_path).get_fdata()
            print(f"\tImage shape: {input_volume.shape}")
            
        # Case: input path is a directory, so load up tiff files locally
        elif input_type == "tif":
            print(f"Loading tiff files from {input_path}")
            fns = natsorted(glob.glob(input_path + "/*.tif", recursive=True))
            fns = fns[::-1]
            image = readTifSection(str(fns[0]))
            w, h = image.shape
            z = len(fns)
            print(f"\tImage shape: {z} x {w} x {h}")
            input_volume = zarr.open(input_path, shape=(z, w, h), chunks=(1, w, h), dtype='u2')
            Parallel(n_jobs=-2, verbose=13)(delayed(readTifWrapper)(i, input_volume, fn) for i, fn in enumerate(fns))
            input_volume = input_volume[:]  # Converts to NumPy array
            
        # Create image layer from local volume
        image_layer = neuroglancer.LocalVolume(data=input_volume, 
                                               dimensions=dimensions, 
                                               volume_type='image', 
                                               voxel_offset=[0, 0, 0])
        url = image_layer
    return image_layer, url


def create_annotation_template_layer(segmentation_path, dimensions):
    print(f"Loading segmentation annotation from {segmentation_path}")
    segmentation_volume = nib.load(segmentation_path).get_fdata()
    print(f"\tAtlas segmentation shape: {segmentation_volume.shape}")
    segmentation_layer = neuroglancer.LocalVolume(
        data=segmentation_volume,
        dimensions=dimensions,
        volume_type='image',
        voxel_offset=[0, 0, 0],
    )
    return segmentation_layer


def create_points_layer(points_path, annotation_color):
    print(f"Loading input points from {points_path}")
    points = np.loadtxt(points_path, skiprows=2)  # Adjust skiprows if necessary
    points = sorted(points, key=lambda point: point[0], reverse=True)
    annotations = [neuroglancer.PointAnnotation(id=f'point{i}', point=[point[2], point[1], point[0]]) 
                   for i, point in enumerate(points)]
    annotation_layer = neuroglancer.AnnotationLayer(annotations=annotations,
                                                    annotation_color=annotation_color)
    return annotation_layer