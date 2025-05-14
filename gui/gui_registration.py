"""
The framework for cell counting for the U01 workflow GUI.
Andy Thai
andy.thai@uci.edu
"""

# Import standard libraries
import os
import sys
import time
import glob
import logging
from natsort import natsorted
from threading import Thread
import subprocess

# Import third-party libraries
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd

# Import PyQt5 libraries
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QFrame, \
                            QLabel, QFileDialog, QSlider, QTabWidget, QDialog, \
                            QCheckBox, QPushButton, QRadioButton, QButtonGroup, QComboBox, QLineEdit, \
                            QProgressBar, QSpinBox, QDoubleSpinBox, QSizePolicy, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Import custom libraries
import run_tissuecyte_stitching_classic
from run_registration_cellcounting import registration, get_input_cell_locations, \
                                          parsePhysicalPointsFromOutputFile, convertPhysicalPointsToIndex, loadTransformMapping, getMappedIndices, \
                                          countCellsInRegions
from registration.reconstruction import createNiiImages
from cellAnalysis.cell_counting import process_counts
import update_summary_structures

# Global formatting variables
BOLD_STYLE = "font-weight: bold; color: black"
DEFAULT_STYLE = "color: black"
TITLE_SPACING = " " * 12
MAX_INT = 2147483647

class RegistrationTab(QWidget):
    def __init__(self, app):
        super().__init__()
        
        # Set up tab settings and layout.
        self.app = app
        self.layout = QVBoxLayout()
        
        # Declare variables to keep track of file paths and settings
        self.input_file = None
        self.input_path = None
        self.input_points = None
        self.channel = 0
        self.is_nii = False
        self.template = None
        self.template_path = None
        self.output_path = None
        self.num_sections = 0
        self.y_shape = 0  # Resolution of the section in the Y direction
        self.x_shape = 0  # Resolution of the section in the X direction
        
        
        ###############################################################################
        ##                                 FILE IO                                   ##
        ###############################################################################
        
        # Title
        file_io_title = "## Registration File I/O"
        self.file_io_title = QLabel(file_io_title, alignment=Qt.AlignCenter)
        self.file_io_title.setTextFormat(Qt.MarkdownText)
        
        # Button to select folder containing data.
        self.input_folder_button = QPushButton("Select input section directory\n⚠️ NO SECTION DATA LOADED")
        self.input_folder_button.clicked.connect(self._select_input_path)
        self.input_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.input_folder_button.setStyleSheet(BOLD_STYLE)
        input_folder_desc = "Select the folder directory containing stitched section image data. The folder should contain TIFF images. " + \
                            "Alternatively, a .nii.gz file can be selected for registration."
        channel_desc = "\n\nThe channel parameter indicates which color channel (0: red, 1: blue, 2: green, 3: far-red/misc.) " + \
                       "the selected sections are. This parameter only affects how the output files are named.\n"
        input_folder_desc += channel_desc
        self.input_folder_desc = QLabel(input_folder_desc, alignment=Qt.AlignCenter)
        self.input_folder_desc.setWordWrap(True)
        self.input_folder_desc.setTextFormat(Qt.MarkdownText)
        #self.input_folder_desc.setMinimumHeight(150)
        
        # Channel parameter
        channel_title = "Channel" + TITLE_SPACING
        self.channel_title = QLabel(channel_title)
        self.channel_spinbox = QSpinBox(minimum=0, maximum=65535, singleStep=1, value=0, alignment=Qt.AlignCenter)
        self.channel_spinbox.valueChanged.connect(self._update_channel)
        
        # Use nii parameter
        nii_title = "Use nii" + TITLE_SPACING
        self.nii_title = QLabel(nii_title)
        self.nii_checkbox = QCheckBox()
        self.nii_checkbox.stateChanged.connect(self._update_nii)
        
        # Input points button
        self.input_points_button = QPushButton("Select input points file\n⚠️ NO CELL POINTS LOADED")
        self.input_points_button.clicked.connect(self._select_input_points)
        self.input_points_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.input_points_button.setStyleSheet(BOLD_STYLE)
        input_points_desc = "Select the input counted cell points file. This should be a .txt file outputted from the cell counting module."
        self.input_points_desc = QLabel(input_points_desc, alignment=Qt.AlignCenter)
        self.input_points_desc.setWordWrap(True)
        self.input_points_desc.setTextFormat(Qt.MarkdownText)
        
        # Button to select atlas to register to.
        self.template_button = QPushButton("Select the template file\n⚠️ NO TEMPLATE DATA LOADED")
        self.template_button.clicked.connect(self._select_template_path)
        self.template_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.template_button.setStyleSheet(BOLD_STYLE)
        template_desc = "Select the template file containing the target atlas to transform your input data towards. " + \
                        "This should normally be a nii.gz file of the 25-micron Allen CCF 2017 average template."
        self.template_desc = QLabel(template_desc, alignment=Qt.AlignCenter)
        self.template_desc.setWordWrap(True)
        self.template_desc.setTextFormat(Qt.MarkdownText)
        #self.template_desc.setMinimumHeight(150)
        
        # Button to select folder to output registered data.
        self.output_folder_button = QPushButton("Select registration output directory\n⚠️ NO OUTPUT FOLDER SELECTED")
        self.output_folder_button.clicked.connect(self._select_output_path)
        #self.output_folder_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.output_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.output_folder_button.setStyleSheet(BOLD_STYLE)
        output_folder_desc = "Select the folder directory to output the registration output to. This will output a folder containing the " + \
                             "transformed counts, region counts, and transformed input volume.\n"
        self.output_folder_desc = QLabel(output_folder_desc, alignment=Qt.AlignCenter)
        self.output_folder_desc.setWordWrap(True)
        self.output_folder_desc.setTextFormat(Qt.MarkdownText)
        
        # Divider line
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        
        ###############################################################################
        ##                          METADATA AND RUN APP                             ##
        ###############################################################################
        
        # Metadata display
        metadata_info = "⚠️ **Select an input directory or volume to display metadata information.**"
        self.metadata_info = QLabel(metadata_info, alignment=Qt.AlignCenter)
        self.metadata_info.setTextFormat(Qt.MarkdownText)
        
        # Register button
        self.register_button = QPushButton("Register data")
        self.register_button.setMinimumSize(100, 50)  # Adjust the size as needed
        self.register_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.register_button.clicked.connect(self._thread_register)
        self.register_button.setEnabled(False)  # Initially disabled
        
        ################### SETUP UI LAYOUT ###################
        
        # Input folder
        self.io_layout = QVBoxLayout()
        self.io_layout.addWidget(self.file_io_title, alignment=Qt.AlignCenter)
        self.io_button_layout = QHBoxLayout()
        self.io_button_layout.addWidget(self.input_folder_button, stretch=1)
        self.io_button_layout.addWidget(self.channel_spinbox)  # Channel
        self.io_button_layout.addWidget(self.channel_title, alignment=Qt.AlignLeft)
        self.io_button_layout.addWidget(self.nii_checkbox)  # Use nii
        self.io_button_layout.addWidget(self.nii_title, alignment=Qt.AlignLeft)
        self.io_layout.addLayout(self.io_button_layout)
        self.io_layout.addWidget(self.input_folder_desc, alignment=Qt.AlignTop)
        self.io_layout.addWidget(self.input_points_button, stretch=1)
        self.io_layout.addWidget(self.input_points_desc, alignment=Qt.AlignTop)
        self.io_layout.addWidget(self.template_button, stretch=1)
        self.io_layout.addWidget(self.template_desc, alignment=Qt.AlignTop)
        
        # Output folder
        #self.output_layout = QHBoxLayout()
        #self.output_layout.addWidget(self.output_folder_button, 1)
        #self.io_layout.addLayout(self.output_layout)
        self.io_layout.addWidget(self.output_folder_button, stretch=1)
        self.io_layout.addWidget(self.output_folder_desc, alignment=Qt.AlignTop)
        self.io_layout.addWidget(h_line)

        ### Register button ###
        self.run_buttons_layout = QHBoxLayout()
        self.run_buttons_layout.addWidget(self.register_button, stretch=1)
        
        # Layout setup
        self.layout.addLayout(self.io_layout)
        self.layout.addWidget(self.metadata_info, alignment=Qt.AlignTop)
        self.layout.addLayout(self.run_buttons_layout)
        
        
        # End
        self.setLayout(self.layout)
        #self.preview_window = self.PreviewWindow(self)


    def _select_input_path(self):
        """
        Select the path to load the input tiles from and saves the selected folder path internally.
        """
        if self.is_nii:
            file_path = QFileDialog.getOpenFileName(None, "Select input nii file", filter="NIfTI files (*.nii *.nii.gz)")[0]
        else: 
            file_path = QFileDialog.getExistingDirectory(None, "Select folder directory with section data")
        if file_path == '':  # If user cancels out of the dialog, exit.
            return
        
        if self.is_nii:
            self.input_folder_button.setText(f"Select input nii file\n✅ {os.path.normpath(file_path)}")
            self.metadata_info.setText("nii file selected.")
        else:
            # Check if the selected folder contains images.
            section_files = glob.glob(os.path.join(file_path, "*.tif"))
            
            if not section_files:  # If user selects a folder without any images, exit and output an error message.
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setWindowTitle("Error")
                err_msg = "Selected folder does not contain any TIF files. Please check your folder layout and select a folder containing TIF sections."
                error_dialog.setText(err_msg)
                error_dialog.exec_()
                return
            self.input_folder_button.setText(f"Select input section directory\n✅ {os.path.normpath(file_path)}")
            
            # Output data metrics
            self.num_sections = len(section_files)
            sample_tif = run_tissuecyte_stitching_classic.read_image(section_files[0])
            metadata_str = "**Metadata**\n\nNumber of sections: " + str(self.num_sections) + \
                        "\n\nSection resolution: " + str(sample_tif.shape)
            self.metadata_info.setText(metadata_str)
            
            y, x = sample_tif.shape
            self.y_shape = y
            self.x_shape = x
            
        self.input_path = file_path
        self.input_folder_button.setStyleSheet(DEFAULT_STYLE)
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        if self.input_path and self.input_points is not None and self.template_path and self.output_path:
            self.register_button.setEnabled(True)
        
        # Update button information
        print(f'Selected input path: {file_path}')
        
        
    def _update_channel(self):
        """
        Update function that runs when the channel spinbox is updated.
        """
        self.channel = self.channel_spinbox.value()
    
    
    def _update_nii(self):
        """
        Update function that runs when the nii checkbox is updated.
        """
        self.is_nii = self.nii_checkbox.isChecked()
        # Reset the input path if the user switches between nii and section data.
        self.input_path = None
        self.register_button.setEnabled(False)
        if self.is_nii:
            self.input_folder_button.setText("Select input nii file\n⚠️ NO NII FILE LOADED")
        else:
            self.input_folder_button.setText("Select input section directory\n⚠️ NO SECTION DATA LOADED")
        self.input_points_button.setStyleSheet(BOLD_STYLE)
        self.metadata_info.setText("⚠️ **Select an input directory or volume to display metadata information.**")
        
        
    def _select_input_points(self):
        """
        Select the path to load the input points from and saves the selected folder path internally.
        """
        file_path = QFileDialog.getOpenFileName(None, "Select input points file", filter="Text files (*.txt)")[0]
        if file_path == '':
            return
        
        """
        try:
            self.input_points = np.loadtxt(file_path, skiprows=2)
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Error")
            err_msg = "Error " + str(e) + " when reading " + str(file_path) + ". Please check the file and try again."
            error_dialog.setText(err_msg)
            error_dialog.exec_()
            return
        """
        self.input_points = file_path
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        if self.input_path and self.input_points is not None and self.template_path and self.output_path:
            self.register_button.setEnabled(True)  # Enable preview button
        
        # Update button information
        print(f'Selected input points path: {file_path}')
        
        self.input_points_button.setText(f"Select input points file\n✅ {os.path.normpath(file_path)}")
        self.input_points_button.setStyleSheet(DEFAULT_STYLE)
        
        
    def _select_template_path(self):
        """
        Select the path to load the input points from and saves the selected folder path internally.
        """
        file_path = QFileDialog.getOpenFileName(None, "Select template file", filter="NIfTI files (*.nii *.nii.gz)")[0]
        if file_path == '':
            return
        
        try:
            print(file_path)
            self.template = nib.load(file_path)
            self.template_path = file_path
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Error")
            err_msg = "Error " + str(e) + " when reading " + str(file_path) + ". Please check the file and try again."
            error_dialog.setText(err_msg)
            error_dialog.exec_()
            return
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        if self.input_path and self.input_points is not None and self.template_path and self.output_path:
            self.register_button.setEnabled(True)  # Enable preview button
        
        # Update button information
        print(f'Selected template path: {file_path}')
        
        self.template_button.setText(f"Select the template file\n✅ {os.path.normpath(file_path)}")
        self.template_button.setStyleSheet(DEFAULT_STYLE)
        
    
    def _select_output_path(self, check_empty=True):
        """
        Select the path to save the outputs to and saves the selected folder path internally.
        """
        folder_path = QFileDialog.getExistingDirectory(None, "Select empty folder to save the output to")
        
        check_empty = False
        if check_empty and os.listdir(folder_path):  # If user selects a non-empty folder, exit and output an error message.
            print("Selected folder is not empty. Please select an empty folder to export the cell counting output to.")
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Error")
            error_dialog.setText("Selected folder is not empty. Please select an empty folder to export to.")
            error_dialog.exec_()
            return
        if folder_path == '':  # If user cancels out of the dialog, exit.
            return
        
        # Save the selected folder path internally.
        self.output_path = folder_path
        
        # If a valid path is given, save the filepath internally.
        if self.input_path and self.input_points is not None and self.template_path and self.output_path:
            self.register_button.setEnabled(True)
            
        # Update button information
        print(f'Selected output folderpath: {folder_path}')
        
        self.output_folder_button.setText(f"Select registration output directory\n✅ {os.path.normpath(self.output_path)}")
        self.output_folder_button.setStyleSheet(DEFAULT_STYLE)
        
        
    def _popup_preview(self):
        """
        Enable the preview window with the selected parameters on a selected section.
        """
        # Update parameters from main window here.
        self.preview_window._update()
        self.preview_window.exec_()
        
        
    def _toggle_buttons(self, toggle: bool):
        """
        Disable all buttons to prevent user input during processing.
        """
        # File IO
        self.input_folder_button.setEnabled(toggle)
        self.input_points_button.setEnabled(toggle)
        self.template_button.setEnabled(toggle)
        self.output_folder_button.setEnabled(toggle)
        self.channel_spinbox.setEnabled(toggle)
        self.nii_checkbox.setEnabled(toggle)
        self.register_button.setEnabled(toggle)
        
        
    def _thread_register(self):
        """
        Thread the registration function.
        """
        t1 = Thread(target=self._register) 
        t1.start()
        
        
    def _register(self):
        """
        Register the moving image to the fixed template image and morph the points.
        """
        """
        This script can generate nii files and perform registration as well as use existing nii files for registration. 
        
        If --img_dir is provided, the script creates nii files according to the channel specified and use them for registration.

        If --img_dir is not provided, then script takes additional arguments --fixed_image, --moving_image. 

        In addition output_dir can be specified to store all intermediary and final results. 

        For performing cell detection and cell counting, additional arguments '--cell_detection' needs to be given. For cell detection, '--img_dir' is a required argument. 
        Threshold can be adjusted for cell counting with argument '--threshold'

        --input_mat is only present for legacy reasons and can be ignored. 
        """
        self._toggle_buttons(False)
        elastix_dir = "./elastix/"
        annotationImagePath = r"./CCF_DATA/annotation_25.nii.gz"
        ATLAS_PATH = r"./CCF_DATA/1_adult_mouse_brain_graph_mapping.csv"
        SUMMARY_STRUCTURES_PATH = r"./CCF_DATA/300_summary_structures.csv"
        REGISTERED_OUTPUT_PATH = self.output_path + "//registered_output//"
        scaledCellLocations = np.loadtxt(self.input_points, skiprows=2)
        input_path = self.input_path

        # Start loading in the files
        t2d = False  # When template being registered to imaged brain volume and the resulting registration used for cell counting
        start = time.time()

        # If section directory is chosen, generate .nii images and load the 25 micron one.
        if not self.is_nii:
            print("Creating nii.gz images at " + self.output_path)
            print("This may take some time...")
            createNiiImages(input_path, self.output_path, self.channel)
            print("nii.gz images created in {} seconds.".format(time.time()- start))
            print("Loading brain_25.nii.gz image...")
            input_path = self.output_path + "//brain_25.nii.gz"
            mImage = nib.load(input_path)
        # Otherwise load existing nii image.
        else:
            print("Skipping nii image creation. nii file provided at " + input_path + ".")
            mImage = nib.load(input_path)
        mData = mImage.get_fdata()
        
        #output_points_file = os.path.join(output_dir, "output_points.txt")
        cell_count_file = os.path.join(REGISTERED_OUTPUT_PATH, "cell_count.csv")
        
        if t2d:
            registration_cmd = [elastix_dir + "elastix", "-m", self.template_path, "-f", input_path, 
                                "-out", REGISTERED_OUTPUT_PATH, "-p" , "001_parameters_Rigid.txt", "-p", "002_parameters_BSpline.txt"]
            transformix_cmd  = [elastix_dir + "transformix", "-def", self.input_points, 
                                "-out", REGISTERED_OUTPUT_PATH, "-tp", os.path.join(REGISTERED_OUTPUT_PATH, "TransformParameters.1.txt")]
            #transformix_cmd2 = [elastix_dir+"transformix","-in",annotationImagePath,"-out", output_dir,"-tp",os.path.join(output_dir,"TransformParameters.1.txt")]

        fImage = nib.load(self.template_path)
        fData = fImage.get_fdata()

        if not os.path.isdir(REGISTERED_OUTPUT_PATH):
            os.mkdir(REGISTERED_OUTPUT_PATH)
            if t2d:
                subprocess.run(registration_cmd)
            else:
                registration(self.template_path, input_path, REGISTERED_OUTPUT_PATH)

            print("Registration done in {}".format(time.time()- start))
        else:
            print("Skipping registration. Registration directory at " + REGISTERED_OUTPUT_PATH + " already present.")


        """
        Two methods for finding the region where the cell is located. Both of these methods give same results
        Method1 : Transform cell locations in data space to the brain image space.
        Method2 : Transform the annotation map to the brain image space.
        """
        annotationImage  = sitk.ReadImage(annotationImagePath)

        # The below transformation needed only in case of registering template to data
        if t2d:
            subprocess.run(transformix_cmd)   #transforms points to annotation image space
            scaledCellLocations = parsePhysicalPointsFromOutputFile(os.path.join(REGISTERED_OUTPUT_PATH, "outputpoints.txt"))
            outputIndices    = convertPhysicalPointsToIndex(scaledCellLocations , annotationImage)
        else:
            mapping = loadTransformMapping(fData.shape, mData.shape, REGISTERED_OUTPUT_PATH)
            np.save("mapping.npy", mapping)
            mapping = np.load("mapping.npy")
            outputIndices = getMappedIndices(scaledCellLocations, mapping)
            try:
                outputIndices = outputIndices[~np.all(outputIndices == [0, 0, 0], axis=1)]
            except:
                outputIndices = np.array(outputIndices)
            np.savetxt(os.path.join(REGISTERED_OUTPUT_PATH, "outputIndices.txt"), outputIndices , "%d %d %d", 
                       header = "index\n" + str(outputIndices.shape[0]), 
                       comments = "")

        print("Cell location transformation done in {}".format(time.time()- start))

        #np.savetxt(output_points_file, scaledCellLocations , "%d %d %d", header = "index\n"+str(scaledCellLocations.shape[0]), comments ="")
        cellRegionCounts, pointIndices = countCellsInRegions(outputIndices, annotationImage)
        pd.DataFrame(dict(cellRegionCounts).items(), columns=["region", "count"]).to_csv(cell_count_file, index=False)
        atlas_df = pd.read_csv(ATLAS_PATH, index_col=None)
        count_df = pd.read_csv(cell_count_file, index_col=None)
        region_df,count_df = process_counts(atlas_df, count_df)
        count_df.to_csv(os.path.join(REGISTERED_OUTPUT_PATH, "cell_region_count.csv"), index=False)
        region_df.to_csv(os.path.join(REGISTERED_OUTPUT_PATH, "region_counts.csv"), index=False)
        
        # Create a summary structures file for this count
        summary_df = pd.read_csv(SUMMARY_STRUCTURES_PATH, index_col=None)
        summary_df = update_summary_structures.update_summary_structures(count_df, atlas_df, summary_df)
        summary_df.to_csv(os.path.join(REGISTERED_OUTPUT_PATH, "summary_structures_counts.csv"), index=False)
        
        self._toggle_buttons(True)


if __name__ == "__main__":
    class U01App(QMainWindow):
        def __init__(self):
            super().__init__()

            # Setup initial window settings
            WINDOW_HEIGHT = 800
            WINDOW_WIDTH = 200
            
            # Setup window information
            self.setWindowTitle("U01 Workflow GUI")
            if os.path.exists('icon.png'):
                window_icon = QIcon('icon.png')
            else:
                window_icon = QIcon('../icon.png')
            self.setWindowIcon(window_icon)
            self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
            
            # Setup layout and tabs
            self.tabs = QTabWidget()
            self.tab_registration = RegistrationTab(self)
            self.tabs.addTab(self.tab_registration, "Registration")
            self.setCentralWidget(self.tabs)
        
    app = QApplication(sys.argv)
    window = U01App()
    window.show()
    sys.exit(app.exec())