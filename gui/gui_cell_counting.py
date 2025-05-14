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

# Import third-party libraries
import cv2
import numpy as np
import joblib

# Import PyQt5 libraries
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QFrame, \
                            QLabel, QFileDialog, QSlider, QTabWidget, QDialog, \
                            QCheckBox, QPushButton, QRadioButton, QButtonGroup, QComboBox, QLineEdit, \
                            QProgressBar, QSpinBox, QDoubleSpinBox, QSizePolicy, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Import matplotlib libraries
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Import custom libraries
import run_tissuecyte_stitching_classic
from cellAnalysis.cell_detection import createShardedPointAnnotation, readSectionTif
import gui.gui_shared as gui_shared

# Global formatting variables
BOLD_STYLE = "font-weight: bold; color: black"
DEFAULT_STYLE = "color: black"
TITLE_SPACING = " " * 12
MAX_INT = 2147483647

class CountingTab(QWidget):
    def __init__(self, app):
        super().__init__()
        
        # Set up tab settings and layout.
        self.app = app
        self.layout = QVBoxLayout()
        
        # Declare variables to keep track of file paths and settings
        self.input_path = None
        self.output_path = None
        self.num_sections = 0
        self.preview_window = None
        self.y_shape = 0  # Resolution of the section in the Y direction
        self.x_shape = 0  # Resolution of the section in the X direction
        
        
        ###############################################################################
        ##                                 FILE IO                                   ##
        ###############################################################################
        
        # Title
        file_io_title = "## Cell Counting File I/O"
        self.file_io_title = QLabel(file_io_title, alignment=Qt.AlignCenter)
        self.file_io_title.setTextFormat(Qt.MarkdownText)
        
        # Button to select folder containing tile data.
        self.input_folder_button = QPushButton("Select input section directory\n⚠️ NO SECTION DATA LOADED")
        self.input_folder_button.clicked.connect(self._select_input_path)
        self.input_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.input_folder_button.setStyleSheet(BOLD_STYLE)
        input_folder_desc = "Select the folder directory containing stitched section image data. The folder should contain TIFF images."
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
        
        # Button to select folder to output stitched data.
        self.output_folder_button = QPushButton("Select cell counting output directory\n⚠️ NO OUTPUT FOLDER SELECTED")
        self.output_folder_button.clicked.connect(self._select_output_path)
        self.output_folder_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.output_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.output_folder_button.setStyleSheet(BOLD_STYLE)
        output_folder_desc = "Select the folder directory to output the cell counting output to. This will output a folder containing the " + \
                             "counts and locations of the detected cells throughout the sections.\n"
        self.output_folder_desc = QLabel(output_folder_desc, alignment=Qt.AlignCenter)
        self.output_folder_desc.setWordWrap(True)
        
        # Divider line
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        
        ###############################################################################
        ##                               PARAMETERS                                  ##
        ###############################################################################
        
        # Title
        parameter_title = "## Counting Parameters"
        self.parameter_title = QLabel(parameter_title, alignment=Qt.AlignCenter)
        self.parameter_title.setTextFormat(Qt.MarkdownText)
        
        ### COUNTING PARAMS ###
        
        # Counting processing parameters
        cell_intensity_thresh_title = "Cell intensity threshold" + TITLE_SPACING
        self.cell_intensity_thresh_title = QLabel(cell_intensity_thresh_title)
        self.cell_intensity_thresh_spinbox = QSpinBox(minimum=0, maximum=255, singleStep=1, value=10, alignment=Qt.AlignCenter)
        cell_intensity_thresh_desc = "**Cell intensity threshold** helps " + \
                         "filter between tissue and brighter highlighted cells. " + \
                         "Lower values will be more sensitive to intensity variations, while higher values " + \
                         "will only grab more brighter cell regions."
        self.cell_intensity_thresh_desc = QLabel(cell_intensity_thresh_desc, alignment=Qt.AlignCenter)
        self.cell_intensity_thresh_desc.setWordWrap(True)
        self.cell_intensity_thresh_desc.setTextFormat(Qt.MarkdownText)
        
        # Min cell size
        min_cell_size_title = "Min. cell size" + TITLE_SPACING  # Min. cell size
        self.min_cell_size_title = QLabel(min_cell_size_title)
        self.min_cell_size_spinbox = QSpinBox(minimum=0, maximum=65535, singleStep=1, value=25, alignment=Qt.AlignCenter)
        self.min_cell_size_spinbox.setSuffix(" px")
        min_cell_size_desc = "**Minimum cell size** only counts regions above the size as cells, excluding regions smaller " + \
                             "than the size threshold. This prevents tiny specks from being accidentally counted as cells."
        self.min_cell_size_desc = QLabel(min_cell_size_desc, alignment=Qt.AlignCenter)
        self.min_cell_size_desc.setWordWrap(True)
        self.min_cell_size_desc.setTextFormat(Qt.MarkdownText)
        
        # Bcakground threshold
        bg_thresh_title = "Background threshold" + TITLE_SPACING
        self.bg_thresh_title = QLabel(bg_thresh_title)
        self.bg_thresh_spinbox = QSpinBox(minimum=0, maximum=255, singleStep=1, value=15, alignment=Qt.AlignCenter)
        #self.bg_thresh_spinbox.setSuffix(" (0-255)")
        bg_thresh_desc = "**Background threshold** helps " + \
                         "determines which pixels are considered as brain tissue and affects the quality of background removal. " + \
                         "Regions that are considered background will not be counted as cells. This parameter " + \
                         "ensures that only areas within the brain tissue will be detected."
        self.bg_thresh_desc = QLabel(bg_thresh_desc, alignment=Qt.AlignCenter)
        self.bg_thresh_desc.setWordWrap(True)
        self.bg_thresh_desc.setTextFormat(Qt.MarkdownText)
        
        # Adv Title
        adv_parameters_title = "### Optional Parameters"
        self.adv_parameters_title = QLabel(adv_parameters_title, alignment=Qt.AlignCenter)
        self.adv_parameters_title.setTextFormat(Qt.MarkdownText)
                
        # Adv parameters
        adv_param_desc = "These parameters affect the background removal and are optional and automatically set, but can be adjusted if needed."
        self.adv_param_desc = QLabel(adv_param_desc, alignment=Qt.AlignCenter)
        self.adv_param_desc.setWordWrap(True)
        self.adv_param_desc.setTextFormat(Qt.MarkdownText)
        # Min size
        processing_min_size_title = "Min. speck size" + TITLE_SPACING
        self.processing_min_size_title = QLabel(processing_min_size_title)
        self.processing_min_size_spinbox = QSpinBox(minimum=0, maximum=MAX_INT, singleStep=1, value=0, alignment=Qt.AlignCenter)
        self.processing_min_size_spinbox.setSuffix(" px")
        self.processing_min_size_spinbox.setEnabled(False)
        processing_min_size_desc = "**Minimum size** helps remove stray small pixels during the background removal process. " + \
                                   "Connected pixel areas below this size will be considered background."
        self.processing_min_size_desc = QLabel(processing_min_size_desc, alignment=Qt.AlignCenter)
        self.processing_min_size_desc.setWordWrap(True)
        self.processing_min_size_desc.setTextFormat(Qt.MarkdownText)
        # Area thresh
        processing_area_thresh_title = "Max. gap size" + TITLE_SPACING  # Area threshold
        self.processing_area_thresh_title = QLabel(processing_area_thresh_title)
        self.processing_area_thresh_spinbox = QSpinBox(minimum=0, maximum=MAX_INT, singleStep=1, value=0, alignment=Qt.AlignCenter)
        self.processing_area_thresh_spinbox.setSuffix(" px")
        self.processing_area_thresh_spinbox.setEnabled(False)
        processing_area_thresh_desc = "**Maximum gap size** is the area threshold used for gaps inside the brain tissue regions. " + \
                             "Holes smaller than this size will be filled in and considered part of the brain tissue."
        self.processing_area_thresh_desc = QLabel(processing_area_thresh_desc, alignment=Qt.AlignCenter)
        self.processing_area_thresh_desc.setWordWrap(True)
        self.processing_area_thresh_desc.setTextFormat(Qt.MarkdownText)
        # Dilation iter parameter
        processing_dilation_iter_title = "Dilation iter." + TITLE_SPACING
        self.processing_dilation_iter_title = QLabel(processing_dilation_iter_title)
        self.processing_dilation_iter_spinbox = QSpinBox(minimum=0, maximum=MAX_INT, singleStep=1, value=0, alignment=Qt.AlignCenter)
        self.processing_dilation_iter_spinbox.setEnabled(False)
        processing_dilation_iter_desc = "The **dilation iterations** parameter affects how many times to expand the edge of the final " + \
                                        "background mask. More iterations will expand the edges of the mask used to delineate tissue and background."
        self.processing_dilation_iter_desc = QLabel(processing_dilation_iter_desc, alignment=Qt.AlignCenter)
        self.processing_dilation_iter_desc.setWordWrap(True)
        self.processing_dilation_iter_desc.setTextFormat(Qt.MarkdownText)
        # Min distance parameter
        processing_min_dist_title = "Min. cell distance" + TITLE_SPACING
        self.processing_min_dist_title = QLabel(processing_min_dist_title)
        self.processing_min_dist_spinbox = QSpinBox(minimum=0, maximum=MAX_INT, singleStep=1, value=5, alignment=Qt.AlignCenter)
        self.processing_min_dist_spinbox.setEnabled(False)
        processing_min_dist_desc = "The **minimum cell distance** determines the minimum distance allowed between two cell centers for them " + \
                                   "to be considered separate cells. This parameter helps prevent double-counting of cells that are too close together." + \
                                    "\n\nYou may preview how different parameters affect image processing using the 'Preview' button."
        self.processing_min_dist_desc = QLabel(processing_min_dist_desc, alignment=Qt.AlignCenter)
        self.processing_min_dist_desc.setWordWrap(True)
        self.processing_min_dist_desc.setTextFormat(Qt.MarkdownText)
        
        # Preview button
        self.preview_button = QPushButton("Preview")
        self.preview_button.setMinimumSize(100, 50)  # Adjust the size as needed
        self.preview_button.clicked.connect(self._popup_preview)
        self.preview_button.setEnabled(False)  # Initially disabled
        
        h_line2 = QFrame()
        h_line2.setFrameShape(QFrame.HLine)
        h_line2.setFrameShadow(QFrame.Sunken)
        
        ###############################################################################
        ##                          METADATA AND RUN APP                             ##
        ###############################################################################
        
        # Metadata display
        metadata_info = "⚠️ **Select an input directory to display metadata information.**"
        self.metadata_info = QLabel(metadata_info, alignment=Qt.AlignCenter)
        self.metadata_info.setTextFormat(Qt.MarkdownText)
        
        # Count button
        self.count_button = QPushButton("Count cells")
        self.count_button.setMinimumSize(100, 50)  # Adjust the size as needed
        self.count_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.count_button.clicked.connect(self._thread_count_cells)
        self.count_button.setEnabled(False)  # Initially disabled
        
        ################### SETUP UI LAYOUT ###################
        
        # Input folder
        self.io_layout = QVBoxLayout()
        self.io_layout.addWidget(self.file_io_title, alignment=Qt.AlignCenter)
        self.io_button_layout = QHBoxLayout()
        self.io_button_layout.addWidget(self.input_folder_button, stretch=1)
        self.io_button_layout.addWidget(self.channel_spinbox)  # Channel
        self.io_button_layout.addWidget(self.channel_title, alignment=Qt.AlignLeft)
        self.io_layout.addLayout(self.io_button_layout)
        self.io_layout.addWidget(self.input_folder_desc, alignment=Qt.AlignTop)
        
        # Output folder
        self.output_layout = QHBoxLayout()
        self.output_layout.addWidget(self.output_folder_button, 1)
        self.io_layout.addLayout(self.output_layout)
        self.io_layout.addWidget(self.output_folder_desc, alignment=Qt.AlignTop)
        self.io_layout.addWidget(h_line)

        ### Parameter row - image processing and corrections ###
        self.prow_layout = QHBoxLayout()
        self.prow_layout.addStretch(1)
        self.prow_layout.addWidget(self.cell_intensity_thresh_spinbox)  # Cell intensity threshold
        self.prow_layout.addWidget(self.cell_intensity_thresh_title, alignment=Qt.AlignLeft)
        self.prow_layout.addWidget(self.min_cell_size_spinbox)  # Min cell size
        self.prow_layout.addWidget(self.min_cell_size_title, alignment=Qt.AlignLeft)
        self.prow_layout.addWidget(self.bg_thresh_spinbox)  # BG thresh
        self.prow_layout.addWidget(self.bg_thresh_title, alignment=Qt.AlignLeft)
        self.prow_layout.addStretch(1)
        
        ### Parameter row - advanced parameters ###
        self.adv_prow_layout = QHBoxLayout()
        self.adv_prow_layout.addStretch(1)
        self.adv_prow_layout.addWidget(self.processing_min_size_spinbox)
        self.adv_prow_layout.addWidget(self.processing_min_size_title, alignment=Qt.AlignLeft)
        self.adv_prow_layout.addWidget(self.processing_area_thresh_spinbox)
        self.adv_prow_layout.addWidget(self.processing_area_thresh_title, alignment=Qt.AlignLeft)
        self.adv_prow_layout.addWidget(self.processing_dilation_iter_spinbox)
        self.adv_prow_layout.addWidget(self.processing_dilation_iter_title, alignment=Qt.AlignLeft)
        self.adv_prow_layout.addWidget(self.processing_min_dist_spinbox)
        self.adv_prow_layout.addWidget(self.processing_min_dist_title, alignment=Qt.AlignLeft)
        self.adv_prow_layout.addStretch(1)
        
        # Add to main layout
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.addWidget(self.parameter_title)
        self.parameter_layout.addLayout(self.prow_layout)
        self.parameter_layout.addWidget(self.cell_intensity_thresh_desc, alignment=Qt.AlignTop)
        self.parameter_layout.addWidget(self.min_cell_size_desc, alignment=Qt.AlignTop)
        self.parameter_layout.addWidget(self.bg_thresh_desc, alignment=Qt.AlignTop)
        # Advanced parameters
        self.adv_parameter_layout = QVBoxLayout()
        self.adv_parameter_layout.addWidget(self.adv_parameters_title)
        self.adv_parameter_layout.addLayout(self.adv_prow_layout)
        self.adv_parameter_layout.addWidget(self.adv_param_desc, alignment=Qt.AlignTop)
        self.adv_parameter_layout.addWidget(self.processing_min_size_desc, alignment=Qt.AlignTop)
        self.adv_parameter_layout.addWidget(self.processing_area_thresh_desc, alignment=Qt.AlignTop)
        self.adv_parameter_layout.addWidget(self.processing_dilation_iter_desc, alignment=Qt.AlignTop)
        self.adv_parameter_layout.addWidget(self.processing_min_dist_desc, alignment=Qt.AlignTop)
        
        # Preview button
        self.adv_parameter_layout.addWidget(self.preview_button)
        self.adv_parameter_layout.addWidget(h_line2)
        
        ### Count button ###
        self.run_buttons_layout = QHBoxLayout()
        self.run_buttons_layout.addWidget(self.count_button, stretch=1)
        
        # Layout setup
        self.layout.addLayout(self.io_layout)
        self.layout.addLayout(self.parameter_layout)
        self.layout.addLayout(self.adv_parameter_layout)
        self.layout.addWidget(self.metadata_info, alignment=Qt.AlignTop)
        self.layout.addLayout(self.run_buttons_layout)
        
        
        # End
        self.setLayout(self.layout)
        #self.preview_window = self.PreviewWindow(self)


    def _select_input_path(self):
        """
        Select the path to load the input tiles from and saves the selected folder path internally.
        """
        folder_path = QFileDialog.getExistingDirectory(None, "Select folder directory with section data")
        if folder_path == '':  # If user cancels out of the dialog, exit.
            return
        
        # Check if the selected folder contains images.
        section_files = glob.glob(os.path.join(folder_path, "*.tif"))
        
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        if not section_files:  # If user selects a folder without any images, exit and output an error message.
            err_msg = "Selected folder does not contain any TIF files. Please check your folder layout and select a folder containing TIF sections."
            error_dialog.setText(err_msg)
            error_dialog.exec_()
            return

        self.input_path = folder_path
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        self.processing_min_size_spinbox.setEnabled(True)
        self.processing_area_thresh_spinbox.setEnabled(True)
        self.processing_dilation_iter_spinbox.setEnabled(True)
        self.processing_min_dist_spinbox.setEnabled(True)
        
        self.preview_button.setEnabled(True)  # Enable preview button
        if self.input_path and self.output_path:  # Enable stitch button if all required paths are provided.
            self.count_button.setEnabled(True)
        
        # Update button information
        print(f'Selected input folderpath: {folder_path}')
        
        self.input_folder_button.setText(f"Select input section directory\n✅ {os.path.normpath(self.input_path)}")
        self.input_folder_button.setStyleSheet(DEFAULT_STYLE)
        
        # Output data metrics
        self.num_sections = len(section_files)
        sample_tif = run_tissuecyte_stitching_classic.read_image(section_files[0])
        metadata_str = "**Metadata**\n\nNumber of sections: " + str(self.num_sections) + \
                       "\n\nSection resolution: " + str(sample_tif.shape)
        self.metadata_info.setText(metadata_str)
        
        y, x = sample_tif.shape
        self.y_shape = y
        self.x_shape = x
        min_size = max(64, int(max(y, x) * 0.20))      # Largest allowable size for specks
        area_thresh = max(64, int(max(y, x) * 20))     # Fill in holes smaller than this size
        dilation_iter = int(max(y, x) / 500)           # 20
        self.processing_min_size_spinbox.setValue(min_size)
        self.processing_area_thresh_spinbox.setValue(area_thresh)
        self.processing_dilation_iter_spinbox.setValue(dilation_iter)
                
        # Setup stitching preview window values
        self.preview_window = self.PreviewWindow(self)
        
    
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
        if self.input_path and self.output_path:
            self.count_button.setEnabled(True)
            
        # Update button information
        print(f'Selected output folderpath: {folder_path}')
        
        self.output_folder_button.setText(f"Select cell counting output directory\n✅ {os.path.normpath(self.output_path)}")
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
        self.output_folder_button.setEnabled(toggle)
        self.channel_spinbox.setEnabled(toggle)
        # Counting parameters
        self.cell_intensity_thresh_spinbox.setEnabled(toggle)
        self.min_cell_size_spinbox.setEnabled(toggle)
        self.bg_thresh_spinbox.setEnabled(toggle)
        # Optional parameters
        self.processing_min_size_spinbox.setEnabled(toggle)
        self.processing_area_thresh_spinbox.setEnabled(toggle)
        self.processing_dilation_iter_spinbox.setEnabled(toggle)
        self.processing_min_dist_spinbox.setEnabled(toggle)
        # Buttons
        self.preview_button.setEnabled(toggle)
        self.count_button.setEnabled(toggle)
        
        
    def _thread_count_cells(self):
        """
        Thread the stitching function.
        """
        t1 = Thread(target=self._count_cells) 
        t1.start()
        
        
    def _count_cells(self):
        """
        Count the cells in the selected sections with the selected parameters.
        """
        ##########################################################################################################
        self._toggle_buttons(False)
        # Settings
        img_dir = self.input_path
        out_dir = self.output_path
        input_points_file = out_dir + "/inputpoints.txt"
        input_points_file_scaled = out_dir + "/inputpoints_scaled.txt"
        
        intensity_threshold = self.cell_intensity_thresh_spinbox.value()  # Intensity threshold for cell detection
        size_threshold = self.min_cell_size_spinbox.value()  # Minimum cell size for cell detection (try 25 as default)
        channel = self.channel_spinbox.value()  # Channel to detect cells from

        # Background removal
        bg_thresh = self.bg_thresh_spinbox.value()
        y = self.y_shape
        x = self.x_shape
        min_size = self.processing_min_size_spinbox.value()  # Largest allowable size for specks
        area_thresh = self.processing_area_thresh_spinbox.value()  # Fill in holes smaller than this size
        dilation_iter = self.processing_dilation_iter_spinbox.value()  # 20

        # Temporary image path solution for kiwi server
        if sys.platform == 'linux':
            imgfiles = natsorted(glob.glob(img_dir + "/*-{}.tif".format(channel), recursive=True))
        else:
            imgfiles = natsorted(glob.glob(img_dir + "/**/*1_{}.tif".format(channel), recursive=True))

        # Detect cells for all sections
        start = time.time()
        cells = []
        for i, img_file in enumerate(imgfiles):
            print("\n" + str(i), img_file)
            image = readSectionTif(img_file)
            image[image < 0] = 0
            c = gui_shared.get_cell_locations(img_file, index = i, 
                                              intensity_threshold = intensity_threshold, size_threshold = size_threshold,
                                              bg_threshold = bg_thresh, min_size = min_size, area_threshold = area_thresh, dilation_iter = dilation_iter)
            if c.size:
                cells.append(c)
            #break
        end = time.time()

        # Adjust indices
        cell_locations = np.vstack(cells)
        cell_locations[:, 0] = len(imgfiles) - cell_locations[:,0] - 1
        #cell_locations[:,0] = mData.shape[0] - cell_locations[:,0]-1


        # points = cell_locations
        createShardedPointAnnotation(cell_locations, out_dir) # Create sharded points for Neuroglancer layer
        # Scale cell locations to smaller registered version
        scaledCellLocations = np.round(cell_locations * [1, 1/20, 1/20]).astype(int)
        np.savetxt(input_points_file, cell_locations , "%d %d %d", 
                header = "index\n" + str(cell_locations.shape[0]), 
                comments = "")
        np.savetxt(input_points_file_scaled, scaledCellLocations , "%d %d %d", 
                header = "index\n" + str(cell_locations.shape[0]), 
                comments = "")
        print("Cell detection done in {} seconds.".format(end - start))
        print("Average processing time: {} seconds per section".format((end - start) / len(imgfiles)))
        self._toggle_buttons(True)
    

    class PreviewWindow(QDialog):
        def __init__(self, parent):
            super().__init__()
            POPUP_HEIGHT = 800
            POPUP_WIDTH = 600
            self.setWindowTitle("Cell Counting Preview")
            self.resize(POPUP_WIDTH, POPUP_HEIGHT)
            
            # Setup layout
            self.layout = QVBoxLayout()
            
            # Setup preview window settings and variables
            self.parent = parent  # Parent app
            self.idx = 0  # Section index to preview
            self.last_idx = -1  # Last section index previewed
            self.original_section = None  # Original previewed section
            self.current_section = None   # Current previewed section (with parameters applied)
            self.cells = None
            self.points = None
            self.show_cells = True
            
            # Setup matplotlib figure and canvas
            self.histogram_figure, self.histogram_ax = plt.subplots()
            self.histogram_canvas = FigureCanvas(self.histogram_figure)
            self.figure, self.ax = plt.subplots()
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            # Parameters
            parameters_title = "## Preview Parameters"
            self.parameters_title = QLabel(parameters_title, alignment=Qt.AlignCenter)
            self.parameters_title.setTextFormat(Qt.MarkdownText)
            
            # Cell intensity threshold
            cell_intensity_thresh_title = "Cell intensity threshold" + TITLE_SPACING  # Cell intensity threshold
            self.cell_intensity_thresh_title = QLabel(cell_intensity_thresh_title)
            self.cell_intensity_thresh_spinbox = QSpinBox(minimum=0, maximum=255, singleStep=1, value=10, alignment=Qt.AlignCenter)
            # Min cell size parameter
            min_cell_size_title = "Min. cell size" + TITLE_SPACING
            self.min_cell_size_title = QLabel(min_cell_size_title)
            self.min_cell_size_spinbox = QSpinBox(minimum=0, maximum=65535, singleStep=1, value=25, alignment=Qt.AlignCenter)
            self.min_cell_size_spinbox.setSuffix(" px")
            # Background threshold
            bg_thresh_title = "Background threshold" + TITLE_SPACING
            self.bg_thresh_title = QLabel(bg_thresh_title)
            self.bg_thresh_spinbox = QSpinBox(minimum=0, maximum=255, singleStep=1, value=15, alignment=Qt.AlignCenter)
            
            ### OPTIONAL PARAMETERS ###
            # Min speck size
            processing_min_size_title = "Min. speck size" + TITLE_SPACING  # Min speck size
            self.processing_min_size_title = QLabel(processing_min_size_title)
            self.processing_min_size_spinbox = QSpinBox(minimum=0, maximum=MAX_INT, singleStep=1, value=0, alignment=Qt.AlignCenter)
            self.processing_min_size_spinbox.setSuffix(" px")
            # Max gap size parameter
            processing_area_thresh_title = "Max. gap size" + TITLE_SPACING
            self.processing_area_thresh_title = QLabel(processing_area_thresh_title)
            self.processing_area_thresh = QSpinBox(minimum=0, maximum=MAX_INT, singleStep=1, value=0, alignment=Qt.AlignCenter)
            self.processing_area_thresh.setSuffix(" px")
            # Dilation iter
            processing_dilation_iter_title = "Dilation iter." + TITLE_SPACING
            self.processing_dilation_iter_title = QLabel(processing_dilation_iter_title)
            self.processing_dilation_iter = QSpinBox(minimum=0, maximum=MAX_INT, singleStep=1, value=0, alignment=Qt.AlignCenter)
            # Min distance parameter
            processing_min_dist_title = "Min. cell distance" + TITLE_SPACING
            self.processing_min_dist_title = QLabel(processing_min_dist_title)
            self.processing_min_dist_spinbox = QSpinBox(minimum=0, maximum=MAX_INT, singleStep=1, value=5, alignment=Qt.AlignCenter)    
                    
            self._update()  # Update slider and spinbox values with parent values             
            
            
            # Divider line
            h_line = QFrame()
            h_line.setFrameShape(QFrame.HLine)
            h_line.setFrameShadow(QFrame.Sunken)
            
            # Section number slider text
            section_slider_title = "Section index"
            self.section_slider_title = QLabel(section_slider_title, alignment=Qt.AlignCenter)
            self.current_idx_spinbox = QSpinBox(minimum=1, maximum=self.parent.num_sections, 
                                                singleStep=1, value=1, alignment=Qt.AlignCenter)
            self.current_idx_spinbox.valueChanged.connect(self._update_idx_from_spinbox)
            section_idx_label = "/ " + str(self.parent.num_sections)
            self.section_idx_label = QLabel(section_idx_label, alignment=Qt.AlignCenter)
            self.section_slider = QSlider(Qt.Horizontal)
            self.section_slider.setRange(1, self.parent.num_sections)
            self.section_slider.setValue(1)
            self.section_slider.valueChanged.connect(self._update_idx_from_slider)
            self.section_slider.setTickPosition(QSlider.TicksBelow)
            self.section_slider.setTickInterval(10)
            
            # Generate button
            self.export_button = QPushButton("Save parameters")
            self.export_button.clicked.connect(self._export_parameters)
            self.export_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.export_button.setMinimumSize(200, 25)  # Adjust the size as needed
            self.generate_button = QPushButton("Generate preview")
            self.generate_button.clicked.connect(self._thread_preview)
            self.generate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.generate_button.setMinimumSize(200, 25)  # Adjust the size as needed
            
            # Contrast sliders
            self.contrast_layout = QHBoxLayout()
            
            self.alpha_spinbox = QDoubleSpinBox(minimum=0.00, maximum=9999.00, singleStep=0.01, value=0.50)
            alpha_title = "Alpha" + TITLE_SPACING
            self.alpha_title = QLabel(alpha_title)
            self.beta_spinbox = QDoubleSpinBox(minimum=-9999.0, maximum=9999.00, singleStep=0.01, value=0.00)
            beta_title = "Beta" + TITLE_SPACING
            self.beta_title = QLabel(beta_title)
            #self.beta_spinbox.setEnabled(False)
            
            self.contrast_button = QPushButton("Apply contrast")
            self.contrast_button.clicked.connect(self._redraw)
            self.contrast_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.contrast_button.setMinimumSize(200, 25)  # Adjust the size as needed
            self.contrast_button.setEnabled(False)
            
            self.toggle_cells_button = QPushButton("Toggle cells")
            self.toggle_cells_button.clicked.connect(self._toggle_cells)
            self.toggle_cells_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            #self.contrast_button.setMinimumSize(200, 25)  # Adjust the size as needed
            self.toggle_cells_button.setEnabled(False)
            
            
            ################### SETUP UI LAYOUT ###################
            self.parameter_layout = QVBoxLayout()
            self.parameter_layout.addWidget(self.parameters_title)
            ### Parameter row - image processing and correcetions ###
            self.prow_layout = QHBoxLayout()
            self.prow_layout.addStretch(1)
            self.prow_layout.addWidget(self.cell_intensity_thresh_spinbox)  # Depth
            self.prow_layout.addWidget(self.cell_intensity_thresh_title, alignment=Qt.AlignLeft)
            self.prow_layout.addWidget(self.min_cell_size_spinbox)  # Median threshold
            self.prow_layout.addWidget(self.min_cell_size_title, alignment=Qt.AlignLeft)
            self.prow_layout.addWidget(self.bg_thresh_spinbox)  # Background threshold
            self.prow_layout.addWidget(self.bg_thresh_title, alignment=Qt.AlignLeft)
            self.prow_layout.addStretch(1)
            
            self.prow_layout2 = QHBoxLayout()
            self.prow_layout2.addStretch(1)
            self.prow_layout2.addWidget(self.processing_min_size_spinbox)  # Depth
            self.prow_layout2.addWidget(self.processing_min_size_title, alignment=Qt.AlignLeft)
            self.prow_layout2.addWidget(self.processing_area_thresh)  # Median threshold
            self.prow_layout2.addWidget(self.processing_area_thresh_title, alignment=Qt.AlignLeft)
            self.prow_layout2.addWidget(self.processing_dilation_iter)  # Background threshold
            self.prow_layout2.addWidget(self.processing_dilation_iter_title, alignment=Qt.AlignLeft)
            self.prow_layout2.addWidget(self.processing_min_dist_spinbox)
            self.prow_layout2.addWidget(self.processing_min_dist_title, alignment=Qt.AlignLeft)
            self.prow_layout2.addStretch(1)
            
            # Combine with parameter layout
            self.parameter_layout.addLayout(self.prow_layout)
            self.parameter_layout.addLayout(self.prow_layout2)
            self.parameter_layout.addWidget(h_line)
            self.parameter_layout.addWidget(self.section_slider)
            self.parameter_layout.addWidget(self.section_slider_title)
            
            # Slider index position
            self.slider_text_layout = QHBoxLayout()
            self.slider_text_layout.addStretch(1)
            self.slider_text_layout.addWidget(self.current_idx_spinbox)
            self.slider_text_layout.addWidget(self.section_idx_label)
            self.slider_text_layout.addStretch(1)
            
            # Buttons - export and save
            self.button_layout = QHBoxLayout()
            self.button_layout.addWidget(self.export_button, 1)
            self.button_layout.addWidget(self.generate_button, 1)
            
            # Contrast adjustments
            self.contrast_layout.addStretch(1)
            self.contrast_layout.addWidget(self.alpha_spinbox)
            self.contrast_layout.addWidget(self.alpha_title, alignment=Qt.AlignLeft)
            self.contrast_layout.addWidget(self.beta_spinbox)
            self.contrast_layout.addWidget(self.beta_title, alignment=Qt.AlignLeft)
            self.contrast_layout.addStretch(1)
            
            ######### ADD TO MAIN LAYOUT #########
            self.layout.addLayout(self.parameter_layout)
            self.layout.addLayout(self.slider_text_layout)
            self.layout.addLayout(self.button_layout)
            #self.layout.addWidget(self.histogram_canvas)
            self.layout.addWidget(self.canvas)
            self.layout.addWidget(self.toolbar)
            self.layout.addLayout(self.contrast_layout)
            self.layout.addWidget(self.contrast_button, 1)
            self.layout.addWidget(self.toggle_cells_button, 1)
            self.setLayout(self.layout)
            
        
        def _toggle_buttons(self, toggle: bool):
            """
            Disable all buttons to prevent user input during processing.
            """
            # Counting parameters
            self.cell_intensity_thresh_spinbox.setEnabled(toggle)
            self.min_cell_size_spinbox.setEnabled(toggle)
            self.bg_thresh_spinbox.setEnabled(toggle)
            # Optional parameters
            self.processing_min_size_spinbox.setEnabled(toggle)
            self.processing_area_thresh.setEnabled(toggle)
            self.processing_dilation_iter.setEnabled(toggle)
            self.processing_min_dist_spinbox.setEnabled(toggle)
            # Sliders
            self.section_slider.setEnabled(toggle)
            self.current_idx_spinbox.setEnabled(toggle)
            # Buttons
            self.export_button.setEnabled(toggle)
            self.generate_button.setEnabled(toggle)
            # Contrast
            self.alpha_spinbox.setEnabled(toggle)
            self.beta_spinbox.setEnabled(toggle)
            self.contrast_button.setEnabled(toggle)
            self.toggle_cells_button.setEnabled(toggle)
            
            
        def _redraw(self):
            """
            Function to toggle contrast for the current image in the canvas.
            """
            self.current_section = gui_shared.auto_contrast(self.original_section.copy(), 
                                                            alpha=self.alpha_spinbox.value(), 
                                                            beta=self.beta_spinbox.value())
            
            # Keep track of X and Y limits
            x_limits = self.ax.get_xlim()
            y_limits = self.ax.get_ylim()
            self.ax.clear()

            # Reset the limits if they were set before.
            if x_limits != (0.0, 1.0) and y_limits != (0.0, 1.0):
                self.ax.set_xlim(x_limits)
                self.ax.set_ylim(y_limits)

            self.ax.imshow(self.current_section, cmap='gray')
            self.ax.contour(self.current_mask, colors='yellow', alpha=0.5, linewidths=0.5)
            if self.show_cells and len(self.cells) > 0:
                self.points = self.ax.plot(self.cells[:, 2], self.cells[:, 1], 'r.')
            #self.ax.set_title('PyQt Matplotlib Example')
            self.canvas.draw()
            
            
        def _update_idx_from_slider(self):
            """
            Update the section index when the slider is moved.
            """
            self.idx = self.section_slider.value() - 1
            self.current_idx_spinbox.setValue(self.section_slider.value())
            
            
        def _update_idx_from_spinbox(self):
            """
            Update the section index when the spinbox is moved.
            """
            self.idx = self.current_idx_spinbox.value() - 1
            self.section_slider.setValue(self.current_idx_spinbox.value())
            
            
        def _update(self):
            """
            Updates the preview window with the current parameters from the main window.
            """
            self.cell_intensity_thresh_spinbox.setValue(self.parent.cell_intensity_thresh_spinbox.value())
            self.min_cell_size_spinbox.setValue(self.parent.min_cell_size_spinbox.value())
            self.bg_thresh_spinbox.setValue(self.parent.bg_thresh_spinbox.value())
            self.processing_min_size_spinbox.setValue(self.parent.processing_min_size_spinbox.value())
            self.processing_area_thresh.setValue(self.parent.processing_area_thresh_spinbox.value())
            self.processing_dilation_iter.setValue(self.parent.processing_dilation_iter_spinbox.value())
            self.processing_min_dist_spinbox.setValue(self.parent.processing_min_dist_spinbox.value())
            
            
        def _export_parameters(self):
            """
            Exports the preview values back to the main window.
            """
            self.parent.cell_intensity_thresh_spinbox.setValue(self.cell_intensity_thresh_spinbox.value())
            self.parent.min_cell_size_spinbox.setValue(self.min_cell_size_spinbox.value())
            self.parent.bg_thresh_spinbox.setValue(self.bg_thresh_spinbox.value())
            self.parent.processing_min_size_spinbox.setValue(self.processing_min_size_spinbox.value())
            self.parent.processing_area_thresh_spinbox.setValue(self.processing_area_thresh.value())
            self.parent.processing_dilation_iter_spinbox.setValue(self.processing_dilation_iter.value())
            self.parent.processing_min_dist_spinbox.setValue(self.processing_min_dist_spinbox.value())
            
    
        def _toggle_cells(self):
            """
            Toggle the cell points from the plot.
            """
            self.show_cells = not self.show_cells
            if not self.show_cells and len(self.points) > 0:
                for pt in self.points:
                    pt.remove()
            else:
                if len(self.cells) > 0:
                    self.points = self.ax.plot(self.cells[:, 2], self.cells[:, 1], 'r.')
            self.canvas.draw()
            
        
        def _thread_preview(self):
            """
            Thread the preview function.
            """
            t1 = Thread(target=self._generate_preview) 
            t1.start()
        
        
        def _generate_preview(self):
            """
            Generate the selected section with the selected parameters.
            """
            start = time.time()
            self._toggle_buttons(False)
            intensity_threshold = self.cell_intensity_thresh_spinbox.value()
            size_threshold = self.min_cell_size_spinbox.value()
            bg_thresh = self.bg_thresh_spinbox.value()
            min_size = self.processing_min_size_spinbox.value()
            area_thresh = self.processing_area_thresh.value()
            dilation_iter = self.processing_dilation_iter.value()
            min_dist = self.processing_min_dist_spinbox.value()
            
            section_files = glob.glob(os.path.join(self.parent.input_path, "*.tif"))
            
            # Count cells for the selected section
            print("Counting cells for index " + str(self.idx + 1) + "...")
            image = readSectionTif(section_files[self.idx])
            image[image < 0] = 0
            cells = gui_shared.get_cell_locations(image, index = self.idx, 
                                                  intensity_threshold = intensity_threshold, size_threshold = size_threshold, bg_threshold = bg_thresh, 
                                                  min_size = min_size, area_threshold = area_thresh, dilation_iter = dilation_iter, min_distance=min_dist)
            self.cells = cells
            self.last_idx = self.idx
            
            # Load the stitched image
            print("Loading images into PyPlot...")
            TEST_IMG = image

            # Set to internal variables
            self.original_section = TEST_IMG
            self.current_section = self.original_section.copy()
            
            # Generate threshold mask
            self.current_mask = gui_shared.remove_artifacts(self.original_section, 
                                                            thresh=self.bg_thresh_spinbox.value(),
                                                            min_size=self.processing_min_size_spinbox.value(),
                                                            area_threshold=self.processing_area_thresh.value(),
                                                            num_iter=self.processing_dilation_iter.value(),
                                                            erode=True, convex_hull=False, debug=False)
            self.current_mask[self.current_mask > 0] = 1
            self.current_mask[self.current_mask < 0] = 0
            
            # Keep track of X and Y limits
            x_limits = self.ax.get_xlim()
            y_limits = self.ax.get_ylim()
            self.ax.clear()
            if x_limits != (0.0, 1.0) and y_limits != (0.0, 1.0):
                self.ax.set_xlim(x_limits)
                self.ax.set_ylim(y_limits)
                
            # Display the image
            self.current_section = gui_shared.auto_contrast(self.original_section.copy(), 
                                                            alpha=self.alpha_spinbox.value(), 
                                                            beta=self.beta_spinbox.value())
            self.ax.imshow(self.current_section, cmap='gray')
            if len(self.cells) > 0:
                self.points = self.ax.plot(cells[:, 2], cells[:, 1], 'r.')
            self.ax.contour(self.current_mask, colors='yellow', alpha=0.5, linewidths=0.5)
            self.canvas.draw()

            self._toggle_buttons(True)
            self.show_cells = True
            end = time.time()
            print("Preview generation took " + str(end - start) + " seconds.")
            

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
            self.tab_counting = CountingTab(self)
            self.tabs.addTab(self.tab_counting, "Cell Counting")
            self.setCentralWidget(self.tabs)
        
    app = QApplication(sys.argv)
    window = U01App()
    window.show()
    sys.exit(app.exec())
