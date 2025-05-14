"""
A simple GUI framework for the U01 stitching workflow.
Andy Thai
andy.thai@uci.edu
"""

# Import standard libraries
import os
import sys
import time
import glob
import logging
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
import gui.gui_shared as gui_shared

# Global formatting variables
BOLD_STYLE = "font-weight: bold; color: black"
DEFAULT_STYLE = "color: black"
TITLE_SPACING = " " * 12


class StitchingTab(QWidget):
    """
    Stitching tab for loading up tile data and stitching them together.
    """
    def __init__(self, app):
        super().__init__()
        
        # Set up tab settings and layout.
        self.app = app
        self.layout = QVBoxLayout()
        
        # Declare variables to keep track of file paths and settings
        self.input_path = None
        self.output_path = None
        self.bezier_path = os.path.normpath(os.getcwd()) + "\\bezier16x.pkl"
        self.num_sections = 0
        self.preview_window = None
        
        # Setup Bezier patch file information
        corners1 = np.asarray([[33, 10], [796, 21], [30, 813], [793, 818]])
        corners2 = np.asarray([[20, 20], [776, 20], [20, 794], [776, 794]])
        self.H, _ = cv2.findHomography(corners1, corners2)
        gridp = run_tissuecyte_stitching_classic.create_perfect_grid(42, 43, 4, 18)
        gridp = gridp[20:794, 20:776]
        
        
        ###############################################################################
        ##                                 FILE IO                                   ##
        ###############################################################################
        
        # Title
        file_io_title = "## Stitching File I/O"
        self.file_io_title = QLabel(file_io_title, alignment=Qt.AlignCenter)
        self.file_io_title.setTextFormat(Qt.MarkdownText)
        
        # Button to select folder containing tile data.
        self.input_folder_button = QPushButton("Select input tile directory\n⚠️ NO TILE DATA LOADED")
        self.input_folder_button.clicked.connect(self._select_input_path)
        self.input_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.input_folder_button.setStyleSheet(BOLD_STYLE)
        input_folder_desc = "Select the folder directory containing the tile data. The folder should contain a Mosaic text file and " + \
                            "a set of folders with the tile data. Each folder represents a section and contains TIFFs representing " + \
                            "tiles to be stitched together.\n"
        self.input_folder_desc = QLabel(input_folder_desc, alignment=Qt.AlignCenter)
        self.input_folder_desc.setWordWrap(True)
        #self.input_folder_desc.setMinimumHeight(150)
        
        # Button to select folder to output stitched data.
        self.output_folder_button = QPushButton("Select stitching output directory\n⚠️ NO OUTPUT FOLDER SELECTED")
        self.output_folder_button.clicked.connect(self._select_output_path)
        self.output_folder_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.output_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.output_folder_button.setStyleSheet(BOLD_STYLE)
        
        # Save undistorted flag
        self.save_undistorted_checkbox = QCheckBox("Save undistorted images", checked=False)
        save_undistorted_desc = "Check the 'Save undistorted images' checkbox to save the outputs without any distortion correction."
        output_folder_desc = "Select the folder directory to output the stitching output to. This will output a folder containing the " + \
                             "computed average tiles each channel of the volume and folders containing stitched sections for their " + \
                             "corresponding color channels (0: red, 1: green, 2: blue). " + save_undistorted_desc + "\n"
        self.output_folder_desc = QLabel(output_folder_desc, alignment=Qt.AlignCenter)
        self.output_folder_desc.setWordWrap(True)
        #adjust_label_min_height(self.output_folder_desc)
        #self.output_folder_desc.setMinimumHeight(52*3)
        
        # Bezier patch file
        self.bezier_button = QPushButton("Select Bezier patch file\n⚠️ NO BEZIER PATCH FILE SELECTED")
        self.bezier_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.bezier_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.bezier_button.clicked.connect(self._select_bezier_path)
        
        # If the bezier patch file exists, load the file and update the button text.
        if os.path.exists(self.bezier_path):
            self.bezier_button.setText("Select Bezier patch file\n✅ " + self.bezier_path)
            kx, ky = joblib.load(self.bezier_path)
            # Double the size to preserve sampling , need to downsample later
            self.pX_, self.pY_ = run_tissuecyte_stitching_classic.get_deformation_map(gridp.shape[0], gridp.shape[1], kx, ky)
        # If the bezier patch file does not exist, set the bezier path to None and require user to upload.
        else:
            self.bezier_path = None  # Invalid path or missing file.
        bezier_desc = "Select the filepath of the Bezier patch file to use for stitching correction. By default, the application " + \
                      "automatically looks for 'bezier16x.pkl' in the current working directory.\n"
        self.bezier_desc = QLabel(bezier_desc, alignment=Qt.AlignCenter)
        self.bezier_desc.setTextFormat(Qt.MarkdownText)
        self.bezier_desc.setWordWrap(True)
        
        # Divider line
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        
        ###############################################################################
        ##                               PARAMETERS                                  ##
        ###############################################################################
        
        # Title
        parameter_title = "## Stitching Parameters"
        self.parameter_title = QLabel(parameter_title, alignment=Qt.AlignCenter)
        self.parameter_title.setTextFormat(Qt.MarkdownText)
        
        ### BRIGHTNESS TILE CORRECTIONS PARAMS ###
        
        # Brightness normalization parameters
        bg_thresh_title = "Background threshold" + TITLE_SPACING
        self.bg_thresh_title = QLabel(bg_thresh_title)
        self.bg_thresh_spinbox = QSpinBox(minimum=0, maximum=255, singleStep=1, value=15, alignment=Qt.AlignCenter)
        #self.bg_thresh_spinbox.setSuffix(" (0-255)")
        bg_thresh_desc = "**Background threshold** helps " + \
                         "determines which pixels are considered as brain tissue and affects the quality of brightness correction. " + \
                         "Pixels that are considered background will not be affected by brightness correction methods. This parameter " + \
                         "ensures that tiling artifacts are not introduced into the background."
        self.bg_thresh_desc = QLabel(bg_thresh_desc, alignment=Qt.AlignCenter)
        self.bg_thresh_desc.setWordWrap(True)
        self.bg_thresh_desc.setTextFormat(Qt.MarkdownText)
        
        median_thresh_title = "Median threshold" + TITLE_SPACING  # Median
        self.median_thresh_title = QLabel(median_thresh_title)
        self.median_thresh_spinbox = QSpinBox(minimum=0, maximum=65535, singleStep=1, value=20, alignment=Qt.AlignCenter)
        #self.median_thresh_spinbox.setSuffix(" (0-65535)")
        median_thresh_desc = "**Median threshold** helps determines which tiles are considered background and excludes tiles with median values under " + \
                             "the threshold. This prevents background tiles from overly influencing the average tile values. " + \
                             "This parameter affects the quality of edge blending between adjacent tiles."
        self.median_thresh_desc = QLabel(median_thresh_desc, alignment=Qt.AlignCenter)
        self.median_thresh_desc.setWordWrap(True)
        self.median_thresh_desc.setTextFormat(Qt.MarkdownText)
        
        # Depth parameter
        depth_title = "Depth" + TITLE_SPACING
        self.depth_title = QLabel(depth_title)
        self.depth_spinbox = QSpinBox(minimum=0, maximum=65535, singleStep=1, value=1, alignment=Qt.AlignCenter)
        depth_desc = "The **depth** parameter affects the indexing when computing and retrieving section data. " + \
                     "Leave this at 1 unless you know what you're doing."
        preview_desc = "\n\nYou may preview how different parameters affect image processing using the 'Preview' button."
        self.depth_desc = QLabel(depth_desc + preview_desc, alignment=Qt.AlignCenter)
        self.depth_desc.setTextFormat(Qt.MarkdownText)
        
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
        
        # Stitching instructions
        stitch_desc = "Press 'Stitch' to start the stitching process. You may select the number of processes to use for stitching. " + \
                      "More processes will speed up the stitching, but will take up more resources. " + \
                      "By default, the number of processes is set to the number of CPU cores - 3.\n"
        self.stitch_desc = QLabel(stitch_desc, alignment=Qt.AlignCenter)
        self.stitch_desc.setWordWrap(True)
        
        # Stitch button
        self.stitch_button = QPushButton("Stitch")
        self.stitch_button.setMinimumSize(100, 50)  # Adjust the size as needed
        self.stitch_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.stitch_button.clicked.connect(self._thread_stitching)
        self.stitch_button.setEnabled(False)  # Initially disabled
        
        # Number of processes
        num_cpus = os.cpu_count()
        self.num_processes_spinbox = QSpinBox(minimum=1, maximum=num_cpus, singleStep=1, value=max(num_cpus - 3, 1), alignment=Qt.AlignCenter)
        num_processes_title = "Num. processes"
        self.num_processes_title = QLabel(num_processes_title)
       
        
        ################### SETUP UI LAYOUT ###################
        
        # Input folder
        self.io_layout = QVBoxLayout()
        self.io_layout.addWidget(self.file_io_title, alignment=Qt.AlignCenter)
        self.io_layout.addWidget(self.input_folder_button)
        self.io_layout.addWidget(self.input_folder_desc, alignment=Qt.AlignTop)
        
        # Output folder
        self.output_layout = QHBoxLayout()
        #self.output_layout.addStretch(1)
        self.output_layout.addWidget(self.output_folder_button, 1)
        self.output_layout.addWidget(self.save_undistorted_checkbox)
        #self.output_layout.addStretch(1)
        #self.output_layout.setAlignment(Qt.AlignCenter)
        self.io_layout.addLayout(self.output_layout)
        self.io_layout.addWidget(self.output_folder_desc, alignment=Qt.AlignTop)
        
        # Bezier patch file
        self.io_layout.addWidget(self.bezier_button)
        self.io_layout.addWidget(self.bezier_desc, alignment=Qt.AlignTop)
        self.io_layout.addWidget(h_line)
        
        # Parent parameter layout
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.addWidget(self.parameter_title)

        ### Parameter row - image processing and correcetions ###
        self.prow_layout = QHBoxLayout()
        self.prow_layout.addStretch(1)
        self.prow_layout.addWidget(self.bg_thresh_spinbox)  # Background threshold
        self.prow_layout.addWidget(self.bg_thresh_title, alignment=Qt.AlignLeft)
        self.prow_layout.addWidget(self.median_thresh_spinbox)  # Median threshold
        self.prow_layout.addWidget(self.median_thresh_title, alignment=Qt.AlignLeft)
        self.prow_layout.addWidget(self.depth_spinbox)  # Depth
        self.prow_layout.addWidget(self.depth_title, alignment=Qt.AlignLeft)
        self.prow_layout.addStretch(1)
        # Add to main layout
        self.parameter_layout.addLayout(self.prow_layout)
        self.parameter_layout.addWidget(self.bg_thresh_desc, alignment=Qt.AlignTop)
        self.parameter_layout.addWidget(self.median_thresh_desc, alignment=Qt.AlignTop)
        self.parameter_layout.addWidget(self.depth_desc, alignment=Qt.AlignTop)
        
        # Preview button
        self.parameter_layout.addWidget(self.preview_button)
        self.parameter_layout.addWidget(h_line2)
        
        ### Stitch buttons ###
        self.run_buttons_layout = QHBoxLayout()
        self.run_buttons_layout.addWidget(self.stitch_button, stretch=1)
        self.run_buttons_layout.addWidget(self.num_processes_spinbox) # Number of processes        
        self.run_buttons_layout.addWidget(self.num_processes_title)
        
        ######### ADD TO MAIN LAYOUT #########
        self.layout.addLayout(self.io_layout)
        self.layout.addLayout(self.parameter_layout)
        self.layout.addWidget(self.metadata_info, alignment=Qt.AlignTop)
        self.layout.addWidget(self.stitch_desc, alignment=Qt.AlignTop)
        self.layout.addLayout(self.run_buttons_layout)
        #self.run_buttons_layout.setAlignment(Qt.AlignCenter)
        
        # End
        self.setLayout(self.layout)
        
    def _select_input_path(self):
        """
        Select the path to load the input tiles from and saves the selected folder path internally.
        """
        folder_path = QFileDialog.getExistingDirectory(None, "Select folder directory with tile data")
        if folder_path == '':  # If user cancels out of the dialog, exit.
            return
        
        # Check if the selected folder contains a Mosaic file.
        mosaic_files = glob.glob(os.path.join(folder_path, "Mosaic*.txt"))
        
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        
        if not mosaic_files:  # If user selects a folder without a Mosaic file, exit and output an error message.
            err_msg = "Selected folder does not contain a Mosaic file. Please check your folder layout and select a folder with a Mosaic file."
            error_dialog.setText(err_msg)
            error_dialog.exec_()
            return
        elif len(mosaic_files) > 1:  # If user selects a folder with multiple Mosaic files, exit and output an error message.
            err_msg = "Selected folder contains multiple Mosaic files. Please select a folder with only one Mosaic file."
            error_dialog.setText(err_msg)
            error_dialog.exec_()
            return
        
        self.input_path = folder_path
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        self.preview_button.setEnabled(True)  # Enable preview button
        if self.input_path and self.output_path and self.bezier_path:  # Enable stitch button if all required paths are provided.
            self.stitch_button.setEnabled(True)
        
        # Update button information
        print(f'Selected input folderpath: {folder_path}')
        
        self.input_folder_button.setText(f"Select input tile directory\n✅ {os.path.normpath(self.input_path)}")
        self.input_folder_button.setStyleSheet(DEFAULT_STYLE)
        
        # Output data metrics
        folder_paths = glob.glob(os.path.join(folder_path, "*/"))  # Get paths to all directories within folder_path
        folder_paths = [p for p in folder_paths if not p.endswith("\\trigger\\") and not p.endswith("/trigger")]
        self.num_sections = len(folder_paths)
        idx_0_tifs = glob.glob(os.path.join(folder_paths[0], "*.tif"))  # Get paths to all TIFFs in the first section
        num_tiles_per_section = len(idx_0_tifs)
        sample_tif = run_tissuecyte_stitching_classic.read_image(idx_0_tifs[0])
        metadata_str = "**Metadata**\n\nNumber of sections: " + str(self.num_sections) + \
                       "\n\nTiles per section: " + str(num_tiles_per_section) + \
                       "\n\nTile resolution: " + str(sample_tif.shape)
        self.metadata_info.setText(metadata_str)
        
        # Setup stitching preview window values
        self.preview_window = self.PreviewWindow(self)
        
    
    def _select_output_path(self, check_empty=True):
        """
        Select the path to save the outputs to and saves the selected folder path internally.
        """
        folder_path = QFileDialog.getExistingDirectory(None, "Select empty folder to save the output to")
        
        check_empty = False
        if check_empty and os.listdir(folder_path):  # If user selects a non-empty folder, exit and output an error message.
            print("Selected folder is not empty. Please select an empty folder to export TIFFs to.")
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
        if self.input_path and self.output_path and self.bezier_path:
            self.stitch_button.setEnabled(True)
            
        # Update button information
        print(f'Selected output folderpath: {folder_path}')
        
        self.output_folder_button.setText(f"Select stitching output directory\n✅ {os.path.normpath(self.output_path)}")
        self.output_folder_button.setStyleSheet(DEFAULT_STYLE)
    
    
    def _select_bezier_path(self):
        # TODO
        pass
    
    
    def _popup_preview(self):
        """
        Enable the preview window with the selected parameters on a selected section.
        """
        # Update parameters from main window here.
        self.preview_window.update()
        self.preview_window.exec_()
        
        
    def _disable_buttons(self):
        """
        Disable all buttons to prevent user input during processing.
        """
        self.input_folder_button.setEnabled(False)
        self.output_folder_button.setEnabled(False)
        self.save_undistorted_checkbox.setEnabled(False)
        self.bezier_button.setEnabled(False)
        self.bg_thresh_spinbox.setEnabled(False)
        self.median_thresh_spinbox.setEnabled(False)
        self.depth_spinbox.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.stitch_button.setEnabled(False)
        self.num_processes_spinbox.setEnabled(False)
        
        
    def _enable_buttons(self):
        """
        Enable all buttons after stitching is completed.
        """
        self.input_folder_button.setEnabled(True)
        self.output_folder_button.setEnabled(True)
        self.save_undistorted_checkbox.setEnabled(True)
        self.bezier_button.setEnabled(True)
        self.bg_thresh_spinbox.setEnabled(True)
        self.median_thresh_spinbox.setEnabled(True)
        self.depth_spinbox.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.stitch_button.setEnabled(True)
        self.num_processes_spinbox.setEnabled(True)
        
        
    def _thread_stitching(self):
        """
        Thread the stitching function.
        """
        t1 = Thread(target=self._run_stitching) 
        t1.start() 
        
        
    def _run_stitching(self):
        """
        Run the stitching process with the selected parameters.
        """
        self._disable_buttons()
        
        # Setup backend for joblib import
        joblib_backend = None
        if sys.platform == 'win32':
            joblib_backend = 'multiprocessing'

        n_threads = self.num_processes_spinbox.value()

        # Run main
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        #parser = argparse.ArgumentParser()
        #parser.add_argument('--input_dir', type=str)
        #parser.add_argument('--output_dir', type=str)
        #parser.add_argument('--depth', default = 1, type=int)
        #parser.add_argument('--sectionNum', default = 0, type=int)
        #parser.add_argument('--save_undistorted', default=False, type=bool)
        #args = parser.parse_args()

        root_dir = os.path.join(self.input_path, '')
        output_dir = os.path.join(self.output_path, '')
        depth = self.depth_spinbox.value()
        sectionNum = -1  # Default -1
        save_undistorted = self.save_undistorted_checkbox.isChecked()
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        print("Creating Stitching JSON for sections...")
        mosaic_data, section_jsons = run_tissuecyte_stitching_classic.get_section_data(root_dir, n_threads, depth, sectionNum)

        channel_count = int(mosaic_data['channels'])
        print("Creating intermediate directories...")
        for ch in range(channel_count):
            ch_dir  = os.path.join(output_dir, "stitched_ch{}".format(ch),"")
            if not os.path.isdir(ch_dir):
                os.mkdir(ch_dir)

        if save_undistorted:
            undistorted_dir = output_dir + "/undistorted"

            if not os.path.isdir(undistorted_dir):
                os.mkdir(undistorted_dir)

            for ch in range(channel_count):
                ch_dir = os.path.join(undistorted_dir, "ch{}".format(ch),"")
                if not os.path.isdir(ch_dir):
                    os.mkdir(ch_dir)

        average_tiles = []
        if sectionNum == -1:
            avg_tiles_dir = os.path.join(output_dir, "avg_tiles")
            print("Generating average tiles...")
            run_tissuecyte_stitching_classic.generate_avg_tiles(section_jsons, avg_tiles_dir, 
                                                                n_threads, median_thresh=self.median_thresh_spinbox.value())
            for i in range(4):
                average_tiles.append(run_tissuecyte_stitching_classic.load_average_tile(os.path.join(avg_tiles_dir,"avg_tile_" + str(i) + ".tif")))
        else:
            for i in range(4):
                average_tiles.append(np.ones((832,832)))
        print("Stitching...")
        #Parallel(n_jobs=1, backend=joblib_backend)(delayed(stitch_section)(section_json,average_tiles, output_dir) for section_json in tqdm(section_jsons))
        joblib.Parallel(n_jobs=n_threads, verbose=13)(
            joblib.delayed(run_tissuecyte_stitching_classic.stitch_section)(
                section_json, average_tiles, output_dir, self.H, self.pX_, self.pY_, 
                self.bg_thresh_spinbox.value(), None, save_undistorted) for section_json in section_jsons)
        
        self._enable_buttons()


        
    class PreviewWindow(QDialog):
        def __init__(self, parent):
            super().__init__()
            POPUP_HEIGHT = 800
            POPUP_WIDTH = 600
            self.setWindowTitle("Stitching Preview")
            self.resize(POPUP_WIDTH, POPUP_HEIGHT)
            
            # Setup layout
            self.layout = QVBoxLayout()
            
            # Setup preview window settings and variables
            self.parent = parent  # Parent app
            self.idx = 0  # Section index to preview
            self.last_idx = -1  # Last section index previewed
            self.last_median_thresh = -1  # Last median threshold previewed
            self.original_section = None  # Original previewed section
            self.current_section = None   # Current previewed section (with parameters applied)
            self.current_median_mask = None # Current median mask for the section
            self.current_mask = None # Current background thresh mask for the section
            
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
            # Background threshold
            
            bg_thresh_title = "Background threshold" + TITLE_SPACING
            self.bg_thresh_title = QLabel(bg_thresh_title)
            self.bg_thresh_spinbox = QSpinBox(minimum=0, maximum=255, singleStep=1, value=15, alignment=Qt.AlignCenter)
            # Median threshold
            median_thresh_title = "Median threshold" + TITLE_SPACING  # Median
            self.median_thresh_title = QLabel(median_thresh_title)
            self.median_thresh_spinbox = QSpinBox(minimum=0, maximum=65535, singleStep=1, value=20, alignment=Qt.AlignCenter)
            # Depth parameter
            depth_title = "Depth" + TITLE_SPACING
            self.depth_title = QLabel(depth_title)
            self.depth_spinbox = QSpinBox(minimum=0, maximum=65535, singleStep=1, value=1, alignment=Qt.AlignCenter)
            
            self.update()  # Update slider and spinbox values with parent values             
            
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
            self.contrast_button.clicked.connect(self._contrast_button_click)
            self.contrast_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.contrast_button.setMinimumSize(200, 25)  # Adjust the size as needed
            self.contrast_button.setEnabled(False)
            
            
            ################### SETUP UI LAYOUT ###################
            self.parameter_layout = QVBoxLayout()
            self.parameter_layout.addWidget(self.parameters_title)
            ### Parameter row - image processing and correcetions ###
            self.prow_layout = QHBoxLayout()
            self.prow_layout.addStretch(1)
            self.prow_layout.addWidget(self.bg_thresh_spinbox)  # Background threshold
            self.prow_layout.addWidget(self.bg_thresh_title, alignment=Qt.AlignLeft)
            self.prow_layout.addWidget(self.median_thresh_spinbox)  # Median threshold
            self.prow_layout.addWidget(self.median_thresh_title, alignment=Qt.AlignLeft)
            self.prow_layout.addWidget(self.depth_spinbox)  # Depth
            self.prow_layout.addWidget(self.depth_title, alignment=Qt.AlignLeft)
            self.prow_layout.addStretch(1)
            
            # Combine with parameter layout
            self.parameter_layout.addLayout(self.prow_layout)
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
            self.layout.addWidget(self.histogram_canvas)
            self.layout.addWidget(self.canvas)
            self.layout.addWidget(self.toolbar)
            self.layout.addLayout(self.contrast_layout)
            self.layout.addWidget(self.contrast_button, 1)
            self.setLayout(self.layout)
            
        
        def _disable_buttons(self):
            """
            Disable all buttons to prevent user input during processing.
            """
            self.bg_thresh_spinbox.setEnabled(False)
            self.median_thresh_spinbox.setEnabled(False)
            self.depth_spinbox.setEnabled(False)
            self.section_slider.setEnabled(False)
            self.current_idx_spinbox.setEnabled(False)
            self.export_button.setEnabled(False)
            self.generate_button.setEnabled(False)
            self.alpha_spinbox.setEnabled(False)
            self.beta_spinbox.setEnabled(False)
            self.contrast_button.setEnabled(False)
            
        def _enable_buttons(self):
            """
            Enable all buttons after preview is completed.
            """
            self.bg_thresh_spinbox.setEnabled(True)
            self.median_thresh_spinbox.setEnabled(True)
            self.depth_spinbox.setEnabled(True)
            self.section_slider.setEnabled(True)
            self.current_idx_spinbox.setEnabled(True)
            self.export_button.setEnabled(True)
            self.generate_button.setEnabled(True)
            self.alpha_spinbox.setEnabled(True)
            self.beta_spinbox.setEnabled(True)
            self.contrast_button.setEnabled(True)
            
            
        def _contrast_button_click(self):
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
            self.ax.contour(self.current_mask, colors='yellow', alpha=0.5)
            self.ax.contour(self.current_median_mask, colors='green', alpha=0.5)
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
            self.bg_thresh_spinbox.setValue(self.parent.bg_thresh_spinbox.value())
            self.median_thresh_spinbox.setValue(self.parent.median_thresh_spinbox.value())
            self.depth_spinbox.setValue(self.parent.depth_spinbox.value())
            
            
        def _export_parameters(self):
            """
            Exports the preview values back to the main window.
            """
            self.parent.bg_thresh_spinbox.setValue(self.bg_thresh_spinbox.value())
            self.parent.median_thresh_spinbox.setValue(self.median_thresh_spinbox.value())
            self.parent.depth_spinbox.setValue(self.depth_spinbox.value())
            
        
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
            self._disable_buttons()
            
            save_undistorted = self.parent.save_undistorted_checkbox.isChecked()
            
            # Temporary preview output directory
            output_dir = "./temp/"
            output_mask_dir = "./temp_mask/"
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            if not os.path.isdir(output_mask_dir):
                os.mkdir(output_mask_dir)
            
            print("Getting section data...")
            mosaic_data, section_jsons = run_tissuecyte_stitching_classic.get_section_data(self.parent.input_path + "/", 
                                                                                           1, 
                                                                                           self.depth_spinbox.value(), 
                                                                                           self.idx)
            channel_count = int(mosaic_data['channels'])
            print("Creating intermediate directories...")
            for ch in range(channel_count):
                ch_dir = os.path.join(output_dir, "stitched_ch{}".format(ch), "")
                mask_ch_dir = os.path.join(output_mask_dir, "stitched_ch{}".format(ch), "")
                # Main stitching output
                if not os.path.isdir(ch_dir):
                    os.mkdir(ch_dir)
                # Median preview
                if not os.path.isdir(mask_ch_dir):
                    os.mkdir(mask_ch_dir)
                    
            # Check if preview image already exists
            ch = 0
            sample_path = os.path.join(output_dir, "stitched_ch{}".format(ch), section_jsons[0]['slice_fname']+"_{}.tif".format(ch))
            sample_mask = os.path.join(output_mask_dir, "stitched_ch{}".format(ch), section_jsons[0]['slice_fname']+"_{}.tif".format(ch))
            print("Checking if preview image already exists at " + sample_path + "...")
            #sample_path = glob.glob(os.path.join(output_dir + "/stitched_ch0/*.tif"))[0]
            if os.path.exists(sample_path):
                print("Preview image exists! Skipping preview stitching.")
                skip_stitching = True
            else:
                print("Preview image does not exist! Generating preview image...")
                skip_stitching = False
                
            # Setup placeholder average tiles
            average_tiles = []
            for i in range(4):
                average_tiles.append(np.ones((832,832)))
                
            # Stitch and generate the preview image if it doesn't exist.
            if not skip_stitching:
                print("Stitching section...")
                run_tissuecyte_stitching_classic.stitch_section(section_jsons[0], average_tiles, output_dir, 
                                                                self.parent.H, self.parent.pX_, self.parent.pY_, 
                                                                self.bg_thresh_spinbox.value(), ch, save_undistorted)
                print("Stitching done.")
                
            # Redo median mask binary stitch if needed.
            if self.last_median_thresh != self.median_thresh_spinbox.value() or self.last_idx != self.idx:
                print("Stitching median mask preview...")
                median_json = section_jsons[0].copy()
                run_tissuecyte_stitching_classic.stitch_section(median_json, average_tiles, output_mask_dir, 
                                                                self.parent.H, self.parent.pX_, self.parent.pY_, 
                                                                self.bg_thresh_spinbox.value(), ch, save_undistorted, 
                                                                self.median_thresh_spinbox.value())
                self.current_median_mask = run_tissuecyte_stitching_classic.read_image(sample_mask).T
                self.current_median_mask = np.flip(self.current_median_mask, axis=0)
                self.current_median_mask = np.flip(self.current_median_mask, axis=1)
                self.current_median_mask = np.squeeze(self.current_median_mask)
                print("Median mask done.")
                self.last_median_thresh = self.median_thresh_spinbox.value()
            self.last_idx = self.idx
            
                
            # Load the stitched image
            print("Loading images into PyPlot...")
            TEST_IMG = run_tissuecyte_stitching_classic.read_image(sample_path).T
            TEST_IMG = np.flip(TEST_IMG, axis=0)
            TEST_IMG = np.flip(TEST_IMG, axis=1)
            TEST_IMG = np.squeeze(TEST_IMG)
            
            
            
            # Set to internal variables
            self.original_section = TEST_IMG
            self.current_section = self.original_section.copy()
            
            # Generate threshold mask
            self.current_mask = run_tissuecyte_stitching_classic.generate_mask(self.original_section, self.bg_thresh_spinbox.value())
            
            # Keep track of X and Y limits
            x_limits = self.ax.get_xlim()
            y_limits = self.ax.get_ylim()
            self.ax.clear()
            if x_limits != (0.0, 1.0) and y_limits != (0.0, 1.0):
                self.ax.set_xlim(x_limits)
                self.ax.set_ylim(y_limits)
                
            # Compute and display section histogram values
            hist, bins = np.histogram(self.original_section.flatten(), bins=256, range=[0, 50])

            # Compute the width of each bin for plotting
            bin_width = bins[1] - bins[0]

            # Plot the histogram
            self.histogram_ax.bar(bins[:-1], hist, width=bin_width, color='blue', alpha=0.7)
            self.histogram_ax.set_title('Section Histogram')
            self.histogram_ax.set_xlabel('Pixel Value')
            self.histogram_ax.set_ylabel('Frequency')
            self.histogram_canvas.draw()

            # Display the image
            self.current_section = gui_shared.auto_contrast(self.original_section.copy(), 
                                                            alpha=self.alpha_spinbox.value(), 
                                                            beta=self.beta_spinbox.value())
            self.ax.imshow(self.current_section, cmap='gray')
            self.ax.contour(self.current_mask, colors='yellow', alpha=0.5)
            self.ax.contour(self.current_median_mask, colors='green', alpha=0.5)
            #self.ax.set_title('PyQt Matplotlib Example')
            self.canvas.draw()
                        
            self._enable_buttons()
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
            self.tab_stitching = StitchingTab(self)
            #self.tab_counting = CountingTab(self)
            self.tabs.addTab(self.tab_stitching, "Stitching")
            #self.tabs.addTab(self.tab_counting, "Cell Counting")
            self.setCentralWidget(self.tabs)
            
    app = QApplication(sys.argv)
    window = U01App()
    window.show()
    sys.exit(app.exec())
