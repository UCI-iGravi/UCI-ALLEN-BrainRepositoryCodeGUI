"""
A simple GUI for the U01 workflow.
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
from gui.gui_cell_counting import CountingTab
from gui.gui_stitching import StitchingTab
from gui.gui_registration import RegistrationTab

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
        self.tab_counting = CountingTab(self)
        self.tab_registration = RegistrationTab(self)
        
        self.tabs.addTab(self.tab_stitching, "Stitching")
        self.tabs.addTab(self.tab_counting, "Cell Counting")
        self.tabs.addTab(self.tab_registration, "Registration")

        self.setCentralWidget(self.tabs)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = U01App()
    window.show()
    sys.exit(app.exec())
