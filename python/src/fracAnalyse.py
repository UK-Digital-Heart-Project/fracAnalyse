import sys
import os
import numpy as np
import cv2
import pickle
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QApplication, QInputDialog, QSizePolicy, QMainWindow, QFileDialog, QMessageBox, QSlider, QPushButton, QLabel, QVBoxLayout, QWidget, QLineEdit, QListWidget, QHBoxLayout, QRadioButton, QGroupBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.path import Path
import matplotlib.pyplot as plt
plt.ion()

from matplotlib.widgets import EllipseSelector, PolygonSelector
from skimage.draw import ellipse
from scipy.ndimage import label
from skimage.measure import regionprops
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from loadDicom import loadDicom
from dep.NIfTI_20140122.load_untouch_nii import load_untouch_nii
from dep.exportfig.export_fig import export_fig
from fdStatistics import fdStatistics
from fracDimBatch import fracDimBatch
from dep.NIfTI_20140122.load_nii import load_nii
from dep.misc.rdir import rdir


class RoiWindow(QMainWindow):
    roi_selected = pyqtSignal(dict)  # Signal to emit when ROI is confirmed

    def __init__(self, img, roi_type):
        super().__init__()

        self.img = img  # The image to draw the ROI on
        self.roi_type = roi_type
        self.setWindowTitle(f"{roi_type.capitalize()} ROI Selection Window")
        self.setGeometry(100, 100, 800, 800)

        # Main widget container
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Set up Matplotlib figure and canvas
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Add a button to confirm the ROI (acts like "double-click to confirm" in MATLAB)
        self.confirm_button = QPushButton("Confirm ROI", self)
        self.confirm_button.clicked.connect(self.confirm_roi)  # Connect to confirm ROI function
        self.layout.addWidget(self.confirm_button)

        # Create axes in the figure
        self.ax = self.fig.add_subplot(111)

        # Display the image in grayscale
        self.ax.imshow(self.img, cmap='gray')

        # Create EllipseSelector widget to interactively select ROI
        if self.roi_type == 'ellipse':
            self.ellipse_selector = EllipseSelector(self.ax, self.onselect_ellipse, interactive=True)
            self.poly_selector = None
        elif self.roi_type == 'polygon':
            self.ellipse_selector = None
            self.poly_selector = PolygonSelector(self.ax, self.onselect_polygon, useblit=True)
            
            for artist in self.poly_selector.artists:
                artist.set_color('lightblue')  # Set polygon line color to light blue
                artist.set_linewidth(2)  # Set the line width
                artist.set_alpha(0.7)  # Set transparency to 70%
            
        # Variables to hold ROI information
        self.roi_center = None
        self.roi_width = None
        self.roi_height = None
        self.verts = []

    def onselect_ellipse(self, eclick, erelease):
        """Callback to handle the ellipse drawing"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Store the ellipse information
        self.roi_center = center
        self.roi_width = width
        self.roi_height = height

        print(f"Ellipse center: {center}, width: {width}, height: {height}")

    def onselect_polygon(self, verts):
        """Callback to handle polygon drawing"""
        self.verts = verts  # Store the vertices of the polygon
        print(f"Polygon vertices: {verts}")
        
    def confirm_roi(self):
        """Emit signal with the selected ROI upon confirmation (acts like double-click in MATLAB)"""
        if self.roi_type == 'ellipse':
            if self.roi_center and self.roi_width and self.roi_height:
                roi_data = {
                    'roi_type': 'ellipse',  # Add the roi_type here
                    'center': self.roi_center,
                    'width': self.roi_width,
                    'height': self.roi_height,
                    'image': self.img  
                }

                # Emit the signal to pass the ROI data back to the main window
                self.roi_selected.emit(roi_data)
                self.close()
            else:
                print("No ROI selected yet.")
        
            if self.ellipse_selector:
                self.ellipse_selector.set_active(False)  # Deactivate the selector
        
        elif self.roi_type == 'polygon':
            if self.verts:
                # Create a mask for the polygon
                path = Path(self.verts)
                x, y = np.meshgrid(np.arange(self.img.shape[1]), np.arange(self.img.shape[0]))
                points = np.vstack((x.flatten(), y.flatten())).T
                mask = path.contains_points(points)
                mask = mask.reshape(self.img.shape)

                # Create the binary mask and crop
                binary_mask_full = np.zeros_like(self.img, dtype=np.uint8)
                binary_mask_full[mask] = 1
                x, y, w, h = cv2.boundingRect(binary_mask_full.astype(np.uint8))
                crop_img = self.img[y:y+h, x:x+w]
                binary_mask_crop = binary_mask_full[y:y+h, x:x+w]
                
                roi_data = {
                    'roi_type': 'polygon',  # Add the roi_type here
                    'vertices': self.verts,
                    'crop_img': crop_img,
                    'binary_mask_crop': binary_mask_crop,
                    'binary_mask_full': binary_mask_full
                }
                self.roi_selected.emit(roi_data)
                self.close()
            else:
                print("No polygon ROI selected yet.")
                
            if self.poly_selector:
                self.poly_selector.set_active(False)  # Deactivate the selector

class FracAnalyse(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Fractal Analysis")
        self.resize(500, 500) 
        
        # Disable the maximize button and the resizing option
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        # Initialize the GUI elements
        self.initUI()
        
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        # Initialize the data
        self.filepath = ''
        self.img_data = None
        self.voxel_size = None
        self.gray_img_data = {}
        self.crop_img_data = {}
        self.thres_img_data = {}
        self.binary_mask_data = {}
        self.binary_mask_full_data = {}
        self.rois = {}
        self.data_z = []
        self.data_t = []
        self.fd_data = []
        self.slices_flip_status = False
        self.current_roi = 0
        self.current_slice = 0  
        self.current_phase = 0
        self.fd_mode = 'LV' 
        self.current_study = "Study_1"
 
    def initUI(self):
        # Define GUI elements for Load section
        self.load_button = QPushButton('Select Folder', self)
        self.load_button.clicked.connect(self.load_folder)

        self.load_study_button = QPushButton('Load Study', self)
        self.load_study_button.setEnabled(False)
        self.load_study_button.clicked.connect(self.load_study)

        self.reanalyse_button = QPushButton('ReAnalyse All', self)
        self.reanalyse_button.setEnabled(False)
        self.reanalyse_button.clicked.connect(self.reanalyse_all)

        self.flip_slice_loc_button = QPushButton('Flip Slice Location', self)
        self.flip_slice_loc_button.setEnabled(False)
        self.flip_slice_loc_button.clicked.connect(self.flip_slice_loc)

        self.listbox = QListWidget(self)
        self.listbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.text_z = QLineEdit(self)
        self.text_z.setReadOnly(True)

        self.text_t = QLineEdit(self)
        self.text_t.setReadOnly(True)

        self.lock_phase_button = QPushButton('Lock Phase', self)
        self.lock_phase_button.setEnabled(False)
        self.lock_phase_button.clicked.connect(self.toggle_lock_phase)

        self.load_rois_button = QPushButton("Load ROIs", self)
        self.load_rois_button.clicked.connect(self.load_rois)
        self.load_rois_button.setEnabled(False)

        self.save_rois_button = QPushButton("Save ROIs", self)
        self.save_rois_button.clicked.connect(self.save_rois)
        self.save_rois_button.setEnabled(False)

        self.roi_button_ellipse = QPushButton('ROI (Ellipse)', self)
        self.roi_button_ellipse.setEnabled(False)
        self.roi_button_ellipse.clicked.connect(self.set_roi_ellipse)

        self.roi_button_poly = QPushButton('ROI (Polygon)', self)
        self.roi_button_poly.setEnabled(False)
        self.roi_button_poly.clicked.connect(self.set_roi_polygon)

        # Radio buttons for selecting the FD mode  
        self.lv_radio = QRadioButton("LV")
        self.rv_radio = QRadioButton("RV")
        self.lv_radio.setEnabled(False)
        self.rv_radio.setEnabled(False)
        # lv_radio.setChecked(True)  # Default to LV
        self.lv_radio.toggled.connect(self.set_fd_mode)
        self.rv_radio.toggled.connect(self.set_fd_mode)

        fd_mode_layout = QHBoxLayout()  # Layout to hold radio buttons
        fd_mode_layout.addWidget(QLabel("FD Mode:"))
        fd_mode_layout.addWidget(self.lv_radio)
        fd_mode_layout.addWidget(self.rv_radio)

        self.comp_fd_button = QPushButton('Compute FD', self)
        self.comp_fd_button.setEnabled(False)
        self.comp_fd_button.clicked.connect(self.compute_fd)

        # Group the elements for Load in a QGroupBox
        load_group = QGroupBox("Load")
        load_layout = QVBoxLayout()
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.listbox)
        load_layout.addWidget(self.load_study_button)
        load_layout.addWidget(self.reanalyse_button)
        load_layout.addWidget(self.flip_slice_loc_button)
        load_group.setLayout(load_layout)

        # Group the elements for Analyse in a QGroupBox
        analyse_group = QGroupBox("Analyse")
        analyse_layout = QVBoxLayout()

        # Add Slice and Phase controls
        slice_phase_layout = QHBoxLayout()
        slice_phase_layout.addWidget(QLabel("Slice Location"))
        slice_phase_layout.addWidget(self.text_z)
        slice_phase_layout.addWidget(QLabel("Phase"))
        slice_phase_layout.addWidget(self.text_t)

        analyse_layout.addLayout(slice_phase_layout)
        analyse_layout.addWidget(self.lock_phase_button)

        # Add Load/Save ROI and ROI type controls
        roi_buttons_layout = QHBoxLayout()
        roi_buttons_layout.addWidget(self.load_rois_button)
        roi_buttons_layout.addWidget(self.save_rois_button)

        analyse_layout.addLayout(roi_buttons_layout)
        analyse_layout.addWidget(self.roi_button_ellipse)
        analyse_layout.addWidget(self.roi_button_poly)

        # Add FD Mode and Compute FD
        analyse_layout.addLayout(fd_mode_layout)
        analyse_layout.addWidget(self.comp_fd_button)

        analyse_group.setLayout(analyse_layout)

        # Main layout to add both groups
        main_control_layout = QVBoxLayout()
        main_control_layout.addWidget(load_group)
        main_control_layout.addWidget(analyse_group)
        
        # Image display QLabel
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(620, 620)  # Set your preferred size here
        self.image_label.setStyleSheet("background-color: white;")  # Optional, gives the label a background
        
        self.slider_hort = QSlider(Qt.Horizontal, self)
        self.slider_hort.valueChanged.connect(self.img_refresh)
        self.slider_hort.setEnabled(False)  # Initially disabled
        self.slider_hort.setFixedHeight(20)

        self.slider_vert = QSlider(Qt.Vertical, self)
        self.slider_vert.valueChanged.connect(self.img_refresh)
        self.slider_vert.setEnabled(False)  # Initially disabled
        self.slider_vert.setFixedWidth(20)
        
        # Horizontal layout for the image and the vertical slider
        img_slider_layout = QHBoxLayout()
        img_slider_layout.addWidget(self.image_label)  # Add the image label
        img_slider_layout.addWidget(self.slider_vert)  # Add the vertical slider

        # Vertical layout for the sliders and image
        image_and_sliders_layout = QVBoxLayout()
        image_and_sliders_layout.addLayout(img_slider_layout)  
        image_and_sliders_layout.addWidget(self.slider_hort)

        # Horizontal layout to place the image label and control panel side by side
        main_layout = QHBoxLayout()
        # main_layout.addWidget(self.image_label)  # Add the image display area with a stretch factor
        main_layout.addLayout(image_and_sliders_layout)
        main_layout.addLayout(main_control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def set_fd_mode(self):
        if self.lv_radio.isChecked():
            self.fd_mode = "LV"
        elif self.rv_radio.isChecked():
            self.fd_mode = "RV"
        print(f"fdMode: {self.fd_mode}")
        
    def load_folder(self):
        options = QFileDialog.Options()
        self.filepath = QFileDialog.getExistingDirectory(self, "Select Directory containing study DICOMs", "", options=options)
        if self.filepath:
            file_type, ok = QInputDialog.getItem(self, "Select File Type", "Choose the file type to analyze:", ["DICOM", "NIfTI"], 0, False)
            
            dicom_files = [f for f in os.listdir(self.filepath) if f.lower().endswith('.dcm')]
            nifti_files = [f for f in os.listdir(self.filepath) if f.lower().endswith(('.nii', '.nii.gz'))]
            
            folder_name = os.path.basename(self.filepath)
            self.listbox.clear()  
            self.listbox.addItem(folder_name)
            
            if file_type == "DICOM":
                if dicom_files:
                    QMessageBox.information(self, "DICOM Files Detected", f"{len(dicom_files)} DICOM files detected.")
                    self.load_study_button.setEnabled(True)
                    self.reanalyse_button.setEnabled(True)
                else:
                    QMessageBox.warning(self, "No DICOM Files", "No DICOM files detected in the selected directory.")
                    self.load_study_button.setEnabled(False)
                    self.reanalyse_button.setEnabled(False)

            elif file_type == "NIfTI":
                if nifti_files:
                    QMessageBox.information(self, "NIFTI Files Detected", f"{len(nifti_files)} NIFTI files detected.")
                    self.load_study_button.setEnabled(True)
                    self.reanalyse_button.setEnabled(True)
                else:
                    QMessageBox.warning(self, "No NIFTI Files", "No NIFTI files detected in the selected directory.")
                    self.load_study_button.setEnabled(False)
                    self.reanalyse_button.setEnabled(False)

        else:
            QMessageBox.warning(self, "No Valid Files Detected", "You did not select a file type.")
            self.load_study_button.setEnabled(False)
            self.reanalyse_button.setEnabled(False)
    
    def load_study(self):
        if self.filepath:
            dicom_files = [f for f in os.listdir(self.filepath) if f.lower().endswith('.dcm')]
            nifti_files = [f for f in os.listdir(self.filepath) if f.lower().endswith(('.nii', '.nii.gz'))]

            if dicom_files:
                self.load_dicom_study()
            elif nifti_files:
                self.load_nifti_study()
                
            if self.img_data is not None:
                self.slider_vert.setMaximum(self.img_data.shape[2] - 1)  # Set maximum slice index
                self.slider_hort.setMaximum(self.img_data.shape[3] - 1)  # Set maximum phase index
                
            self.load_rois_button.setEnabled(True)
            self.save_rois_button.setEnabled(True)
            self.comp_fd_button.setEnabled(True)
            self.lv_radio.setEnabled(True)
            self.rv_radio.setEnabled(True)
            
            self.img_refresh()

    def load_dicom_study(self):
        try:
            self.img_data, self.voxel_size = loadDicom(self.filepath, 0)
            if self.img_data is not None:
                self.enable_controls()
                QMessageBox.information(self, "DICOM Study Loaded", "DICOM study loaded successfully.")
            else:
                QMessageBox.warning(self, "Load Error", "Failed to load DICOM study.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading DICOMs: {str(e)}")

    def load_nifti_study(self):
        try:
            nii_file = os.path.join(self.filepath, 'sa_ED.nii.gz')
            nii_data = load_untouch_nii(nii_file)
            self.img_data = nii_data.get_fdata()
            self.voxel_size = nii_data.header.get_zooms()[:3]
            if self.img_data is not None:
                self.enable_controls()
                QMessageBox.information(self, "NIFTI Study Loaded", "NIFTI study loaded successfully.")
            else:
                QMessageBox.warning(self, "Load Error", "Failed to load NIFTI study.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading NIFTI: {str(e)}")

    def img_refresh(self):
        self.current_phase = self.slider_hort.value()
        self.current_slice = self.slider_vert.value()
        
        if self.img_data is not None:
            try:
                img_slice = self.img_data[:, :, self.current_slice, self.current_phase]
                img_slice_normalized = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX)
                img_slice_normalized = img_slice_normalized.astype(np.uint8)
                img_resized = cv2.resize(img_slice_normalized, (600, 600), interpolation=cv2.INTER_CUBIC)
                qimg = QImage(img_resized.data, img_resized.shape[1], img_resized.shape[0], img_resized.strides[0], QImage.Format_Grayscale8)
                # qimg = QImage(img_slice.data, img_slice.shape[1], img_slice.shape[0], img_slice.strides[0], QImage.Format_Grayscale8)
                self.image_label.setPixmap(QPixmap.fromImage(qimg))
                self.text_z.setText(f"{self.current_slice}")
                self.text_t.setText(f"{self.current_phase}")
            except IndexError:
                QMessageBox.warning(self, "Error", "Invalid slice index.")

    def enable_controls(self):
        self.slider_hort.setEnabled(True)
        self.slider_vert.setEnabled(True)
        
        if self.img_data is not None:
            max_slices = self.img_data.shape[2]
            max_phases = self.img_data.shape[3]

            # Set slider ranges
            self.slider_vert.setRange(0, max_slices - 1)
            self.slider_hort.setRange(0, max_phases - 1)

            # Set single steps
            self.slider_vert.setSingleStep(1)
            self.slider_hort.setSingleStep(1)
            
        self.flip_slice_loc_button.setEnabled(True)
        self.roi_button_ellipse.setEnabled(True)
        self.roi_button_poly.setEnabled(True)
        self.comp_fd_button.setEnabled(True)
        self.lock_phase_button.setEnabled(True)
        self.reanalyse_button.setEnabled(True)
        # Populate the listbox with example data for demo purposes
        # self.listbox.addItems([f"Study {i+1}" for i in range(5)])

    def reanalyse_all(self):
        folder_list = [f for f in os.listdir(self.filepath) if os.path.isdir(os.path.join(self.filepath, f))]
        input_file_type = "DICOM"  # This should be dynamically set based on actual input
        flag_error = 0
        folder_error = []
    
        for foldercnt, folder in enumerate(folder_list, start=1):
            folder_path = os.path.join(self.filepath, folder)
            print(f'Analysing Case {foldercnt}/{len(folder_list)}')
    
            if input_file_type == "DICOM":
                file_list = rdir(folder_path, '**/*.dcm')
                file_list = [f for f in file_list if not os.path.isdir(f)]
    
                if len(file_list) > 0:
                    if file_list[0].endswith('.DS_STORE'):
                        file_list.pop(0)
                        print('Removed .DS_STORE')
    
                if len(file_list) != 0:
                    img_data, voxel_size = loadDicom(folder_path)
                    max_z, max_t = img_data.shape[2], img_data.shape[3]
                else:
                    continue
                
            elif input_file_type == "NIFTI":
                nii_file = os.path.join(folder_path, 'sa_ED.nii.gz')
                if os.path.exists(nii_file):
                    nii_data = load_nii(nii_file)
                    img_data = nii_data.get_fdata()
                    voxel_size = nii_data.header.get_zooms()[:3]
                    max_z = img_data.shape[2]
                    max_t = 1
                else:
                    QMessageBox.critical(self, "Error", f'NIFTI not found in folder {folder}')
                    return
    
                self.lock_phase_button.setText('Unlock Phase')
                self.slider_hort.setEnabled(False)
    
            current_study_no = foldercnt
    
            # Set slider handles
            if max_t == 1:
                self.slider_hort.setEnabled(False)
                self.slider_hort.setMinimum(1)
                self.slider_hort.setMaximum(1)
                self.slider_hort.setValue(1)
            else:
                self.slider_hort.setMinimum(1)
                self.slider_hort.setMaximum(max_t)
                self.slider_hort.setValue(1)
                self.slider_hort.setSingleStep(1 / (max_t - 1) * 2)
    
            if max_z > 1:
                self.slider_vert.setMinimum(1)
                self.slider_vert.setMaximum(max_z)
                self.slider_vert.setValue(1)
                self.slider_vert.setSingleStep(1 / (max_z - 1) * 2)
            else:
                self.slider_vert.setMinimum(1)
                self.slider_vert.setMaximum(1)
                self.slider_vert.setValue(1)
                self.slider_vert.setSingleStep(1)
    
            # Set textboxes
            self.text_z.setText('1')
            self.text_t.setText('1')
            self.text_z.setText(f'Subject ID: {folder_list[current_study_no-1]}')
            self.set_roi_buttons_enabled(True)
    
            # Reset variables for the new analysis
            self.gray_img_data.clear()
            self.crop_img_data.clear()
            self.thres_img_data.clear()
            self.binary_mask_data.clear()
            self.binary_mask_full_data.clear()
            self.data_z.clear()
            self.data_t.clear()
            self.fd_data.clear()
            self.slices_flip_status = False
    
            # Store the current image and voxel size
            self.img_data = img_data
            self.voxel_size = voxel_size
            self.total_phases = max_t
            self.total_slices = max_z
            self.current_study_no = current_study_no
    
            # Check and load masks
            status_code = self.load_masks()
    
            if not status_code:
                flag_error += 1
                folder_error.append(folder_list[foldercnt - 1])
            else:
                self.compute_fd(modifier_called_func=True)
    
        if flag_error > 0:
            QMessageBox.warning(self, "Errors", f"Errors occurred in folders: {', '.join(folder_error)}")
        else:
            QMessageBox.information(self, "Re-Analysis Complete", f"Re-analysis complete for all cases.")

    def compute_fd(self, modifier_called_func=False):
        print("Debug: Starting Fractal Dimension computation.")
        sigma = 4
        epsilon = 3
        
        print(f"self.data.z: {self.data_z}")

        if not self.data_z:
            QMessageBox.warning(self, "Error", "ROI not set.")
            return

        sort_index = np.argsort(self.data_z)
        sorted_data_z = np.array(self.data_z)[sort_index]
        sorted_data_t = np.array(self.data_t)[sort_index]
        
        print(f"sort_index: {sort_index}")
        print(f"sorted_data_z: {sorted_data_z}")
        print(f"sorted_data_t: {sorted_data_t}")

        if not np.all(np.diff(sorted_data_z) < 3):
            reg_slices = " ".join(map(str, sorted_data_z))
            QMessageBox.warning(self, "Error", f"Slices not in order. Registered Slices: {reg_slices}")
            return

        thres_img_data = [None] * len(self.data_z)
        fd_data = np.zeros(len(self.data_z))
        
        for i, idx in enumerate(sort_index):
            try:
                key = (sorted_data_z[i], sorted_data_t[i])  # Access using the slice and phase
                crop_img = self.crop_img_data[key]
                binary_mask = self.binary_mask_data[key]
                
                if crop_img.shape != binary_mask.shape:
                    print(f"Shape mismatch: crop_img shape {crop_img.shape}, binary_mask shape {binary_mask.shape}")
                    continue 
                
                print(f"Processing slice {i}: crop_img shape {crop_img.shape}, binary_mask shape {binary_mask.shape}")

                # Perform FD computation
                fd_data[i], thres_img_data[i] = fracDimBatch(crop_img, binary_mask, sigma, epsilon)

            except KeyError:
                fd_data[i] = 0.0
                thres_img_data[i] = None
                print(f"Error: No ROI set for slice {sorted_data_z[i]}, phase {self.current_phase}")
            except Exception as e:
                fd_data[idx] = 0.0
                thres_img_data[idx] = None
                print(f"Error processing slice {idx}: {e}")

        self.export_fd_results(fd_data, sort_index, discard_mode=False)
        self.export_images(fd_data, thres_img_data, sort_index)

        if not modifier_called_func:
            QMessageBox.information(self, "Compute FD", "Fractal Dimension computation finished.")

        self.fd_data = fd_data
        self.thres_img_data = thres_img_data
        
    def export_fd_results(self, fd_data, sort_index, discard_mode):
        min_roi_slice = min(fd_data)
        print(f"Debug: Minimum ROI slice: {min_roi_slice}")
        
        fd_stats = fdStatistics(fd_data[sort_index], discard_mode)
        print(f"Debug: Exporting results for study: {self.current_study}")
        
        fd_output = [f"Study_{self.current_study}"]
        fd_output.append(f"{fd_stats['evalSlices']}, {fd_stats['usedSlices']}, LV")
        fd_output.append(f"{fd_stats['globalFD']:.9f}, {fd_stats['meanApicalFD']:.9f}, {fd_stats['maxApicalFD']:.9f}, {fd_stats['meanBasalFD']:.9f}, {fd_stats['maxBasalFD']:.9f}")

        # Example of exporting FD results
        output_file = os.path.join(self.filepath, 'FDSummary.csv')
        with open(output_file, 'a') as file:
            file.write(','.join(fd_output) + '\n')

    def export_images(self, fd_data, thres_img_data, sort_index):
        export_dir = os.path.join(self.filepath, 'ThresImg', f"Study_{self.current_study}")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        for idx in sort_index:
            if fd_data[idx] < 0.1:
                continue

            thres_img = thres_img_data[idx]
            crop_img = self.crop_img_data[idx]
            binary_mask = self.binary_mask_full_data[idx]

            # Saving images
            cv2.imwrite(os.path.join(export_dir, f"thresSlice{self.data_z[idx]}Phase{self.data_t[idx]}.png"), thres_img)
            cv2.imwrite(os.path.join(export_dir, f"binaryMaskSlice{self.data_z[idx]}Phase{self.data_t[idx]}.png"), binary_mask)
            cv2.imwrite(os.path.join(export_dir, f"cropSlice{self.data_z[idx]}Phase{self.data_t[idx]}.png"), crop_img)

            # Example of exporting additional images
            export_fig(thres_img, os.path.join(export_dir, f"bcPlotSlice{self.data_z[idx]}Phase{self.data_t[idx]}.png"))

    def on_roi_selected(self, roi_data):
        print("on_roi_selected")
        
        if roi_data['roi_type'] == 'ellipse':
            # Capture the ROI data once the ellipse has been selected and the window is closed
            print(f"Ellipse ROI selected with center: {roi_data['center']}, width: {roi_data['width']}, height: {roi_data['height']}")

            # Create an elliptical mask based on the ROI data
            center = roi_data['center']
            width = roi_data['width']
            height = roi_data['height']
            interp_img = roi_data['image']

            binary_mask_full = np.zeros_like(interp_img, dtype=np.uint8)
            rr, cc = ellipse(int(center[1]), int(center[0]), int(height / 2), int(width / 2), interp_img.shape)
            binary_mask_full[rr, cc] = 1

            # Compute the cropped image and mask
            x, y, w, h = cv2.boundingRect(binary_mask_full.astype(np.uint8))
            crop_img = interp_img[y:y+h, x:x+w]
            binary_mask_crop = binary_mask_full[y:y+h, x:x+w]

            # Save the mask for the current phase and slice
            key = (self.current_slice, self.current_phase)
            self.rois[key] = {
                'center': center,
                'width': width,
                'height': height,
                'crop_img': crop_img,
                'binary_mask_crop': binary_mask_crop,
                'binary_mask_full': binary_mask_full
            }
        
        elif roi_data['roi_type'] == 'polygon':
            # This is a polygon ROI
            print(f"Polygon ROI selected with vertices: {roi_data['vertices']}")

            # Save the polygon ROI data
            key = (self.current_slice, self.current_phase)
            self.rois[key] = {
                'vertices': roi_data['vertices'],
                'crop_img': roi_data['crop_img'],
                'binary_mask_crop': roi_data['binary_mask_crop'],
                'binary_mask_full': roi_data['binary_mask_full']
            }
        
        # Add the current slice to data_z if it's not already present
        # if self.current_slice not in self.data_z:
        #     self.data_z.append(self.current_slice)
        if self.current_slice + 1 not in self.data_z:  # Add +1 to convert to 1-based indexing
            self.data_z.append(self.current_slice + 1)
            print(f"Updated data_z: {self.data_z}")

        # If you're tracking phases, add the current phase to data_t as well
        if self.current_phase not in self.data_t:
            self.data_t.append(self.current_phase)
            print(f"Updated data_t: {self.data_t}")
            
        # Update crop_img_data and binary_mask_data with the current slice and phase
        self.crop_img_data[key] = roi_data.get('crop_img', None)
        self.binary_mask_data[key] = roi_data.get('binary_mask_crop', None)
        
        print(f"ROI saved for slice {self.current_slice}, phase {self.current_phase}")
        print(f"Updated data_z: {self.data_z}, data_t: {self.data_t}")

    def interp_image(self, source_img, voxel_size):
        # Image Grayscaling and Resizing
        target_img = cv2.normalize(source_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculate new dimensions based on voxel size
        dim_row, dim_col = target_img.shape
        interp_dim_row = int(dim_row * voxel_size[0] * 4)  # Assuming 1 mm to 4 pixels
        interp_dim_col = int(dim_col * voxel_size[1] * 4)

        # Resizing image using cubic interpolation
        target_img = cv2.resize(target_img, (interp_dim_col, interp_dim_row), interpolation=cv2.INTER_CUBIC)

        return target_img

    def toggle_lock_phase(self):
        if self.slider_hort.isEnabled():
            self.lock_phase_button.setText("Unlock Phase")
            self.slider_hort.setEnabled(False)
        else:
            self.lock_phase_button.setText("Lock Phase")
            self.slider_hort.setEnabled(True)
    
    def keyPressEvent(self, event):
        key = event.key()
        print(f"key: {key}")
        if key in [Qt.Key_Up, Qt.Key_W]:
            if self.slider_vert.value() < self.slider_vert.maximum():
                self.slider_vert.setValue(self.slider_vert.value() + 1)
                self.img_refresh()
        elif key in [Qt.Key_Down, Qt.Key_S]:
            if self.slider_vert.value() > self.slider_vert.minimum():
                self.slider_vert.setValue(self.slider_vert.value() - 1)
                self.img_refresh()
        elif key in [Qt.Key_Left, Qt.Key_A]:
            if self.slider_hort.isEnabled() and self.slider_hort.value() > self.slider_hort.minimum():
                self.slider_hort.setValue(self.slider_hort.value() - 1)
                self.img_refresh()
        elif key in [Qt.Key_Right, Qt.Key_D]:
            if self.slider_hort.isEnabled() and self.slider_hort.value() < self.slider_hort.maximum():
                self.slider_hort.setValue(self.slider_hort.value() + 1)
                self.img_refresh()
        elif key == Qt.Key_E and self.current_roi == 0:
            self.set_roi_ellipse()
        elif key == Qt.Key_R and self.current_roi == 0:
            self.set_roi_polygon()

    def load_masks(self):
        status_code = False
        current_study = f"Study_{self.current_study}"  # Example of how the study name could be stored

        try:
            # Attempt to load existing masks using your custom logic
            former_mask_present, reuse_mask_stack_data, return_code = pft_FindPreviousBinaryMasks(self.filepath, current_study)
            if return_code == 'OK':
                status_code = True
                self.apply_loaded_masks(reuse_mask_stack_data)
            else:
                print(f"Error: {return_code}")
        except Exception as e:
            print(f"Error loading masks: {e}")

        return status_code

    def apply_loaded_masks(self, reuse_mask_stack_data):
        print("apply_loaded_masks")
        # Clear existing data
        self.gray_img_data = {}
        self.crop_img_data = {}
        self.binary_mask_data = {}
        self.binary_mask_full_data = {}
        self.data_z = []
        self.data_t = []

        # Example logic to apply loaded masks
        max_z = self.img_data.shape[2]
        for current_z in range(max_z):
            current_t = 1  # Assuming time = 1 for ED slices
            if reuse_mask_stack_data[current_z].any():
                crop_img, binary_mask_crop, binary_mask_full = self.roi_select(self.img_data[:, :, current_z, current_t], self.voxel_size, 'reuse', reuse_mask_stack_data[current_z])
                self.crop_img_data.append(crop_img)
                self.binary_mask_data.append(binary_mask_crop)
                self.binary_mask_full_data.append(binary_mask_full)
                self.gray_img_data.append(self.img_data[:, :, current_z, current_t])
                self.data_z.append(current_z)
                self.data_t.append(current_t)

        self.img_refresh()

    def flip_slice_loc(self):
        if self.img_data is not None:
            self.img_data = np.flip(self.img_data, axis=2)
            self.slices_flip_status = not self.slices_flip_status
            self.img_refresh()

    def set_roi_ellipse(self):
        interp_img = self.img_data[:, :, self.current_slice, self.current_phase]  
        self.roi_window = RoiWindow(interp_img, roi_type='ellipse')  
        self.roi_window.roi_selected.connect(self.on_roi_selected)  
        self.roi_window.show()

    def set_roi_polygon(self):
        interp_img = self.img_data[:, :, self.current_slice, self.current_phase]  
        self.roi_window = RoiWindow(interp_img, roi_type='polygon')  # Specify the roi_type as 'polygon'
        self.roi_window.roi_selected.connect(self.on_roi_selected)  
        self.roi_window.show()
        
    def set_roi_reuse(self):
        interp_img = self.img_data[:, :, self.current_slice, self.current_phase]

        # Assuming you have a way to get the existing mask
        existing_mask = self.binary_mask_data.get((self.current_slice, self.current_phase), None)  

        if existing_mask is not None:
            self.binary_mask_full_data[(self.current_slice, self.current_phase)] = existing_mask
            self.img_refresh()
            reg_props_output = self.calculate_bounding_box(existing_mask)
            crop_img, binary_mask_crop = self.crop_image(interp_img, existing_mask, reg_props_output)
            print(f"Debug: Cropped image and binary mask for reuse case at slice {self.current_slice}, phase {self.current_phase}")
        else:
            QMessageBox.warning(self, "No Existing Mask", "No existing mask found for this slice and phase.")

    def calculate_bounding_box(self, binary_mask):
        # Calculate the bounding box around the region of interest (similar to MATLAB's regionprops)
        labeled_img, num_features = ndimage.label(binary_mask)
        regions = regionprops(labeled_img)

        if regions:
            return regions[0].bbox  # (min_row, min_col, max_row, max_col)
        else:
            return None

    def crop_image(self, gray_img, binary_mask, bounding_box):
        # Crops the image and binary mask according to the bounding box (similar to MATLAB's imcrop)
        if bounding_box:
            min_row, min_col, max_row, max_col = bounding_box
            cropped_img = gray_img[min_row:max_row, min_col:max_col] * binary_mask[min_row:max_row, min_col:max_col]
            binary_mask_crop = binary_mask[min_row:max_row, min_col:max_col]
            return cropped_img, binary_mask_crop
        else:
            return gray_img, binary_mask

    def set_roi(self, roi_type):
        print("set_roi")
        current_t = self.slider_hort.value()
        current_z = self.slider_vert.value()
        
        key = (current_z, current_t)
        
        if key in self.crop_img_data:  # Checking if the key exists in self.crop_img_data
            # If the key already exists, update the existing entry
            print(f"Updating existing ROI for slice {current_z}, phase {current_t}")
        else:
            print(f"Creating new ROI for slice {current_z}, phase {current_t}")
        
        # Just call roi_select, no need to unpack return values
        self.roi_select(self.img_data[:, :, current_z, current_t], self.voxel_size, roi_type)
        
        # if key in self.data_z:  # Checking if the key exists in self.data_z
        #     # If the key already exists, update the existing entry
        #     self.crop_img_data[key], self.binary_mask_data[key], self.binary_mask_full_data[key] = self.roi_select(
        #         self.img_data[:, :, current_z, current_t], self.voxel_size, roi_type)
        #     self.gray_img_data[key] = self.img_data[:, :, current_z, current_t]
        # else:
        #     # If the key does not exist, add a new entry
        #     self.crop_img_data[key], self.binary_mask_data[key], self.binary_mask_full_data[key] = self.roi_select(
        #         self.img_data[:, :, current_z, current_t], self.voxel_size, roi_type)
        #     self.gray_img_data[key] = self.img_data[:, :, current_z, current_t]
        #     # self.data_z.append(key)  # Assuming data_z is still meant to be a list
        #     self.rois[key] = (self.crop_img_data[key], self.binary_mask_data[key], self.binary_mask_full_data[key])

        #     self.img_refresh()

    def save_rois(self):
        # Check if any ROI is set (i.e., self.rois should not be empty)
        if not self.rois:
            QMessageBox.warning(self, "Save ROIs", "No ROI is set. Please set an ROI before saving.")
            return
        
        # Save ROIs to a file
        file_path = QFileDialog.getSaveFileName(self, 'Save ROIs', '', 'ROI Files (*.roi)')[0]
        if file_path:
            with open(file_path, 'wb') as f:
                pickle.dump(self.rois, f)
            QMessageBox.information(self, "Save ROIs", "ROIs saved successfully.")
    
    def load_rois(self):
        # Load ROIs from a file
        file_path = QFileDialog.getOpenFileName(self, 'Load ROIs', '', 'ROI Files (*.roi)')[0]
        if file_path:
            with open(file_path, 'rb') as f:
                self.rois = pickle.load(f)
            QMessageBox.information(self, "Load ROIs", "ROIs loaded successfully.")
            self.img_refresh()  # Refresh the image to show loaded ROIs

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FracAnalyse()
    window.show()
    sys.exit(app.exec_())