from tkinter import *
from tkinter import ttk
import cv2
import yaml
import os
import shutil
from PIL import Image, ImageTk
import threading
import numpy as np
import glob
import time
import math


class CalibratorGui(Tk):
    # Termination criteria for point detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    # Depents on the internal corners of the chessboard
    pattern_size = (8, 5)
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...., (6,5,0)
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:pattern_size[0],
        0:pattern_size[1]
    ].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images
    # 3d point in real world space
    obj_points = []
    # 2d points in image plane
    img_points = []
    # Number of captured images for calibration
    num_images = 10
    # Camera matrix
    mtx = []
    # Camera distortion
    dist = []
    # Button triggers
    calibration_method = -1
    drawing_method = -1
    capture_trigger = False
    # Axis coordinates for cube of image display
    axis_c = np.float32([
        [0, 0, 0],
        [0, 3, 0],
        [3, 3, 0],
        [3, 0, 0],
        [0, 0, -3],
        [0, 3, -3],
        [3, 3, -3],
        [3, 0, -3]
    ])
    # Axis coordinates for axis of image display
    axis_a = np.float32([
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, -3]
    ]).reshape(-1, 3)
    # App status
    is_playing = False
    is_camera_ready = False
    was_playing = False
    buttons = [None] * 11
    buttons_status = ['disabled'] * 11
    # Click points
    image_clicks = []

    def __init__(self):
        super().__init__()
        # Creating main tkinter window/toplevel
        self.title('Camera Calibrator')
        self.geometry('1000x650')
        self.width = 800
        self.height = 600
        self.bind('<Configure>', self.resize)
        self.paused = False
        self.id = None
        # Buttons frame
        buttons_frame = ttk.LabelFrame(self, text='Controls')
        buttons_frame.pack(side=LEFT, fill=BOTH, padx=5, pady=5)
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=2)
        # Image frame
        image_frame = ttk.LabelFrame(self, text='Image')
        image_frame.pack(side=TOP, fill=BOTH, padx=5, pady=5)
        # Bottom frame
        btm_frame = ttk.LabelFrame(self, text='Stats')
        btm_frame.pack(side=BOTTOM, fill=BOTH, padx=5, pady=5)
        self.pixel_vars = [DoubleVar(), DoubleVar(), DoubleVar(), DoubleVar()]
        lp_1 = ttk.Label(
            btm_frame,
            textvariable=self.pixel_vars[0],
            width=20,
            anchor=E
        )
        lp_1.grid(row=0, column=0, sticky=E, pady=2)
        lp_2 = ttk.Label(
            btm_frame,
            textvariable=self.pixel_vars[1],
            width=20,
            anchor=E
        )
        lp_2.grid(row=0, column=1, sticky=E, pady=2)
        lp_3 = ttk.Label(
            btm_frame,
            textvariable=self.pixel_vars[2],
            width=20,
            anchor=E
        )
        lp_3.grid(row=0, column=2, sticky=E, pady=2)
        lp_4 = ttk.Label(
            btm_frame,
            textvariable=self.pixel_vars[3],
            width=20,
            anchor=E
        )
        lp_4.grid(row=0, column=3, sticky=E, pady=2)
        # This will create a label widget
        self.camera_status = StringVar()
        self.camera_status.set('waiting...')
        l3 = ttk.Label(buttons_frame, text='Camera Status:', width=20)
        l3.grid(row=0, column=0, sticky=NSEW, pady=2)
        l3_t = ttk.Label(
            buttons_frame,
            textvariable=self.camera_status,
            width=20,
            anchor=E
        )
        l3_t.grid(row=0, column=1, sticky=E, pady=2)
        self.num_of_images = IntVar()
        self.num_of_images.set(0)
        l1 = ttk.Label(buttons_frame, text='Captured Images:', width=20)
        l1.grid(row=1, column=0, sticky=NSEW, pady=2)
        l1_t = ttk.Label(
            buttons_frame,
            textvariable=self.num_of_images,
            width=20,
            anchor=E
        )
        self.check_images()
        l1_t.grid(row=1, column=1, sticky=E, pady=2)
        self.app_status = StringVar()
        self.app_status.set('...')
        l2 = ttk.Label(buttons_frame, text='Status:', width=20)
        l2.grid(row=2, column=0, sticky=NSEW, pady=2)
        l2_t = ttk.Label(
            buttons_frame,
            textvariable=self.app_status,
            width=20,
            anchor=E
        )
        l2_t.grid(row=2, column=1, sticky=E, pady=2)
        # Select dimensions
        l_dim = ttk.Label(buttons_frame, text='Select dimensions:')
        l_dim.grid(row=3, column=0, sticky=NSEW, pady=2)
        # Combobox creation
        self.current_dimension = StringVar()
        self.camera_dimensions = ttk.Combobox(
            buttons_frame, width=15, state='disabled', textvariable=self.current_dimension
        )
        self.camera_dimensions.bind(
            '<<ComboboxSelected>>', self.change_dimensions
        )
        # Adding combobox drop down list
        self.camera_dimensions['values'] = (
            '640x480',
            '848x480',
            '1280x720'
        )
        self.camera_dimensions.grid(row=3, column=1, sticky=E, pady=2)
        self.camera_dimensions.current(0)
        ttk.Separator(buttons_frame, orient='horizontal').grid(
            row=4,
            columnspan=2,
            sticky=NSEW,
            pady=5
        )
        # Button widget
        self.buttons[0] = ttk.Button(
            buttons_frame,
            text='Start Camera',
            command=lambda: self.start_camera(),
            state=self.buttons_status[0]
        )
        self.buttons[0].grid(row=5, column=0, sticky=NSEW, pady=2)
        self.buttons[1] = ttk.Button(
            buttons_frame,
            text='Stop Camera',
            command=lambda: self.stop_camera(),
            state=self.buttons_status[1]
        )
        self.buttons[1].grid(row=5, column=1, sticky=NSEW, pady=2)
        self.buttons[2] = ttk.Button(
            buttons_frame,
            text='Capture',
            command=lambda: self.capture_image(),
            state=self.buttons_status[2]
        )
        self.buttons[2].grid(row=6, columnspan=2, sticky=NSEW, pady=2)
        self.buttons[3] = ttk.Button(
            buttons_frame,
            text='Clear Images',
            command=lambda: self.clear_images(),
            state=self.buttons_status[2]
        )
        self.buttons[3].grid(row=7, columnspan=2, sticky=NSEW, pady=2)
        ttk.Separator(buttons_frame, orient='horizontal').grid(
            row=8,
            columnspan=2,
            sticky=NSEW,
            pady=5
        )
        self.buttons[4] = ttk.Button(
            buttons_frame,
            text='Compute Params',
            command=lambda: self.compute_params(),
            state=self.buttons_status[3]
        )
        self.buttons[4].grid(row=9, columnspan=2, sticky=NSEW, pady=2)
        self.buttons[5] = ttk.Button(
            buttons_frame,
            text='Load Params',
            command=lambda: self.load_params(),
            state=self.buttons_status[4]
        )
        self.buttons[5].grid(row=10, columnspan=2, sticky=NSEW, pady=2)
        self.buttons[6] = ttk.Button(
            buttons_frame,
            text='Offline Process',
            command=lambda: self.perform_offline_process(),
            state=self.buttons_status[5]
        )
        self.buttons[6].grid(row=11, columnspan=2, sticky=NSEW, pady=2)
        ttk.Separator(buttons_frame, orient='horizontal').grid(
            row=12,
            columnspan=2,
            sticky=NSEW,
            pady=5
        )
        self.buttons[7] = ttk.Button(
            buttons_frame,
            text='Calibrate M1',
            command=lambda: self.set_calibration(0),
            state=self.buttons_status[6]
        )
        self.buttons[7].grid(row=13, column=0, sticky=NSEW, pady=2)
        self.buttons[8] = ttk.Button(
            buttons_frame,
            text='Calibrate M2',
            command=lambda: self.set_calibration(1),
            state=self.buttons_status[7]
        )
        self.buttons[8].grid(row=13, column=1, sticky=NSEW, pady=2)
        self.buttons[9] = ttk.Button(
            buttons_frame,
            text='Add Cube',
            command=lambda: self.set_draw(0),
            state=self.buttons_status[8]
        )
        self.buttons[9].grid(row=14, column=0, sticky=NSEW, pady=2)
        self.buttons[10] = ttk.Button(
            buttons_frame,
            text='Add Axis',
            command=lambda: self.set_draw(1),
            state=self.buttons_status[9]
        )
        self.buttons[10].grid(row=14, column=1, sticky=NSEW, pady=2)
        ttk.Separator(buttons_frame, orient='horizontal').grid(
            row=15,
            columnspan=2,
            sticky=NSEW,
            pady=5
        )
        self.params_text = Text(buttons_frame, height=15, width=10)
        self.params_text.config(font=('Courier', 9))
        self.params_text.grid(
            row=16, columnspan=2, column=0, sticky=NSEW, pady=2
        )
        # Bind the app with Escape keyboard to quit app whenever pressed
        self.bind('<Escape>', lambda e: self.quit())
        # Create a label and display it on app
        self.label_widget = Label(image_frame)
        self.label_widget.pack(padx=10, pady=10)
        opencv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        captured_image = Image.fromarray(opencv_image)
        # Convert captured image to photoimage
        photo_image = ImageTk.PhotoImage(image=captured_image)
        # Displaying photoimage in the label
        self.label_widget.photo_image = photo_image
        # Configure image in the label
        self.label_widget.configure(image=photo_image)
        self.label_widget.bind('<Button-1>', self.get_pixel)
        self.label_widget.bind('<Motion>', self.hover_img)
        # Camera thread
        cam_thread = threading.Thread(target=lambda: self.init_camera())
        cam_thread.daemon = True
        cam_thread.start()

    def resize(self, state):
        if (self.id != None and self.paused == False):
            self.label_widget.after_cancel(self.id)
            self.paused = True
            self.paused_time = time.time()
            self.after(1000, self.controller)

    def controller(self):
        if (self.paused):
            self.open_camera()
            self.paused = False

    def init_camera(self):
        # Set message
        self.app_status.set('init camera...')
        # Define a video capture object
        self.vid = cv2.VideoCapture(0)
        # Declare the width and height in variables
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        c_dim = f'{int(self.width)}x{int(self.height)}'
        if (c_dim not in self.camera_dimensions['values']):
            self.camera_dimensions['values'] = (
                *self.camera_dimensions['values'], c_dim
            )
        self.current_dimension.set(c_dim)
        self.is_camera_ready = True
        self.camera_status.set('connected')
        self.camera_dimensions.config(state='readonly')
        for i in range(len(self.buttons)):
            self.buttons_status[i] = '!disabled'
            self.buttons[i].config(state=self.buttons_status[i])

    def get_pixel(self, values):
        self.pixel_vars[0].set('x: ' + str(values.x))
        self.pixel_vars[1].set('y: ' + str(values.y))
        if (len(self.image_clicks) < 2):
            self.params_text.delete(1.0, END)
            self.image_clicks.append([values.x, values.y])
            self.params_text.insert(END, self.image_clicks)
        else:
            self.image_clicks = []
            self.params_text.delete(1.0, END)
        self.measure_distance()

    def measure_distance(self):
        # Measure it first
        pixels_per_metric = 100 / 30
        if (len(self.image_clicks) == 2):
            dist_pixels = math.sqrt(
                (self.image_clicks[1][0] - self.image_clicks[0][0])**2 +
                (self.image_clicks[1][1] - self.image_clicks[0][1])**2
            )
            dist_mm = dist_pixels/pixels_per_metric
            self.params_text.delete(1.0, END)
            self.params_text.insert(
                END, 'Measured pixels\n'
            )
            self.params_text.insert(
                END, f'from : {self.image_clicks[0]}\n'
            )
            self.params_text.insert(
                END, f'to : {self.image_clicks[1]}\n'
            )
            self.params_text.insert(
                END, f'distance pixels : {round(dist_pixels, 3)}\n'
            )
            self.params_text.insert(
                END, f'distance mm : {round(dist_mm, 3)}\n'
            )

    def hover_img(self, values):
        self.pixel_vars[2].set('x: ' + str(values.x))
        self.pixel_vars[3].set('y: ' + str(values.y))

    def change_dimensions(self, option):
        c_thr = threading.Thread(target=lambda: self.change_it())
        c_thr.daemon = True
        c_thr.start()

    def change_it(self):
        self.camera_dimensions.config(state='disabled')
        for i in range(len(self.buttons)):
            self.buttons_status[i] = 'disabled'
            self.buttons[i].config(state=self.buttons_status[i])
        # Define a video capture object
        dim = self.current_dimension.get()
        x = dim.split('x')
        self.width, self.height = int(x[0]), int(x[1])
        self.stop_camera(self.was_playing)
        # Set the width and height
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, int(x[0]))
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, int(x[1]))
        print(int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.camera_dimensions.config(state='readonly')
        for i in range(len(self.buttons)):
            self.buttons_status[i] = '!disabled'
            self.buttons[i].config(state=self.buttons_status[i])
        # Start again
        if (self.was_playing):
            self.start_camera()

    def check_images(self):
        path = './images/'
        fileList = os.listdir(path)
        self.num_of_images.set(len(fileList))

    def clear_images(self):
        # Set message
        self.app_status.set('clear images')
        folder = './images/'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                self.check_images()
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def start_camera(self):
        if (not self.is_playing):
            if (self.vid.read()[0] == False):
                self.vid = cv2.VideoCapture(0)
            self.is_playing = True
            self.was_playing = True
            self.open_camera()

    def open_camera(self):
        if (self.is_camera_ready and self.is_playing):
            # Capture the video frame by frame
            _, frame = self.vid.read()
            # Main processing
            self.img = frame
            frame_calib = self.calibrate_method(frame)
            frame_chess, ret, corners = self.display_chessboard(
                frame_calib,
                self.capture_trigger
            )
            rst = self.draw_method(ret, frame_chess, corners)
            # Convert image from one color space to other
            opencv_image = cv2.cvtColor(rst, cv2.COLOR_BGR2RGBA)
            # Capture the latest frame and transform to image
            captured_image = Image.fromarray(opencv_image)
            # Convert captured image to photoimage
            photo_image = ImageTk.PhotoImage(image=captured_image)
            # Displaying photoimage in the label
            self.label_widget.photo_image = photo_image
            # Configure image in the label
            self.label_widget.configure(image=photo_image)
            # Repeat the same process after every 10 milliseconds
            self.id = self.label_widget.after(10, self.open_camera)

    def stop_camera(self, state=False):
        self.is_playing = False
        if (state):
            self.was_playing = True
        else:
            self.was_playing = False
        opencv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        captured_image = Image.fromarray(opencv_image)
        # Convert captured image to photoimage
        photo_image = ImageTk.PhotoImage(image=captured_image)
        # Displaying photoimage in the label
        self.label_widget.photo_image = photo_image
        # Configure image in the label
        self.label_widget.configure(image=photo_image)
        try:
            self.label_widget.after_cancel(self.id)
            self.vid.release()
        except:
            pass

    def capture_image(self):
        # Set message
        if (self.is_playing):
            self.app_status.set('capturing')
            self.capture_trigger = True

    def load_params(self):
        # Set message
        self.app_status.set('loading params')
        # Open the file and load the file
        with open('./calibration_params/calibration_matrix.yaml') as f:
            temp = yaml.safe_load_all(f)
            data_arr = next(temp)
            self.mtx = np.array(data_arr['camera_matrix'])
            self.dist = np.array(data_arr['dist_coeff'])
            # Display results
            print('------------Camera matrix------------')
            print(self.mtx)
            print('------------Distortion coefficients------------')
            print(self.dist)
            self.params_text.delete(1.0, END)
            self.params_text.insert(END, 'Camera matrix\n\n')
            self.params_text.insert(END, self.mtx)
            self.params_text.insert(END, '\n\nDistortion coefficients\n\n')
            self.params_text.insert(END, self.dist)

    def perform_offline_process(self):
        # Set message
        self.app_status.set('off. calibration')
        # Loop images of the folder
        images = glob.glob('./images/*.jpg')
        for filename in images:
            # Read image
            img = cv2.imread(filename)
            self.img = img
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.pattern_size,
                None,
                cv2.CALIB_CB_FAST_CHECK
            )
            # If found, add object points, image points (after refining them)
            if ret:
                # Finds the positions of internal corners of the chessboard
                corn = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    self.criteria
                )
                # Save object points and corners
                self.obj_points.append(self.objp)
                self.img_points.append(corn)
        self.compute_params()

    def compute_params(self):
        # Set message
        self.app_status.set('compute params')
        if (len(self.obj_points) > 0) and (len(self.img_points)):
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            # Disable points capturing
            self.mtx = []
            self.dist = []
            # Calibration process
            print('Calibration started. Please wait...')
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.obj_points,
                self.img_points,
                gray.shape[::-1],
                None,
                None
            )
            # Display results
            print('------------Camera matrix------------')
            print(self.mtx)
            print('------------Distortion coefficients------------')
            print(self.dist)
            print('------------Rotation vectors------------')
            print(self.rvecs)
            print('------------Transformation vectors------------')
            print(self.tvecs)
            print('------------------------------------')
            # Re-projection error calculation
            mean_error = 0
            for i in range(len(self.obj_points)):
                imgpoints2, _ = cv2.projectPoints(
                    self.obj_points[i],
                    self.rvecs[i],
                    self.tvecs[i],
                    self.mtx,
                    self.dist
                )
                error = cv2.norm(
                    self.img_points[i],
                    imgpoints2,
                    cv2.NORM_L2)/len(imgpoints2
                                     )
                mean_error += error
            print(
                'Total mean re-projection error: ' +
                str(mean_error/len(self.obj_points))
            )
            print('----------------------')
            print('Calibration finished...')
            # Transform the matrix and distortion coefficients to writable lists
            data = {
                'camera_matrix': np.asarray(self.mtx).tolist(),
                'dist_coeff': np.asarray(self.dist).tolist()
            }
            # Save it to a file
            with open('./calibration_params/calibration_matrix.yaml', 'w') as f:
                yaml.dump(data, f)
            # Init for next points capturing
            self.obj_points = []
            self.img_points = []

    def set_calibration(self, method):
        if (method == self.calibration_method):
            self.calibration_method = -1
            # Set message
            self.app_status.set('raw data')
        else:
            self.calibration_method = method
            # Set message
            self.app_status.set(f'calibrating {method}')

    def set_draw(self, method):
        if (method == self.drawing_method):
            self.drawing_method = -1
        else:
            self.drawing_method = method
            # Set message
            self.app_status.set(f'draw 3d {method}')

    def calibrate_method(self, img):
        # Return calibrated image using cv2.undistort()
        if ((self.calibration_method == 0) and (len(self.mtx) > 0) and (len(self.dist) > 0)):
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx,
                self.dist,
                (w, h),
                1,
                (w, h)
            )
            dst = cv2.undistort(
                img,
                self.mtx,
                self.dist,
                None,
                newcameramtx
            )
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            return dst
        # Return calibrated image using remapping
        elif ((self.calibration_method == 1) and (len(self.mtx) > 0) and (len(self.dist) > 0)):
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx,
                self.dist,
                (w, h),
                1,
                (w, h)
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                self.mtx,
                self.dist,
                None,
                newcameramtx,
                (w, h),
                5
            )
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            return dst
        else:
            return img

    def draw_method(self, ret, img, corners):
        if (self.drawing_method == 0):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Display pose estimation (Cube)
            if ((ret == True) and (len(self.mtx) > 0) and (len(self.dist))):
                # Finds the positions of internal corners of the chessboard
                corn = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    self.criteria
                )
                # Find the rotation and translation vectors
                _, rvec, tvec, _ = cv2.solvePnPRansac(
                    self.objp,
                    corn,
                    self.mtx,
                    self.dist
                )
                # Project 3D points to image plane
                imgpts, _ = cv2.projectPoints(
                    self.axis_c,
                    rvec,
                    tvec,
                    self.mtx,
                    self.dist
                )
                # Draw lines
                img = self.draw_cube(img, corn, imgpts)
        elif (self.drawing_method == 1):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Display pose estimation (Axis)
            if ((ret == True) and (len(self.mtx) > 0) and (len(self.dist))):
                # Finds the positions of internal corners of the chessboard
                corn = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    self.criteria
                )
                # Find the rotation and translation vectors
                _, rvec, tvec, _ = cv2.solvePnPRansac(
                    self.objp,
                    corn,
                    self.mtx,
                    self.dist
                )
                # Project 3D points to image plane
                imgpts, _ = cv2.projectPoints(
                    self.axis_a,
                    rvec,
                    tvec,
                    self.mtx,
                    self.dist
                )
                # Draw lines
                img = self.draw_axis(img, corn, imgpts)
        return img

    def display_chessboard(self, img, capture=False):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            None,
            cv2.CALIB_CB_FAST_CHECK
        )
        # If found, add object points, image points (after refining them)
        if ret:
            # Finds the positions of internal corners of the chessboard
            corn = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                self.criteria
            )
            # Capture images and image points
            if (capture):
                self.obj_points.append(self.objp)
                self.img_points.append(corn)
                print(f'Captured images: {len(self.img_points)}')
                cv2.imwrite(f'./images/{len(self.img_points)}.jpg', img)
                self.capture_trigger = False
                self.check_images()
            # Draw and display the corners
            img = cv2.drawChessboardCorners(
                img,
                self.pattern_size,
                corn,
                ret
            )
        else:
            self.capture_trigger = False
        return img, ret, corners

    def keep_aspect(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """ Resize image by keeping aspect ratio. Just give desired width or desired height to be resized.
         If both, original image will be returned. """
        # Initialize the dimensions of the image to be resized and grab the image size
        dim = None
        (h, w) = image.shape[:2]
        # If both the width and height are None, then return the original image
        if width is None and height is None:
            return image
        # Check to see if the width is None
        if width is None:
            # Calculate the ratio of the height and construct the Dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # Otherwise, the height is None
        else:
            # Calculate the ratio of the width and construct the Dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        # Return the resized image
        return dim

    def draw_axis(self, img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)
        corners = np.int32(corners).reshape(-1, 2)
        img = cv2.line(
            img,
            tuple(corners[0]),
            tuple(imgpts[0]),
            (255, 0, 0),
            5
        )
        img = cv2.line(
            img,
            tuple(corners[0]),
            tuple(imgpts[1]),
            (0, 255, 0),
            5
        )
        img = cv2.line(
            img,
            tuple(corners[0]),
            tuple(imgpts[2]),
            (0, 0, 255),
            5
        )
        return img

    def draw_cube(self, img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # Draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
        # Draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
        # Draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
        return img


def main():
    app = CalibratorGui()
    app.mainloop()


if __name__ == '__main__':
    main()
