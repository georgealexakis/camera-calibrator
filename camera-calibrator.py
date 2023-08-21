import numpy as np
import cv2
import yaml
import glob


class Calibrator:
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
    calibration_ready = False
    calibration = False
    display1 = False
    display2 = False
    display3 = False
    display4 = False
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

    def __init__(self):
        # Opencv optimizer
        cv2.useOptimized()
        # Init video capture
        print('Initializing camera...')
        self.cam = cv2.VideoCapture(0)
        print('Calibrator started -- Options:\n   esc - exit\n   c - points capturing and calibration\n   u - undistort image method 1\n   r - undistort image method 2\n   p - points pose estimation (axis)\n   d - points pose estimation (cube)\n   l - load parameters\n   o - offline process')
        # Start process
        self.calibrate()

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
            img, tuple(corners[0]),
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

    def calibrate(self):
        while True:
            # Read camera frames
            _, img = self.cam.read()
            self.img = img
            key = cv2.waitKey(3)
            # Key actions
            if key == 27:
                break
            elif key == 99:
                if self.calibration:
                    print('Calibration points capturing process is off.')
                    self.calibration = False
                else:
                    print('Calibration points capturing process is on.')
                    self.calibration = True
            elif key == 117:
                if self.display1:
                    print('Calibrated image using cv2.undistort() is off.')
                    cv2.destroyWindow('Calibrated image using cv2.undistort()')
                    self.display1 = False
                else:
                    print('Calibrated image using cv2.undistort() is on.')
                    self.display1 = True
            elif key == 114:
                if self.display2:
                    print('Calibrated image using remapping is off.')
                    cv2.destroyWindow('Calibrated image using remapping')
                    self.display2 = False
                else:
                    print('Calibrated image using remapping is on.')
                    self.display2 = True
            elif key == 112:
                if self.display3:
                    print('Point esstimation is off.')
                    self.display3 = False
                else:
                    print('Point esstimation is on.')
                    self.display3 = True
            elif key == 100:
                if self.display4:
                    print('Cube esstimation is off.')
                    self.display4 = False
                else:
                    print('Cube esstimation is on.')
                    self.display4 = True
            elif key == 108:
                self.load_params()
            elif key == 111:
                self.perform_offline_process()
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
                if ((self.calibration) and (key == 32)):
                    self.obj_points.append(self.objp)
                    self.img_points.append(corn)
                    print(f'Captured images: {len(self.img_points)}')
                    cv2.imwrite(f'./images/{len(self.img_points)}.jpg', img)
                    # Capture images and then stop
                    if (len(self.img_points) == self.num_images):
                        self.calibration_ready = True
                # Draw and display the corners
                img = cv2.drawChessboardCorners(
                    img, self.pattern_size, corn, ret)
            # Calibration process
            if self.calibration_ready:
                # Disable points capturing
                self.calibration = False
                cv2.destroyWindow('Calibration points')
                self.cam.release()
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
                        cv2.NORM_L2
                    )/len(imgpoints2)
                    mean_error += error
                print('Total mean re-projection error: ' +
                      str(mean_error/len(self.obj_points)))
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
                self.cam = cv2.VideoCapture(0)
                self.obj_points = []
                self.img_points = []
                self.calibration_ready = False
                print('Calibrator started -- Options:\n   esc - exit\n   c - points capturing and calibration\n   u - undistort image method 1\n   r - undistort image method 2\n   p - points pose estimation (axis)\n   d - points pose estimation (cube)\n   l - load parameters\n   o - offline process')
            # Display calibrated image using cv2.undistort()
            if self.display1 and (len(self.mtx) > 0):
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
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                # Display result
                dim = self.keep_aspect(dst, width=900)
                cv2.namedWindow(
                    'Calibrated image using cv2.undistort()',
                    cv2.WINDOW_NORMAL
                )
                cv2.resizeWindow(
                    'Calibrated image using cv2.undistort()',
                    dim[0],
                    dim[1]
                )
                cv2.imshow(
                    'Calibrated image using cv2.undistort()',
                    dst
                )
            # Display calibrated image using remapping
            if self.display2 and (len(self.mtx) > 0):
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                    self.mtx, self.dist, (w, h), 1, (w, h))
                mapx, mapy = cv2.initUndistortRectifyMap(
                    self.mtx, self.dist, None, newcameramtx, (w, h), 5)
                dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                # Display result
                dim = self.keep_aspect(dst, width=900)
                cv2.namedWindow(
                    'Calibrated image using remapping',
                    cv2.WINDOW_NORMAL
                )
                cv2.resizeWindow(
                    'Calibrated image using remapping',
                    dim[0],
                    dim[1]
                )
                cv2.imshow(
                    'Calibrated image using remapping',
                    dst
                )
            # Display pose estimation axis
            if self.display3 and (len(self.mtx) > 0) and (not self.display4):
                if ret == True:
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
                    axis = np.float32([
                        [3, 0, 0],
                        [0, 3, 0],
                        [0, 0, -3]
                    ]).reshape(-1, 3)
                    imgpts, _ = cv2.projectPoints(
                        axis,
                        rvec,
                        tvec,
                        self.mtx,
                        self.dist
                    )
                    # Draw lines
                    img = self.draw_axis(img, corn, imgpts)
            # Display pose estimation cube
            if self.display4 and (len(self.mtx) > 0) and (not self.display3):
                if ret == True:
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
                    axis = np.float32([
                        [0, 0, 0],
                        [0, 3, 0],
                        [3, 3, 0],
                        [3, 0, 0],
                        [0, 0, -3],
                        [0, 3, -3],
                        [3, 3, -3],
                        [3, 0, -3]
                    ])
                    imgpts, _ = cv2.projectPoints(
                        axis,
                        rvec,
                        tvec,
                        self.mtx,
                        self.dist
                    )
                    # Draw cube
                    img = self.draw_cube(img, corn, imgpts)
            dim = self.keep_aspect(img, width=900)
            cv2.namedWindow('Calibration points', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Calibration points', dim[0], dim[1])
            cv2.imshow('Calibration points', img)
        cv2.destroyAllWindows()

    def load_params(self):
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

    def perform_offline_process(self):
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


def main():
    Calibrator()


if __name__ == '__main__':
    main()
