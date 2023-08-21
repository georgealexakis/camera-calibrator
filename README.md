# Camera Calibrator

Camera Calibrator is a tiny Python script that performs geometric camera calibration. It uses the default web-camera of the computer. The basic usage is for experimental purposes and it can display the computed parameters. Numpy and OpenCV are used for the developement.

## Permissions

Change the permissions of the python file with the command below:

`$ chmod +x camera-calibrator.py`
`$ chmod +x camera-calibrator-gui.py`

## Menu Options

* 99  - 'c': Calibration points capturing process is on/off.
* 117 - 'u': Calibrated image using cv2.undistort() is on/off.
* 114 - 'r': Calibrated image using remapping is on/off.
* 112 - 'p': Points pose estimation (axis).
* 100 - 'd': Points pose estimation (cube).
* 108 - 'l': Load parameters.
* 111 - 'o': Offline process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.