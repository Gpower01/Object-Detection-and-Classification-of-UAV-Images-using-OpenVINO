![Intel® Edge AI Foundation Course](https://img.shields.io/badge/Udacity-Intel%C2%AE%20Edge%20AI%20Foundation%20Course-blue?logo=Udacity&color=33bbff&style=flat)

[image1]: ./imgs/Out_UAV_image.jpg "Inference on Image"
[image2]: ./imgs/Inference_on_video_img.JPG "Inference on Videos"

# Object-Detection-and-Classification-using-OpenVINO

This Object detection and Classification application can be used to detect and classify the objects of the drone (UAV) image or video streams using Intel® OpenVINOOpenVINO™ toolkit.
This application is been developed as a part of project showcase challenge.

**Inference on Image**

![Inference on Image][image1]

**Inference on Video**

![Inference on Video][image2]

[Inference video link](https://www.dropbox.com/s/a206rgm8fgdat5v/out_uav_video.mp4?dl=0)

## Models

We used [person-vehicle-bike-detection-crossroad-1016](https://docs.openvinotoolkit.org/latest/_models_intel_person_vehicle_bike_detection_crossroad_1016_description_person_vehicle_bike_detection_crossroad_1016.html) to detect and classify the objects in a frame streamed by drone (UAV).

## Requirements

### Software

*  [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download?)
*  OR Udacity classroom workspace for the related course

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

For your convenience, you can utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
- [Mac](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)
- [Windows](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

## Run the Program 

### Command Line Arguments

 - `-i`  : The location of the input file or "CAM" for the camera streaming
 - `-d`  : The device name, if not "CPU"
 - `-ct` : The confidence threshold to use with the bounding boxes for face detection, default value is 0.5
 
### Command to run the program
 Example for input image or video 
 
 ```python app.py -i input/image/or/video/path -ct 0.4```
 
 Example for camera streaming
 
 ```python app.py -i CAM -ct 0.4```
 
## Future Works 
 - implement the Server Communications for this application
 - Detect more objects through this application
 - This application can be extended to identify the different objects in nature
 - Can be used in Traffic signal control through vehicles and pedestrians detection
 - Tuning the application can also be used for intruders detection in boarders

### Contributors
[God'spower Onyenanu](https://github.com/Gpower01)

[Govind Savara](https://github.com/govind-savara) 
