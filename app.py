import argparse
import cv2
import numpy as np
from inference import Network

INPUT_STREAM = "./resource/test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
DET_MODEL = "/home/workspace/models/person-vehicle-bike-detection-crossroad-1016.xml"

CLASSES = ['Other', 'Vehicle', 'Pedestrian', 'Bike']
COLOR = [(255, 181, 51), (255, 51, 190), (66, 255, 51), (51, 134, 255)]


def get_args():
    """
    Gets the arguments from the command line.
    """
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    parser.add_argument("-d", help=d_desc, default='CPU')
    parser.add_argument("-ct", help=ct_desc, default=0.5)
    args = parser.parse_args()

    args.ct = float(args.ct)

    return args


def preprocessing(input_frame, height, width):
    """
    Given an input frame, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    """
    frame = cv2.resize(input_frame, (width, height))
    frame = frame.transpose((2, 0, 1))
    frame = frame.reshape(1, *frame.shape)

    return frame


def draw_boxes(frame, result, args, width, height):
    """
    Draw bounding boxes onto the frame.
    """
    for box in result[0][0]:  # output shape is 1x1xNx7
        conf = box[2]

        if conf >= args.ct:
            print(box)
            x_min = int(box[3] * width)
            y_min = int(box[4] * height)
            x_max = int(box[5] * width)
            y_max = int(box[6] * height)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLOR[int(box[1])], 2)

    return frame


def infer_on_data(args, model):
    """
    Handles input image, video or webcame and detect the faces.
    """

    # Handle image, video or webcam
    # create a flag for single image
    image_flag = False

    # Check if the input is a webcam
    if args.i == 'CAM':
        args.i = 0
    elif args.i.endswith('.jpg') or args.i.endswith('.bmp') or args.i.endswith('.png'):
        image_flag = True

    # Initialize the Inference Engine
    plugin = Network()

    # Load the network models into the IE
    plugin.load_model(model, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    if not image_flag:
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        out = cv2.VideoWriter('out_uav_video.mp4', 0x00000021, 30, (width, height))
    else:
        out = None

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame = preprocessing(frame, net_input_shape[2], net_input_shape[3])

        # Perform inference on the frame to detect face
        plugin.async_inference(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()

            out_frame = draw_boxes(frame, result, args, width, height)

            # Writing out the frame, depending on image or video
            if image_flag:
                cv2.imwrite('output_uav_img.jpg', out_frame)
            else:
                out.write(out_frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Closing the stream and any windows at the end of the application
    if not image_flag:
        out.release()

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    model = DET_MODEL
    infer_on_data(args, model)


if __name__ == "__main__":
    main()