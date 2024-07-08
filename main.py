import numpy as np
import cv2
import os
from keras.models import load_model
from collections import deque
import tensorflow as tf

def detect_violence_from_camera():
    print("Loading model ...")
    model = load_model('modelnew.h5')  # Ensure the model path is correct
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(0)  # Use the default camera
    (W, H) = (None, None)

    while True:
        # read the next frame from the camera
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then there was an issue with the camera
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # clone the output frame, then convert it from BGR to RGB
        output = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        # make predictions on the frame and then update the predictions queue
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        # perform prediction averaging over the current history of previous predictions
        results = np.array(Q).mean(axis=0)
        label = (results > 0.50)[0]

        text_color = (0, 255, 0)  # default: green

        if label:  # Violence prob
            text_color = (0, 0, 255)  # red

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # release the camera and close any open windows
    print("[INFO] cleaning up...")
    vs.release()
    cv2.destroyAllWindows()


def print_results(video, limit=None):
    if not os.path.exists('output'):
        os.mkdir('output')

    print("Loading model ...")
    model = load_model('modelnew.h5')
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(video)
    writer = None
    (W, H) = (None, None)

    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # clone the output frame, then convert it from BGR to RGB
        output = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        # make predictions on the frame and then update the predictions queue
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        # perform prediction averaging over the current history of previous predictions
        results = np.array(Q).mean(axis=0)
        label = (results > 0.50)[0]

        text_color = (0, 255, 0)  # default: green

        if label:  # Violence prob
            text_color = (0, 0, 255)  # red

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output/v_output.avi", fourcc, 30, (W, H), True)

        # write the output frame to disk
        writer.write(output)

        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            cv2.imwrite('saved_frame.jpg', output)
            print("Frame saved as 'saved_frame.jpg'")
        elif key == ord("q"):
            break

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()

print_results("test.mp4")