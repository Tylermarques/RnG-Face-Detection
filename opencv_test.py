import time
from SNCRZ25N.camera import CameraThreaded
import cv2 as cv
import numpy as np
from urllib.request import Request, urlopen
import urllib.error
import base64
import json

from requests.auth import HTTPBasicAuth


camera_is_flipped = True

cascPath = "./haar_cascades/haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)
point_1 = (220, 180)
point_2 = (420, 300)


def center_camera_on_coordinates_relative(camera_obj, x, y, w, h, flipped=False):
    # image is 640 x 480
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    # TODO Handle the flip, if that is in place.
    # if top center plus or minus 40 pixels, return
    # if left center plus or minus 30 pixels, return
    if (point_1[0] <= center_x <= point_2[0]) and (point_1[1] <= center_y <= point_2[1]):
        # If the camera doesn't need to move, the face is in the center, and the face is not more than 30% of the frame,
        #   then zoom in.

        if not (w > 192) or not (h > 144):
            camera_obj.zoom_in(1)

    if center_x < point_1[0]:
        print(f"center_x < {point_1[0]} -- {center_x < point_1[0]}")
        camera_obj.pan_ccw(1)

    if center_x > point_2[0]:
        print(f"center_x > {point_2[0]} -- {center_x > point_2[0]}")
        camera_obj.pan_cw(1)

    if center_y < point_1[1]:
        print(f"center_y < {point_1[1]} -- {center_y < point_1[1]}")
        camera_obj.tilt_negative(1)

    if center_y > point_2[1]:
        print(f"center_y > {point_2[1]} -- {center_y > point_2[1]}")
        camera_obj.tilt_positive(1)


def center_camera_absolute_move(camera_obj: CameraThreaded, x, y, w, h):
    """
    To do absolute moves, we need to know how much to move. The issue is, how far is the face? We're going to have to
        estimate the average size of a face and just go from there. If we can do the math on how many degrees to move
        based on the current zoom amount, and the position of the face in the frame.
    :param camera_obj:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """

    # I had planned on needing some sort of factor between the x,y of the face center on the screen and the position
    #   number that needs to be sent to the camera, but it appears someone has already thought of that. The scale
    #   appears to be perfect. One pixel on the image corresponds to one unit of movement for the camera.

    pan_scale_factor = 1
    tilt_scale_factor = 1

    center_point = {
        "x": x + (w / 2),
        "y": y + (h / 2)
    }

    # How many pixels off is the camera from where we want it?
    x_delta = int((640 / 2) - center_point["x"] * pan_scale_factor)
    y_delta = int((480 / 2) - center_point["y"] * tilt_scale_factor)

    updated_pan = camera_obj.current_pan - x_delta

    if updated_pan < 0 or updated_pan > 65535:
        # If it's less than zero or greater than 65535, wrap around.
        updated_pan = (65535 - camera_obj.current_pan) - x_delta

    camera_obj.current_pan = updated_pan

    # 65535 is the horizon, anything lower is "down" when camera is in mounted position
    camera_obj.current_tilt = camera_obj.current_tilt - y_delta

    # print(f"Updated_Pan: {updated_pan}")
    camera_obj.send_position_update()
    return


def connect_to_cam(video_url, username, password):
    n = 1
    request = Request(video_url)
    auth_string = "%s:%s" % (username, password)
    b64auth = base64.standard_b64encode(auth_string.encode('ascii'))
    request.add_header("Authorization", f"Basic {b64auth.decode('ascii')}")
    while True:
        # Try to connect the camera.
        try:
            # use the opener to fetch a URL
            stream = urlopen(request)
            break
        except urllib.error.URLError:
            print(f"Can't connect to Camera. Sleeping for {n} seconds")
            n += 1
            time.sleep(n)
    return stream


def connect_to_cam_with_fail(video_url, username, password):
    request = Request(video_url)
    auth_string = "%s:%s" % (username, password)
    b64auth = base64.standard_b64encode(auth_string.encode('ascii'))
    request.add_header("Authorization", f"Basic {b64auth.decode('ascii')}")
    return urlopen(request)


def relative_move_face_detection(cam, user, password):
    iterator = 0
    video_url = "http://" + cam.ip + '/image'
    stream_bytes = b''
    stream = connect_to_cam(video_url, user, password)
    while True:
        stream_bytes += stream.read(16384)
        a = stream_bytes.find(b'\xff\xd8')
        b = stream_bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = stream_bytes[a:b + 2]
            stream_bytes = stream_bytes[b + 2:]
            i = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.COLOR_BGR2GRAY)
            # i = cv.cvtColor(i, cv.COLOR_BGR2GRAY)

            if camera_is_flipped:
                i = cv.flip(i, 0)
            faces = faceCascade.detectMultiScale(
                i,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv.CASCADE_SCALE_IMAGE
            )
            # cv.rectangle(i, (point_1[0], point_1[1]), (point_2[0], point_2[1]), (0, 255, 0), 2)
            for (x, y, w, h) in faces:
                # TODO Center on Face
                # TODO handle case where face is getting bigger (person is getting closer to the lens) and zoom out.
                # TODO Set a "scan" area where the camera will pan to look for people
                # TODO Make it better at looking for faces that are actual faces, as well as after it has identified
                #  someone, move on to the next face.

                if iterator > 4:
                    # Only activate once per second (approx 15 fps)
                    center_camera_absolute_move(cam, x, y, w, h)
                    iterator = 0
                cv.rectangle(i, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # White is Y
                cv.circle(i, (400, int(y + (h / 2))), radius=4, color=(255, 0, 255), thickness=4)
                # Black is X
                cv.circle(i, (int((x + (w / 2))), 400), radius=4, color=(255, 0, 255), thickness=4)
                # Print the center coordinates of the face
                print(int((x + (w / 2))), int(y + (h / 2)))
            print(f"FACES: {len(faces)} ITER: {iterator}")

            if len(faces) == 0 and iterator > 5:
                cam.zoom_out(1)
                iterator = 0
            cv.imshow('Sony SNC IPELA', i)
            if cv.waitKey(1) == 27:
                exit(0)
            iterator += 1


def flip_display_image(cam, auth):
    video_url = "http://" + cam.ip + '/image'
    stream_bytes = b''
    stream = connect_to_cam_with_fail(video_url, auth["username"], auth["password"])
    while True:
        stream_bytes += stream.read(16384)
        a = stream_bytes.find(b'\xff\xd8')
        b = stream_bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = stream_bytes[a:b + 2]
            stream_bytes = stream_bytes[b + 2:]
            i = cv.imdecode(np.fromstring(jpg, dtype=np.uint8), cv.COLOR_BGR2GRAY)
            if camera_is_flipped:
                i = cv.flip(i, 0)
            cv.circle(i, (0, 0), radius=4, color=(0, 0, 0), thickness=4)
            for x in range(32, 640, 32):
                for y in range(32, 480, 32):
                    cv.circle(i, (x, y), radius=4, color=(255, 0, 255), thickness=4)

            # i = cv.cvtColor(i, cv.COLOR_BGR2GRAY)

            cv.imshow('Sony SNC IPELA', i)
            if cv.waitKey(1) == 27:
                exit(0)


if __name__ == '__main__':
    with open('./config.json', 'r') as config_file:
        config = json.load(config_file)

    camera = CameraThreaded(config["ip"], config["user"], config["password"])
    try:
        relative_move_face_detection(camera, config["user"], config["password"])
        # flip_display_image(camera, auth)
    except KeyboardInterrupt:
        print("EXITING...", end="")
        camera.absolute_pan_tilt(0, 0)
        print("DONE")
