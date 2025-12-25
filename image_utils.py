import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return gray[y:y+h, x:x+w]


def image_to_bytes(image, fmt=".jpg"):
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        return None
    return buffer.tobytes()


def load_image_from_file(path):
    return cv2.imread(path)


def get_file_info(path):
    name = os.path.basename(path)
    fmt = os.path.splitext(name)[1].replace(".", "")
    size = os.path.getsize(path)
    return name, fmt, size