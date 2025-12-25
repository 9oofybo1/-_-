import cv2
from image_utils import extract_face


def capture_faces_from_camera(count):
    cap = cv2.VideoCapture(0)
    faces = []
    print("SPACE — снимок | ESC — выход")

    while len(faces) < count:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break
        if key == 32:
            face = extract_face(frame)
            if face is not None:
                faces.append(face)
                print(f"Снимок {len(faces)}/{count}")
            else:
                print("Лицо не найдено")

    cap.release()
    cv2.destroyAllWindows()
    return faces