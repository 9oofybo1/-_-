import cv2
import numpy as np
from image_utils import extract_face
from recognition import compare_faces
from database import get_all_photos, get_person_by_id, add_recognition_log

THRESHOLD = 50.0  # %

def recognize_from_camera():
    cap = cv2.VideoCapture(0)
    print("ESC — выход")

    photos = get_all_photos()

    if not photos:
        print("В базе нет фотографий")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        face = extract_face(frame)

        text = "NO FACE"
        best_score = 0
        best_person_id = None

        if face is not None:
            for person_id, img_blob in photos:
                db_face = cv2.imdecode(
                    np.frombuffer(img_blob, np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )
                score = compare_faces(face, db_face)

                if score > best_score:
                    best_score = score
                    best_person_id = person_id

            if best_score >= THRESHOLD:
                person = get_person_by_id(best_person_id)
                text = f"{person[1]} {person[2]} ({best_score:.1f}%)"
                add_recognition_log(best_person_id, best_score, "SUCCESS")
            else:
                text = f"НЕОПОЗНАННОЕ ЛИЦО ({best_score:.1f}%)"
                add_recognition_log(None, best_score, "FAILED")

        cv2.putText(
            frame, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
