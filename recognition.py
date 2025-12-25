import cv2


def compare_faces(face1, face2):
    face1 = cv2.resize(face1, (200, 200))
    face2 = cv2.resize(face2, (200, 200))

    h1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([face2], [0], None, [256], [0, 256])

    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)

    score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return max(score, 0) * 100