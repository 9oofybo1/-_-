from database import *
from image_utils import *
from camera import capture_faces_from_camera


def add_person_from_files():
    fn = input("Имя: ")
    ln = input("Фамилия: ")
    grp = input("Группа: ")
    desc = input("Описание: ") or None

    pid = add_person(fn, ln, grp, desc)
    paths = input("Пути к фото через запятую: ").split(",")

    for p in paths:
        img = load_image_from_file(p.strip())
        if img is None:
            continue
        face = extract_face(img)
        if face is None:
            continue
        img_bytes = image_to_bytes(face)
        name, fmt, size = get_file_info(p.strip())
        add_photo(pid, name, fmt, size, img_bytes)


def add_person_from_camera():
    fn = input("Имя: ")
    ln = input("Фамилия: ")
    grp = input("Группа: ")
    desc = input("Описание: ") or None

    pid = add_person(fn, ln, grp, desc)
    count = int(input("Количество снимков: "))

    faces = capture_faces_from_camera(count)
    for i, face in enumerate(faces):
        img_bytes = image_to_bytes(face)
        add_photo(pid, f"camera_{i+1}", "jpg", len(img_bytes), img_bytes)


def edit_person():
    for p in get_all_persons():
        print(p)
    pid = int(input("ID: "))
    p = get_person_by_id(pid)
    if not p:
        return
    fn = input(f"Имя [{p[1]}]: ") or p[1]
    ln = input(f"Фамилия [{p[2]}]: ") or p[2]
    grp = input(f"Группа [{p[3]}]: ") or p[3]
    desc = input(f"Описание [{p[4]}]: ") or p[4]
    update_person(pid, fn, ln, grp, desc)


def remove_person():
    for p in get_all_persons():
        print(p)
    pid = int(input("ID для удаления: "))
    delete_person(pid)