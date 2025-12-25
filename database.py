import sqlite3
from datetime import datetime

DB_NAME = "faces.db"

def connect():
    return sqlite3.connect(DB_NAME)


def init_db():
    with connect() as conn:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            academic_group TEXT NOT NULL,
            description TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_format TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            image_data BLOB NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS recognition_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            recognition_time TEXT NOT NULL,
            recognized_person_id INTEGER,
            similarity_score REAL,
            result TEXT NOT NULL,
            FOREIGN KEY (recognized_person_id) REFERENCES persons(person_id)
        )
        """)

        conn.commit()


def add_person(first_name, last_name, academic_group, description=None):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO persons VALUES (NULL, ?, ?, ?, ?)",
            (first_name, last_name, academic_group, description)
        )
        conn.commit()
        return cur.lastrowid


def add_photo(person_id, file_name, file_format, file_size, image_bytes):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO photos VALUES (NULL, ?, ?, ?, ?, ?)",
            (person_id, file_name, file_format, file_size, image_bytes)
        )
        conn.commit()


def get_all_persons():
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT person_id, first_name, last_name, academic_group FROM persons")
        return cur.fetchall()


def get_person_by_id(person_id):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM persons WHERE person_id=?", (person_id,))
        return cur.fetchone()


def update_person(person_id, first_name, last_name, academic_group, description):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE persons SET first_name=?, last_name=?, academic_group=?, description=? WHERE person_id=?",
            (first_name, last_name, academic_group, description, person_id)
        )
        conn.commit()


def delete_person(person_id):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM photos WHERE person_id=?", (person_id,))
        cur.execute("DELETE FROM recognition_logs WHERE recognized_person_id=?", (person_id,))
        cur.execute("DELETE FROM persons WHERE person_id=?", (person_id,))
        conn.commit()


def get_all_photos():
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT person_id, image_data FROM photos")
        return cur.fetchall()


def add_recognition_log(person_id, score, result):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO recognition_logs VALUES (NULL, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), person_id, score, result)
        )
        conn.commit()

def get_photos_by_person(person_id):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT photo_id, image_data FROM photos WHERE person_id=?",
            (person_id,)
        )
        return cur.fetchall()
