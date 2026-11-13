import sqlite3
import hashlib
from datetime import datetime

DB_NAME = "users.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            student_id TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS course_marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            course_code TEXT NOT NULL,
            course_name TEXT NOT NULL,
            credits REAL DEFAULT 2.0,
            ct1 REAL DEFAULT 0.0,
            ct2 REAL DEFAULT 0.0,
            ct3 REAL DEFAULT 0.0,
            presentation REAL DEFAULT 0.0,
            assignment REAL DEFAULT 0.0,
            midterm REAL DEFAULT 0.0,
            attendance REAL DEFAULT 80.0,
            updated_by TEXT,
            UNIQUE(student_id, course_code)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_read INTEGER DEFAULT 0
        )
    ''')

    demo_users = [
        ("Al-Amin Ahamad", "teacher1", hash_password("1234"), "Teacher", ""),
        ("Sumaiya Akter", "student1", hash_password("1234"), "Student", "STD-2026-01"),
        ("Tariqul Islam", "student2", hash_password("1234"), "Student", "STD-2026-02")
    ]
    for full_name, uname, pwd_hash, role, sid in demo_users:
        c.execute("SELECT id FROM users WHERE username = ?", (uname,))
        if not c.fetchone():
            c.execute(
                "INSERT INTO users (full_name, username, password_hash, role, student_id) VALUES (?, ?, ?, ?, ?)",
                (full_name, uname, pwd_hash, role, sid)
            )

    demo_marks = [
        ("STD-2026-01", "DS 421", "Machine Learning Driven Data Analysis I (DS Major)", 2.0, 13.0, 12.0, 14.0, 7.0, 5.0, 20.0, 88.0, "Al-Amin Ahamad"),
        ("STD-2026-01", "CSE 411", "Artificial Intelligence & Neural Networks", 3.0, 11.0, 10.5, 12.0, 6.5, 4.5, 18.0, 82.0, "Al-Amin Ahamad"),
        ("STD-2026-02", "DS 421", "Machine Learning Driven Data Analysis I (DS Major)", 2.0, 6.0, 7.5, 8.0, 4.0, 3.0, 11.0, 62.0, "Al-Amin Ahamad")
    ]
    for row in demo_marks:
        c.execute("SELECT id FROM course_marks WHERE student_id = ? AND course_code = ?", (row[0], row[1]))
        if not c.fetchone():
            c.execute('''
                INSERT INTO course_marks 
                (student_id, course_code, course_name, credits, ct1, ct2, ct3, presentation, assignment, midterm, attendance, updated_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)

    c.execute("SELECT COUNT(*) FROM notifications")
    if c.fetchone()[0] == 0:
        now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
        c.execute("INSERT INTO notifications (student_id, message, created_at, is_read) VALUES (?, ?, ?, 0)",
                  ("STD-2026-01", "New marks published for DS 421 by Al-Amin Ahamad.", now_str))
        c.execute("INSERT INTO notifications (student_id, message, created_at, is_read) VALUES (?, ?, ?, 0)",
                  ("STD-2026-02", "Early Warning: Attendance & Mid Term score in DS 421 require immediate attention.", now_str))

    conn.commit()
    conn.close()

def register_user(full_name, username, password, role, student_id=""):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_read INTEGER DEFAULT 0
            )
        ''')
        sid = student_id.strip().upper() if role == "Student" else ""
        uname = username.strip().lower()
        c.execute(
            "INSERT INTO users (full_name, username, password_hash, role, student_id) VALUES (?, ?, ?, ?, ?)",
            (full_name.strip(), uname, hash_password(password), role, sid)
        )
        if role == "Student" and sid:
            now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
            c.execute(
                "INSERT INTO notifications (student_id, message, created_at, is_read) VALUES (?, ?, ?, 0)",
                (sid, f"Welcome to Academic AI Portal, {full_name.strip()}! Your Student ID ({sid}) is active.", now_str)
            )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another username."
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT full_name, username, role, student_id FROM users WHERE username = ? AND password_hash = ?",
        (username.strip().lower(), hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return user

def get_all_students():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT student_id, full_name FROM users WHERE role = 'Student' AND student_id != '' ORDER BY student_id")
    rows = c.fetchall()
    conn.close()
    return rows

def get_courses_by_student_id(student_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM course_marks WHERE UPPER(student_id) = UPPER(?) ORDER BY course_code", (student_id.strip(),))
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_course_marks():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT student_id, course_code, course_name, credits, ct1, ct2, ct3, presentation, assignment, midterm, attendance, updated_by
        FROM course_marks ORDER BY student_id, course_code
    ''')
    rows = c.fetchall()
    conn.close()
    return rows

def upsert_course_marks(student_id, course_code, course_name, credits, ct1, ct2, ct3, presentation, assignment, midterm, attendance, updated_by):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    sid = student_id.strip().upper()
    ccode = course_code.strip().upper()
    c.execute("SELECT id FROM course_marks WHERE student_id = ? AND course_code = ?", (sid, ccode))
    exists = c.fetchone()
    if exists:
        c.execute('''
            UPDATE course_marks
            SET course_name=?, credits=?, ct1=?, ct2=?, ct3=?, presentation=?, assignment=?, midterm=?, attendance=?, updated_by=?
            WHERE student_id=? AND course_code=?
        ''', (course_name.strip(), credits, ct1, ct2, ct3, presentation, assignment, midterm, attendance, updated_by, sid, ccode))
        action_txt = "updated"
    else:
        c.execute('''
            INSERT INTO course_marks
            (student_id, course_code, course_name, credits, ct1, ct2, ct3, presentation, assignment, midterm, attendance, updated_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (sid, ccode, course_name.strip(), credits, ct1, ct2, ct3, presentation, assignment, midterm, attendance, updated_by))
        action_txt = "published"

    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    notif_msg = f"📢 Marks {action_txt} for {ccode} ({course_name.strip()}) by {updated_by}."
    c.execute("INSERT INTO notifications (student_id, message, created_at, is_read) VALUES (?, ?, ?, 0)", (sid, notif_msg, now_str))

    conn.commit()
    conn.close()

def add_notification(student_id, message):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    c.execute("INSERT INTO notifications (student_id, message, created_at, is_read) VALUES (?, ?, ?, 0)",
              (student_id.strip().upper(), message, now_str))
    conn.commit()
    conn.close()

def get_notifications(student_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, message, created_at, is_read FROM notifications WHERE UPPER(student_id) = UPPER(?) ORDER BY id DESC LIMIT 15",
              (student_id.strip(),))
    rows = c.fetchall()
    conn.close()
    return rows

def mark_notifications_read(student_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read = 1 WHERE UPPER(student_id) = UPPER(?)", (student_id.strip(),))
    conn.commit()
    conn.close()

def get_student_auto_summary(search_query):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    q = search_query.strip()
    c.execute(
        "SELECT student_id, full_name FROM users WHERE role = 'Student' AND (UPPER(student_id) = UPPER(?) OR LOWER(full_name) LIKE LOWER(?))",
        (q, f"%{q}%")
    )
    student = c.fetchone()
    if not student:
        conn.close()
        return None

    sid, fname = student
    c.execute("SELECT ct1, ct2, ct3, midterm, attendance FROM course_marks WHERE student_id = ?", (sid,))
    courses = c.fetchall()
    conn.close()

    if not courses:
        return {"student_id": sid, "full_name": fname, "has_marks": False, "attendance": 75.0, "ct_avg": 9.0, "midterm": 15.0, "course_count": 0}

    ct_avgs, mids, atts = [], [], []
    for ct1, ct2, ct3, mid, att in courses:
        valid_cts = [x for x in [ct1, ct2, ct3] if x > 0]
        ct_avgs.append(sum(valid_cts) / len(valid_cts) if valid_cts else 0.0)
        mids.append(mid)
        atts.append(att)

    return {"student_id": sid, "full_name": fname, "has_marks": True, "attendance": round(sum(atts) / len(atts), 1), "ct_avg": round(sum(ct_avgs) / len(ct_avgs), 2), "midterm": round(sum(mids) / len(mids), 2), "course_count": len(courses)}