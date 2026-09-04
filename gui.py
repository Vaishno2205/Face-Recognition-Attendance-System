import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import subprocess
import sys


def run_python_script(script, *args):
    subprocess.Popen(
        [sys.executable, script, *args]
    )


def start_attendance():
    run_python_script("attendance_recognition.py")


def train_model():
    subprocess.run(
        [sys.executable, "train.py"]
    )

    messagebox.showinfo(
        "Success",
        "Model Trained Successfully!"
    )


def open_attendance():
    subprocess.Popen(
        ["notepad", "attendance\\Attendance.csv"]
    )


def export_excel():
    subprocess.run(
        [sys.executable, "export_excel.py"]
    )

    messagebox.showinfo(
        "Success",
        "Excel Report Generated Successfully!"
    )


def register_student():
    student_name = simpledialog.askstring(
        "Register Student",
        "Enter Student Name:"
    )

    if not student_name:
        return

    run_python_script(
        "capture_faces.py",
        student_name
    )


root = tk.Tk()

root.title("Face Recognition Attendance System")
root.geometry("550x500")

title = tk.Label(
    root,
    text="Face Recognition Attendance System",
    font=("Arial", 18, "bold")
)

title.pack(pady=25)


btn1 = tk.Button(
    root,
    text="Start Attendance",
    width=30,
    height=2,
    command=start_attendance
)

btn1.pack(pady=10)


btn2 = tk.Button(
    root,
    text="Train Model",
    width=30,
    height=2,
    command=train_model
)

btn2.pack(pady=10)


btn3 = tk.Button(
    root,
    text="View Attendance",
    width=30,
    height=2,
    command=open_attendance
)

btn3.pack(pady=10)


btn4 = tk.Button(
    root,
    text="Export Excel Report",
    width=30,
    height=2,
    command=export_excel
)

btn4.pack(pady=10)


btn5 = tk.Button(
    root,
    text="Register Student",
    width=30,
    height=2,
    command=register_student
)

btn5.pack(pady=10)


btn6 = tk.Button(
    root,
    text="Exit",
    width=30,
    height=2,
    command=root.destroy
)

btn6.pack(pady=10)


root.mainloop()