import tkinter as tk
from tkinter import messagebox, simpledialog
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import Button, Frame
from PIL import Image, ImageTk
import face_recognition
import cv2
import os
import glob
import numpy as np
import shutil
import time

# ----------- Face Recognition Class -------------
class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        try:
            if not os.path.exists(images_path):
                raise FileNotFoundError(f"Directory not found: {images_path}")
            images_path = glob.glob(os.path.join(images_path, "*/*.*"))
            print(f"{len(images_path)} encoding images found.")

            self.known_face_encodings = []
            self.known_face_names = []

            for img_path in images_path:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                basename = os.path.basename(os.path.dirname(img_path))
                filename = os.path.splitext(os.path.basename(img_path))[0]

                encodings = face_recognition.face_encodings(rgb_img)
                if len(encodings) > 0:
                    img_encoding = encodings[0]
                    self.known_face_encodings.append(img_encoding)
                    self.known_face_names.append(basename.strip())
                else:
                    print(f"No faces found in {filename}")

            print(f"Encoding images loaded. Known faces: {len(self.known_face_names)}")
            return True
        except Exception as e:
            print(f"Error loading images: {str(e)}")
            return False

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

# ----------- Main Application Class -------------
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.current_theme = "darkly"
        self.style = Style(self.current_theme)
        self.sfr = SimpleFacerec()
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(self.main_frame, text="Face Recognition System", font=("Helvetica", 24, "bold"))
        title_label.pack(pady=30)

        username_frame = Frame(self.main_frame)
        username_frame.pack(pady=(10, 5))

        tk.Label(username_frame, text="Enter your username:", font=("Helvetica", 14)).pack(side=tk.LEFT, padx=5)

        self.username_entry = tk.Entry(username_frame, font=("Helvetica", 14), width=25)
        self.username_entry.pack(side=tk.LEFT, padx=5)

        button_frame = Frame(self.main_frame)
        button_frame.pack(pady=20)

        Button(button_frame, text="🔓 Login", bootstyle=SUCCESS, width=20, command=self.login).pack(side=tk.LEFT, padx=15)
        Button(button_frame, text="📝 Register", bootstyle=PRIMARY, width=20, command=self.register).pack(side=tk.LEFT, padx=15)

        Button(self.main_frame, text="🌓 Toggle Theme", bootstyle=INFO, width=20, command=self.toggle_theme).pack(pady=10)

        self.status_label = tk.Label(self.main_frame, text="", font=("Helvetica", 12))
        self.status_label.pack(pady=10)

        if not os.path.exists("faces"):
            os.makedirs("faces")

    def toggle_theme(self):
        self.current_theme = "flatly" if self.current_theme == "darkly" else "darkly"
        self.style.theme_use(self.current_theme)

    def initialize_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return None

        for i in range(15):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            time.sleep(0.2)
            self.status_label.config(text=f"Camera initializing... {i+1}/15")
            self.root.update()

        return cap

    def is_frame_black(self, frame, threshold=10):
        if frame is None:
            return True
        return np.mean(frame) < threshold

    def register(self):
        username = self.username_entry.get()
        if not username:
            messagebox.showwarning("Register", "Please enter a username first!")
            return

        user_folder = f"faces/{username}"
        if os.path.exists(user_folder):
            if not messagebox.askyesno("User Exists", f"User '{username}' already exists. Overwrite?"):
                return
            shutil.rmtree(user_folder)

        os.makedirs(user_folder)
        self.capture_images_for_registration(username)

    def capture_images_for_registration(self, username):
        cap = self.initialize_camera()
        if cap is None:
            return

        messagebox.showinfo("Register", "Camera is initializing... Please wait")

        warmup_frame = None
        for _ in range(10):
            ret, warmup_frame = cap.read()
            if ret and not self.is_frame_black(warmup_frame):
                cv2.imshow("Initializing Camera...", warmup_frame)
                cv2.waitKey(100)

        if warmup_frame is None or self.is_frame_black(warmup_frame):
            messagebox.showerror("Error", "Camera failed to initialize properly")
            cap.release()
            cv2.destroyAllWindows()
            return

        cv2.destroyAllWindows()
        messagebox.showinfo("Register", "Camera ready. It will now take 3 pictures...")

        count = 1
        attempts = 0
        max_attempts = 10

        while count <= 3 and attempts < max_attempts:
            ret, frame = cap.read()
            attempts += 1

            if not ret:
                messagebox.showerror("Error", "Failed to capture image.")
                break

            if self.is_frame_black(frame):
                messagebox.showwarning("Warning", "Camera still adjusting. Please wait...")
                time.sleep(0.5)
                continue

            cv2.imshow(f"Registering Face {count}/3 - Press 'q' to cancel", frame)
            key = cv2.waitKey(1000)
            if key == ord('q'):
                break

            img_path = f"faces/{username}/{username}_{count}.jpg"
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")
            count += 1
            attempts = 0

        cap.release()
        cv2.destroyAllWindows()

        if count > 3:
            messagebox.showinfo("Done", f"3 images saved for {username} ✅")
        else:
            messagebox.showinfo("Cancelled", "Registration incomplete or cancelled")

    def login(self):
        username = self.username_entry.get()
        if not username:
            messagebox.showwarning("Login", "Please enter a username first!")
            return

        self.status_label.config(text="Loading face data...")
        self.root.update()

        if not self.sfr.load_encoding_images("faces"):
            messagebox.showerror("Error", "No face encodings found. Please register first.")
            return

        self.status_label.config(text="Initializing camera (this may take a moment)...")
        self.root.update()

        cap = self.initialize_camera()
        if cap is None:
            return

        self.status_label.config(text="Camera warming up...")
        self.root.update()
        for i in range(20):
            cap.read()
            time.sleep(0.9)
            self.status_label.config(text=f"Camera adjusting... {i+1}/20")
            self.root.update()

        recognized_user = None
        start_time = time.time()
        timeout = 45
        self.status_label.config(text="Please look at the camera...")
        self.root.update()
        found_time = None

        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret or self.is_frame_black(frame):
                time.sleep(0.3)
                continue

            face_locations, face_names = self.sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                if name == username and recognized_user is None:
                    recognized_user = name
                    found_time = time.time()
                    self.status_label.config(text="User recognized! Showing camera for 5 more seconds...")
                    self.root.update()

            cv2.imshow("Login - Press 'q' to exit", frame)

            if recognized_user and (time.time() - found_time >= 5):
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status_label.config(text="")

        if recognized_user:
            messagebox.showinfo("Welcome", f"Hello, {recognized_user}!")
            self.open_user_dashboard(recognized_user)
        else:
            messagebox.showinfo("Not Recognized", "User not recognized. Please register or try again.")

    def open_user_dashboard(self, username):
        self.dashboard_window = tk.Toplevel(self.root)
        self.dashboard_window.title(f"Dashboard: {username}")
        self.dashboard_window.geometry("800x600")

        tk.Label(self.dashboard_window, text=f"Welcome, {username}", font=("Helvetica", 24, "bold")).pack(pady=30)

        Button(self.dashboard_window, text="👥 Recognize Team Members", bootstyle=INFO, command=self.recognize_team).pack(pady=20)
        self.display_dashboard_images(username)

    def display_dashboard_images(self, username):
        user_folder = f"faces/{username}"
        image_files = glob.glob(f"{user_folder}/*.jpg")

        if not image_files:
            return

        images_frame = Frame(self.dashboard_window)
        images_frame.pack(pady=20)

        for i, img_path in enumerate(image_files[:3]):
            img = Image.open(img_path)
            img.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(img)

            label = tk.Label(images_frame, image=photo)
            label.image = photo
            label.grid(row=0, column=i, padx=10)

    def recognize_team(self):
        if not self.sfr.load_encoding_images("faces"):
            messagebox.showerror("Error", "No team members registered yet.")
            return

        cap = self.initialize_camera()
        if cap is None:
            return

        recognized_members = set()
        messagebox.showinfo("Team Recognition", "Camera is on. Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret or self.is_frame_black(frame):
                continue

            face_locations, face_names = self.sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                if name != "Unknown":
                    recognized_members.add(name)

            cv2.imshow("Team Recognition - Press 'q' to exit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if recognized_members:
            messagebox.showinfo("Team Members Recognized", f"Recognized members:\n{', '.join(recognized_members)}")
        else:
            messagebox.showinfo("No Members Recognized", "No team members were recognized.")

# ----------- Main Execution -------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
