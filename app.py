import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import model_runtime

class FallDetection:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("900x880")
        self.root.title("Fall Detection")
        self.root.config(bg="#1f1f1f")
        self.ctx = model_runtime.init_model()   # one-time model load

        self.current_file_path = None
        self.flash_job = None
        self.fall_counter = 0
        self.playback_job = None
        self.video_fps = 30
        
        self.setup_navbar()
        self.setup_preview_area()
        self.setup_thumbnail_area()
        self.setup_welcome_section()
        self.setup_file_label()
        self.setup_analyse_button()
        self.setup_result_area()

        self.root.mainloop()

    # ------------------------ UI SETUP METHODS ------------------------ #
    def setup_navbar(self):
        navbar = tk.Frame(self.root, bg="#000000", height=120)
        navbar.pack(fill="x")

        title_frame = tk.Frame(navbar, bg="#000000")
        title_frame.pack(side="left", padx=20)

        fall_label = tk.Label(title_frame, text="fall", font=("Arial Bold", 25), bg="#000000", fg="#fe3330")
        fall_label.pack(side="left")

        detection_label = tk.Label(title_frame, text="Detection", font=("Arial Bold", 24), bg="#000000", fg="#ffffff")
        detection_label.pack(side="left")

        right_frame = tk.Frame(navbar, bg="#000000")
        right_frame.pack(side="right", padx=(20, 0))

        directory_path = os.path.dirname(__file__)
        logo_path = os.path.join(directory_path, 'img/segfault_logo.png')
        refresh_path = os.path.join(directory_path, 'img/refresh.png')

        logo_img = Image.open(logo_path).resize((130, 75), Image.Resampling.LANCZOS)
        self.logo_tk = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(right_frame, image=self.logo_tk, bg="#000000")
        logo_label.pack(side="right", padx=(10, 0))

        refresh_img = Image.open(refresh_path).resize((20, 20), Image.Resampling.LANCZOS)
        self.refresh_tk = ImageTk.PhotoImage(refresh_img)

        self.refresh_btn = tk.Button(
            right_frame, image=self.refresh_tk,
            bg="#000000", bd=0, activebackground="#1a1a1a",
            cursor="hand2", command=self.refresh
        )
        self.refresh_btn.pack(side="right", padx=(0, 10))

    def setup_preview_area(self):
        self.detected_label = tk.Label(self.root, text="", font=("fixedsys", 22), bg="#1f1f1f", fg="#fe3330")
        self.detected_label.pack(side="top", pady=(20, 0))

        self.preview_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.preview_frame.pack(pady=10)

        self.preview_label = tk.Label(self.preview_frame, text="📂 .mp4 .avi .mov",
                                      font=("Courier New", 12), relief="solid",
                                      width=65, height=23, bg="#141414", fg="#cccaca")
        self.preview_label.pack(padx=10, pady=10)
        self.preview_label.bind("<Button-1>", lambda e: self.open_file_locator())
        self.preview_label.bind("<Enter>", lambda e: self.preview_label.config(bg="#1a1a1a", cursor="hand2"))
        self.preview_label.bind("<Leave>", lambda e: self.preview_label.config(bg="#141414", cursor=""))

    def setup_thumbnail_area(self):
        self.thumbnail_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.thumbnail_label = tk.Label(self.thumbnail_frame, text="", bg="#141414", relief="solid", bd=1)

    def setup_welcome_section(self):
        self.welcome_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.welcome_frame.pack(pady=(5, 15))

        self.welcome_label = tk.Label(self.welcome_frame, text="Upload a Video", font=("Arial bold", 26), bg="#1f1f1f", fg="#cccaca")
        self.welcome_label.pack()

        self.description_label = tk.Label(self.welcome_frame,
                                          text="\nPlease select your desired video to analyse for the system\nto detect and identify a fall that occurs.",
                                          font=("Helvetica", 10),
                                          bg="#1f1f1f", fg="#e8e8e8")
        self.description_label.pack()

        self.upload_btn = tk.Button(
            self.root, text="Upload", font=("Courier New Bold", 14),
            bg="#fe3330", fg="#ffffff",
            activebackground="#b32028", activeforeground="#ffffff",
            bd=0, padx=15, pady=5, cursor="hand2",
            command=self.open_file_locator
        )
        self.upload_btn.pack(padx=10, pady=(30, 0))

    def setup_file_label(self):
        self.file_label_frame = tk.Frame(self.root, bg="#1f1f1f")

        self.selected_label = tk.Label(self.file_label_frame, text="Selected:", font=("Courier New", 10), bg="#1f1f1f", fg="#e8e8e8")
        self.selected_label.pack(side="left")

        self.filename_label = tk.Label(self.file_label_frame, text="", font=("Courier New Bold", 10), bg="#1f1f1f", fg="#99e695")
        self.filename_label.pack(side="left")

    def setup_analyse_button(self):
        self.btn_frame = tk.Frame(self.root, bg="#1f1f1f")

        self.analyse_btn = tk.Button(
            self.btn_frame, text="Analyse", font=("Courier New Bold", 12),
            bg="#fe3330", fg="white",
            activebackground="#b32028", activeforeground="white",
            bd=0, padx=15, pady=5, cursor="hand2",
            command=self.analyse_video
        )
        self.analyse_btn.pack(side="left", padx=15)

        self.change_video_btn = tk.Button(
            self.btn_frame, text="Change Video", font=("Courier New", 12),
            bg="#333333", fg="#b8b8b8",
            activebackground="#555555", activeforeground="#b8b8b8",
            bd=0, padx=15, pady=5, cursor="hand2",
            command=self.open_file_locator
        )
        self.change_video_btn.pack(side="left")

    def setup_result_area(self):
        self.result_frame = tk.Frame(self.root, bg="#1f1f1f")

        subsections_container = tk.Frame(self.result_frame, bg="#1f1f1f")
        subsections_container.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        # CLASSIFICATION
        class_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1, width=150)
        class_frame.pack(side="left", fill="both", padx=(0, 5))
        class_frame.pack_propagate(False)

        class_header = tk.Label(
            class_frame, text="CLASSIFICATION",
            font=("Arial Bold", 10),
            bg="#252525", fg="#ffffff",
            anchor="center", padx=10, pady=5
        )
        class_header.pack(fill="x")

        self.class_value = tk.Label(
            class_frame, text="",
            font=("Arial Bold", 16),
            bg="#141414", fg="#ffffff",
            anchor="center", padx=10, pady=20,
            justify="center"
        )
        self.class_value.pack(fill="both", expand=True)

        # DESCRIPTION
        desc_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1)
        desc_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        desc_header = tk.Label(
            desc_frame, text="DESCRIPTION",
            font=("Arial Bold", 10),
            bg="#252525", fg="#ffffff",
            anchor="center", padx=10, pady=5
        )
        desc_header.pack(fill="x")

        self.desc_value = tk.Label(
            desc_frame, text="",
            font=("Arial", 12),
            bg="#141414", fg="#fcfcfc",
            anchor="center", padx=10, pady=20,
            justify="center",
            wraplength=400
        )
        self.desc_value.pack(fill="both", expand=True)

        # FALL TIMESTAMPS
        frames_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1, width=150)
        frames_frame.pack(side="left", fill="both")
        frames_frame.pack_propagate(False)
        tk.Label(frames_frame, text="FALL TIMESTAMPS", font=("Arial Bold", 10),
                 bg="#252525", fg="#ffffff", anchor="center", padx=10, pady=5).pack(fill="x")

        self.frames_container = tk.Frame(frames_frame, bg="#141414")
        self.frames_container.pack(expand=True)
        self.frames_value = tk.Text(self.frames_container,
                                    font=("Courier New", 11),
                                    bg="#141414", fg="#fcfcfc",
                                    wrap="none", relief="flat",
                                    padx=10, pady=25,
                                    highlightthickness=0, highlightbackground="#141414",
                                    highlightcolor="#141414",
                                    takefocus=0)
        self.frames_value.pack(expand=True)
        self.frames_value.tag_configure("center", justify="center")
        self.frames_value.config(state="disabled")

        self.setup_metrics_bar()

    def setup_metrics_bar(self):
        self.metrics_frame = tk.Frame(self.result_frame, bg="#1f1f1f", height=35)
        self.metrics_frame.pack(side="bottom", fill="x", padx=5, pady=(5, 5))

        metrics_container = tk.Frame(self.metrics_frame, bg="#1f1f1f")
        metrics_container.pack(expand=True)

        # accuracy
        self.accuracy_label = tk.Label(
            metrics_container,
            text="Accuracy: --",
            font=("Courier New", 9),
            bg="#1f1f1f",
            fg="#888888"
        )
        self.accuracy_label.pack(side="left", padx=10)

        tk.Label(metrics_container, text="|", font=("Courier New", 9),
                 bg="#1f1f1f", fg="#444444").pack(side="left", padx=5)

        # precision
        self.precision_label = tk.Label(
            metrics_container,
            text="Precision: --",
            font=("Courier New", 9),
            bg="#1f1f1f",
            fg="#888888"
        )
        self.precision_label.pack(side="left", padx=10)

        tk.Label(metrics_container, text="|", font=("Courier New", 9),
                 bg="#1f1f1f", fg="#444444").pack(side="left", padx=5)

        # recall
        self.recall_label = tk.Label(
            metrics_container,
            text="Recall: --",
            font=("Courier New", 9),
            bg="#1f1f1f",
            fg="#888888"
        )
        self.recall_label.pack(side="left", padx=10)


    # ------------------------ UTILITY METHODS ------------------------ #
    def timestamp_to_frame(self, timestamp):
        """Convert timestamp in seconds to frame number"""
        return int(timestamp * self.video_fps)

    def frame_to_timestamp(self, frame):
        """Convert frame number to timestamp in seconds"""
        return frame / self.video_fps

    def format_timestamp(self, seconds):
        """Format timestamp as MM:SS.ms"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"

    # ------------------------ FILE HANDLING ------------------------ #
    def open_file_locator(self):
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if file_path:
            self.handle_file(file_path)

    def handle_file(self, file_path):
        print(f"Selected file: {file_path}")
        self.current_file_path = file_path
        
        # Get video FPS
        cap = cv2.VideoCapture(file_path)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Video FPS: {self.video_fps}")
        
        self.show_analyse_btn()
        self.show_thumbnail(file_path)

    # ------------------------ ACTION METHODS ------------------------ #
    def show_analyse_btn(self):
        if self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack_forget()

        if self.upload_btn.winfo_ismapped():
            self.upload_btn.pack_forget()

        if self.preview_frame.winfo_ismapped():
            self.preview_frame.pack_forget()

        if not self.thumbnail_frame.winfo_ismapped():
            self.thumbnail_frame.pack(pady=5)

        if not self.file_label_frame.winfo_ismapped():
            self.file_label_frame.pack()

        if not self.btn_frame.winfo_ismapped():
            self.btn_frame.pack(padx=10, pady=(50, 5))

        if not self.result_frame.winfo_ismapped():
            self.result_frame.pack(side="bottom", padx=10, pady=10, fill="both", expand=True)

    def analyse_video(self):
        if not self.current_file_path:
            self.class_value.config(text="No video selected")
            self.desc_value.config(text="Please upload or choose a video first.")
            self.frames_value.config(state="normal")
            self.frames_value.delete("1.0", "end")
            self.frames_value.insert("1.0", "—", "center")
            self.frames_value.config(state="disabled")
            print("Analyse called but no video selected.")
            return None

        video_path = self.current_file_path
        print(f"Analyse called for: {video_path}")

        # TEST DATA
        try:
            result = model_runtime.analyse_video(video_path, self.ctx)
        except Exception as e:
            print(f"[ERROR] analysis failed: {e}")
            # graceful fallback to a neutral result so GUI never crashes
            result = {
                "class": "UNKNOWN",
                "desc": "Analysis failed. See console logs.",
                "fall_timestamps": [],
                "metrics": {}
            }

        self.display_analysis_result(result)
        self.last_analysis_path = video_path
        return video_path

    def display_analysis_result(self, result_dict):

        cls = result_dict.get("class", "UNKNOWN")
        desc = result_dict.get("desc", "")
        timestamps = result_dict.get("fall_timestamps", [])
        metrics = result_dict.get("metrics", {})

        if cls == "FALL":
            self.fall_counter += 1
            self.detected_label.config(text=f"FALL DETECTED!")
            if not self.flash_job:
                self.flash_detected_label(True)
        else:
            if self.flash_job:
                self.root.after_cancel(self.flash_job)
                self.flash_job = None
            self.detected_label.config(text="NO FALL DETECTED")

        self.class_value.config(text=cls)
        self.desc_value.config(text=desc if desc else "")
        self.frames_value.config(state="normal")
        self.frames_value.delete("1.0", "end")

        if timestamps:
            for i, timestamp in enumerate(timestamps):
                start = self.frames_value.index("insert")
                formatted_time = self.format_timestamp(timestamp)
                self.frames_value.insert("insert", formatted_time, "center")
                end = self.frames_value.index("insert")

                tag_name = f"timestamp_{timestamp}_{i}"
                self.frames_value.tag_add(tag_name, start, end)
                self.frames_value.tag_config(
                    tag_name,
                    foreground="#3399ff",
                    underline=True,
                )
                self.frames_value.tag_bind(tag_name, "<Enter>", lambda e: self.frames_value.config(cursor="hand2"))
                self.frames_value.tag_bind(tag_name, "<Leave>", lambda e: self.frames_value.config(cursor=""))
                self.frames_value.tag_bind(tag_name, "<Button-1>", lambda e, t=timestamp: self.show_timestamp(t))

                if i < len(timestamps) - 1:
                    self.frames_value.insert("insert", "\n", "center")

        self.frames_value.config(state="disabled")
        self.frames_value.bind("<FocusIn>", lambda e: self.root.focus())

        # update metrics bar
        self.update_metrics(metrics)

        print(f"\n[Analysis {self.fall_counter}] -> Class: {cls}, Timestamps: {timestamps}")

    def update_metrics(self, metrics):
        accuracy = metrics.get("accuracy", None)
        precision = metrics.get("precision", None)
        recall = metrics.get("recall", None)

        # accuracy
        if accuracy is not None:
            self.accuracy_label.config(
                text=f"Accuracy: {accuracy:.1f}%",
                fg="#99e695" if accuracy >= 90 else "#e8e8e8"
            )
        else:
            self.accuracy_label.config(text="Accuracy: --", fg="#888888")

        # precision
        if precision is not None:
            self.precision_label.config(
                text=f"Precision: {precision:.1f}%",
                fg="#99e695" if precision >= 90 else "#e8e8e8"
            )
        else:
            self.precision_label.config(text="Precision: --", fg="#888888")

        # recall
        if recall is not None:
            self.recall_label.config(
                text=f"Recall: {recall:.1f}%",
                fg="#99e695" if recall >= 90 else "#e8e8e8"
            )
        else:
            self.recall_label.config(text="Recall: --", fg="#888888")

    def show_timestamp(self, timestamp):
        if not self.current_file_path:
            print("ERROR: No video loaded.")
            return

        if hasattr(self, 'playback_job') and self.playback_job:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None

        # convert timestamp to frame
        center_frame = self.timestamp_to_frame(timestamp)
        frames_range = int(0.67 * self.video_fps)
        start_frame = max(0, center_frame - frames_range)
        
        cap = cv2.VideoCapture(self.current_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        end_frame = min(center_frame + frames_range, total_frames - 1)

        print(f"Playing frames {start_frame} to {end_frame} (around {self.format_timestamp(timestamp)})")

        self.play_frame_sequence(start_frame, end_frame, start_frame)

    def play_frame_sequence(self, current_frame, end_frame, start_frame):
        if current_frame > end_frame:
            return

        if not self.current_file_path:
            return

        cap = cv2.VideoCapture(self.current_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((650, 400), Image.Resampling.LANCZOS)

            # update thumbnail
            imgtk = ImageTk.PhotoImage(img)
            self.thumbnail_label.configure(image=imgtk)
            self.thumbnail_label.image = imgtk
            self.thumbnail_label.pack(padx=10, pady=10)

            self.playback_job = self.root.after(20, self.play_frame_sequence, current_frame + 1, end_frame, start_frame)
        else:
            print(f"Could not read frame {current_frame}, stopping playback")

    def flash_detected_label(self, state):
        color = "#ffffff" if state else "#fe3330"
        self.detected_label.config(fg=color)
        self.flash_job = self.root.after(250, self.flash_detected_label, not state)

        # Stop flashing after 3 seconds
        if not hasattr(self, "_flash_stop_scheduled") or not self._flash_stop_scheduled:
            self._flash_stop_scheduled = True
            self.root.after(3000, self.stop_flash)

    def stop_flash(self):
        if self.flash_job:
            self.root.after_cancel(self.flash_job)
            self.flash_job = None
        self.detected_label.config(fg="#fe3330")
        self._flash_stop_scheduled = False

    def show_thumbnail(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            img = img.resize((650, 400), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(img)
            self.thumbnail_label.configure(image=imgtk)
            self.thumbnail_label.image = imgtk
            self.thumbnail_label.pack(padx=10, pady=10)

        filename = os.path.basename(video_path)
        self.filename_label.config(text=filename)

    def refresh(self):
        if self.flash_job:
            self.root.after_cancel(self.flash_job)
            self.flash_job = None

        if self.playback_job:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None

        self.current_file_path = None
        self.filename_label.config(text="")
        self.detected_label.config(text="", fg="#fe3330")

        self.class_value.config(text="")
        self.desc_value.config(text="")
        self.frames_value.config(state="normal")
        self.frames_value.delete("1.0", "end")
        self.frames_value.insert("1.0", "", "center")
        self.frames_value.config(state="disabled")

        self.update_metrics({})

        for w in [self.thumbnail_frame, self.btn_frame, self.result_frame, self.file_label_frame]:
            if w.winfo_ismapped():
                w.pack_forget()

        if not self.preview_frame.winfo_ismapped():
            self.preview_frame.pack(pady=10)
        if not self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack(pady=(5, 15))
        if not self.upload_btn.winfo_ismapped():
            self.upload_btn.pack(padx=15, pady=5)


FallDetection()