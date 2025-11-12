import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2, os, threading
import vlc
import platform
from models import get_detector

SYSTEM_PROMPT = (
    "Analyze each input video and determine whether a fall event occurred.\n\n"
    "Output Format\n"
    "Always respond only in valid JSON following this schema exactly:\n\n"
    "{\n"
    ' \"class\": \"FALL\" | \"NO_FALL\",\n'
    ' \"confidence\": <float between 0.0 and 1.0>,\n'
    ' \"reasoning\": <short explanation of why this classification was chosen, up to 600 characters>\n'
    ' \"fall_start\": <time in seconds when the fall began, 0 if NO_FALL>\n'
    ' \"fall_end\": <time in seconds at moment of impact, 0 if NO_FALL>\n'
    "}\n\n"
    "Guidelines\n"
    "- Falls are scenes where the person experiences rapid and uncontrolled descent resulting in impact.\n"
    "- Classify crouching motions, controlled descents, and falling on a bed as NO_FALL.\n"
    "- If the classification is NO_FALL, then fall_start and fall_end should be 0.\n"
    "- The \"confidence\" should reflect certainty about the classification.\n"
    "- The \"reasoning\" must summarise key visual cues (e.g., \"rapid descent followed by lying posture\").\n"
    "- Do not include any text outside the JSON.\n"
    "- When uncertain or visibility is poor, lower the confidence but still choose the best label.\n"
)


class FallDetection:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("900x880")
        self.root.title("Fall Detection")
        self.root.config(bg="#1f1f1f")
        self.root.bind("<space>", self.on_space_pressed)

        self.detector_options = ["GPT-4 Vision", "Gemini 1.5 Flash", "Claude 3.5 Sonnet"]
        self.selected_detector = "GPT-4 Vision"
        self.current_file_path = None
        self.video_fps = 30
        self.fall_start = 0
        self.fall_end = 0

        self.vlc_instance = vlc.Instance("--no-xlib", "--avcodec-hw=dxva2","--no-video-title-show","--width=640", "--height=360")
        self.vlc_player = self.vlc_instance.media_player_new()
        self.is_playing = False
        self.update_slider_job = None

        self.dragging = False
        self.was_playing = False
        self.video_ready = False
        self.vlc_media = None

        self.setup_navbar()
        self.setup_detector_selector()
        self.setup_preview_area()
        self.setup_video_player()
        self.setup_welcome_section()
        self.setup_file_label()
        self.setup_analyse_button()
        self.setup_result_area()

        self.root.mainloop()

    def setup_navbar(self):
        navbar = tk.Frame(self.root, bg="#000000", height=120)
        navbar.pack(fill="x")

        title_frame = tk.Frame(navbar, bg="#000000")
        title_frame.pack(side="left", padx=20)
        tk.Label(title_frame, text="fall", font=("Arial Bold", 25), bg="#000000", fg="#fe3330").pack(side="left")
        tk.Label(title_frame, text="Detection", font=("Arial Bold", 24), bg="#000000", fg="#ffffff").pack(side="left")

        right_frame = tk.Frame(navbar, bg="#000000")
        right_frame.pack(side="right", padx=(20, 0))

        logo_img = Image.open("./img/segfault_logo.png").resize((130, 75), Image.Resampling.LANCZOS)
        self.logo_tk = ImageTk.PhotoImage(logo_img)
        tk.Label(right_frame, image=self.logo_tk, bg="#000000").pack(side="right", padx=(10, 0))

        refresh_img = Image.open("./img/refresh.png").resize((20, 20), Image.Resampling.LANCZOS)
        self.refresh_tk = ImageTk.PhotoImage(refresh_img)
        tk.Button(right_frame, image=self.refresh_tk, bg="#000000", bd=0, activebackground="#1a1a1a",
                  cursor="hand2", command=self.refresh).pack(side="right", padx=(0, 10))

    def setup_detector_selector(self):
        self.selector_frame = tk.Frame(self.root, bg="#1f1f1f")
        tk.Label(self.selector_frame, text="Model:", font=("Courier New Bold", 9), bg="#1f1f1f", fg="#e8e8e8").pack(side="left", padx=(0, 10))

        self.detector_var = tk.StringVar(value=self.selected_detector)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Subtle.TCombobox", fieldbackground="#1f1f1f", background="#1f1f1f",
                        foreground="#e8e8e8", arrowcolor="#fe3330", bordercolor="#555555",
                        lightcolor="#555555", darkcolor="#555555", borderwidth=1, padding=2, font=("Arial", 8))
        style.map("Subtle.TCombobox", fieldbackground=[('readonly', '#1f1f1f')],
                  foreground=[('readonly', '#e8e8e8')], background=[('readonly', '#1f1f1f')],
                  selectbackground=[('readonly', '#1f1f1f')], selectforeground=[('readonly', '#e8e8e8')])

        detector_dropdown = ttk.Combobox(self.selector_frame, textvariable=self.detector_var,
                                         values=self.detector_options, state="readonly", width=20,
                                         font=("Arial", 8), style="Subtle.TCombobox")
        detector_dropdown.pack(side="left")
        detector_dropdown.bind("<<ComboboxSelected>>", self.on_detector_changed)

        self.compare_btn = tk.Button(self.selector_frame, text="Compare Models", font=("Arial", 9), bg="#333333", fg="#b8b8b8", activebackground="#555555",bd=0, padx=15, pady=5, cursor="hand2", command=self.compare_view,relief="flat", highlightthickness=0)
        self.compare_btn.pack(side="left", padx=10)

    def on_detector_changed(self, event=None):
        self.selected_detector = self.detector_var.get()
        print(f"Switched to: {self.selected_detector}")

    def setup_preview_area(self):
        self.detected_label = tk.Label(self.root, text="", font=("fixedsys", 22), bg="#1f1f1f", fg="#fe3330")
        self.detected_label.pack(side="top", pady=(20, 0))

        self.preview_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.preview_frame.pack(pady=10)

        self.preview_label = tk.Label(self.preview_frame, text="📁 .mp4 .avi .mov 📁",
                                      font=("Courier New", 12), relief="solid", width=65, height=23,
                                      bg="#141414", fg="#cccaca")
        self.preview_label.pack(padx=10, pady=10)
        self.preview_label.bind("<Button-1>", lambda e: self.open_file_locator())
        self.preview_label.bind("<Enter>", lambda e: self.preview_label.config(bg="#1a1a1a", cursor="hand2"))
        self.preview_label.bind("<Leave>", lambda e: self.preview_label.config(bg="#141414", cursor=""))

    def setup_video_player(self):
        self.video_frame = tk.Frame(self.root, bg="#1f1f1f")

        self.video_canvas = tk.Frame(self.video_frame, bg="#000000", width=650, height=400)
        self.video_canvas.pack(padx=10, pady=(10, 5))
        self.video_canvas.pack_propagate(False)

        if platform.system() == 'Windows':
            self.vlc_player.set_hwnd(self.video_canvas.winfo_id())
        elif platform.system() == 'Darwin':
            self.vlc_player.set_nsobject(self.video_canvas.winfo_id())
        else:
            self.vlc_player.set_xwindow(self.video_canvas.winfo_id())

        controls_frame = tk.Frame(self.video_frame, bg="#1f1f1f")
        controls_frame.pack(fill="x", padx=10, pady=(0, 10))

        slider_frame = tk.Frame(controls_frame, bg="#1f1f1f")
        slider_frame.pack(fill="x", pady=(0, 5))

        self.play_btn = tk.Button(slider_frame, text="▶", font=("Arial", 11), bg="#1f1f1f",
                                  fg="#e8e8e8", activebackground="#1f1f1f", activeforeground="#ffffff",
                                  bd=0, width=2, cursor="hand2", command=self.toggle_play,
                                  relief="flat", highlightthickness=0)
        self.play_btn.pack(side="left", padx=(0, 8))

        self.time_slider = tk.Scale(slider_frame, from_=0, to=1000, orient="horizontal",
                                    bg="#252525", fg="#e8e8e8", activebackground="#fe3330",
                                    troughcolor="#141414", highlightthickness=0, bd=0,
                                    sliderrelief="flat", showvalue=False)
        self.time_slider.pack(side="left", fill="x", expand=True)
        self.time_slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.time_slider.bind("<ButtonRelease-1>", self.on_slider_release)
        self.time_slider.bind("<B1-Motion>", self.on_slider_drag)

        self.time_label = tk.Label(slider_frame, text="00:00.00 / 00:00", font=("Courier New", 9),
                                   bg="#1f1f1f", fg="#888888", width=20)
        self.time_label.pack(side="right", padx=(8, 0))

    def setup_welcome_section(self):
        self.welcome_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.welcome_frame.pack(pady=(5, 15))

        tk.Label(self.welcome_frame, text="Upload a Video", font=("Arial bold", 26),
                 bg="#1f1f1f", fg="#cccaca").pack()

        tk.Label(self.welcome_frame,
                 text="\nPlease select your desired video to analyse for the system\nto detect and identify a fall that occurs.",
                 font=("Helvetica", 10), bg="#1f1f1f", fg="#e8e8e8").pack()

        self.upload_btn = tk.Button(self.root, text="Upload", font=("Arial", 14),
                                    bg="#fe3330", fg="#ffffff", activebackground="#b32028",
                                    bd=0, padx=15, pady=5, cursor="hand2", command=self.open_file_locator)
        self.upload_btn.pack(padx=10, pady=(30, 0))

    def setup_file_label(self):
        self.file_label_frame = tk.Frame(self.root, bg="#1f1f1f")
        tk.Label(self.file_label_frame, text="Selected:", font=("Courier New", 10),
                 bg="#1f1f1f", fg="#e8e8e8").pack(side="left")
        self.filename_label = tk.Label(self.file_label_frame, text="", font=("Courier New Bold", 10),
                                       bg="#1f1f1f", fg="#99e695")
        self.filename_label.pack(side="left")

    def setup_analyse_button(self):
        self.btn_frame = tk.Frame(self.root, bg="#1f1f1f")

        self.analyse_btn = tk.Button(self.btn_frame, text="Analyse", font=("Arial Bold", 12),
                                     bg="#fe3330", fg="white", activebackground="#b32028",
                                     bd=0, padx=15, pady=5, cursor="hand2", command=self.analyse_video)
        self.analyse_btn.pack(side="left", padx=15)

        self.change_video_btn = tk.Button(self.btn_frame, text="Change Video", font=("Arial", 12),
                                          bg="#333333", fg="#b8b8b8", activebackground="#555555",
                                          bd=0, padx=15, pady=5, cursor="hand2", command=self.open_file_locator)
        self.change_video_btn.pack(side="left")

    def setup_result_area(self):
        self.result_frame = tk.Frame(self.root, bg="#1f1f1f")

        subsections_container = tk.Frame(self.result_frame, bg="#1f1f1f")
        subsections_container.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        class_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1, width=150)
        class_frame.pack(side="left", fill="both", padx=(0, 5))
        class_frame.pack_propagate(False)
        tk.Label(class_frame, text="CLASSIFICATION", font=("Arial Bold", 10), bg="#252525",
                 fg="#ffffff", anchor="center", padx=10, pady=5).pack(fill="x")
        self.class_value = tk.Label(class_frame, text="", font=("Arial Bold", 16), bg="#141414",
                                    fg="#ffffff", anchor="center", padx=10, pady=20, justify="center")
        self.class_value.pack(fill="both", expand=True)

        desc_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1)
        desc_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        tk.Label(desc_frame, text="DESCRIPTION", font=("Arial Bold", 10), bg="#252525",
                 fg="#ffffff", anchor="center", padx=10, pady=5).pack(fill="x")
        self.desc_value = tk.Label(desc_frame, text="", font=("Arial", 12), bg="#141414",
                                   fg="#fcfcfc", anchor="center", padx=10, pady=20,
                                   justify="center", wraplength=500)
        self.desc_value.pack(fill="both", expand=True)

        frames_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1, width=150)
        frames_frame.pack(side="left", fill="both")
        frames_frame.pack_propagate(False)
        tk.Label(frames_frame, text="TIMESTAMPS", font=("Arial Bold", 10), bg="#252525",
                 fg="#ffffff", anchor="center", padx=10, pady=5).pack(fill="x")

        self.frames_container = tk.Frame(frames_frame, bg="#141414")
        self.frames_container.pack(expand=True)
        self.frames_value = tk.Text(self.frames_container, font=("Courier New", 11), bg="#141414",
                                    fg="#fcfcfc", wrap="none", relief="flat", padx=10, pady=25,
                                    highlightthickness=0, takefocus=0)
        self.frames_value.pack(expand=True)
        self.frames_value.tag_configure("center", justify="center")
        self.frames_value.config(state="disabled")

        self.model_analysis_label = tk.Label(self.result_frame, text="", font=("Courier New", 9),
                                             bg="#1f1f1f", fg="#888888")
        self.model_analysis_label.pack(pady=(5, 0))

    def open_file_locator(self):
        file_path = filedialog.askopenfilename(title="Select a video file",
                                               filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")))
        if file_path:
            self.handle_file(file_path)

    def handle_file(self, file_path):
        print(f"Selected file: {file_path}")
        self.current_file_path = file_path

        cap = cv2.VideoCapture(file_path)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.filename_label.config(text=os.path.basename(file_path))
        self.show_video_screen()
        self.load_video(file_path)

    def show_video_screen(self):
        if self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack_forget()
        if self.upload_btn.winfo_ismapped():
            self.upload_btn.pack_forget()
        if self.preview_frame.winfo_ismapped():
            self.preview_frame.pack_forget()

        if not self.video_frame.winfo_ismapped():
            self.video_frame.pack(pady=5)
        if not self.file_label_frame.winfo_ismapped():
            self.file_label_frame.pack()
        if not self.btn_frame.winfo_ismapped():
            self.btn_frame.pack(padx=10, pady=(20, 5))
        if not self.selector_frame.winfo_ismapped():
            self.selector_frame.pack(pady=(15, 0))
        if not self.result_frame.winfo_ismapped():
            self.result_frame.pack(side="bottom", padx=10, pady=10, fill="both", expand=True)

    def load_video(self, file_path):
        self.vlc_media = self.vlc_instance.media_new(file_path)
        self.vlc_player.set_media(self.vlc_media)

        self.vlc_player.play()
        self.root.after(50, self.stop_initial_playback)

    def stop_initial_playback(self):
        self.vlc_player.pause()
        self.is_playing = False
        self.play_btn.config(text="▶")
        length = self.vlc_player.get_length()
        if length > 0:
            self.time_slider.config(to=length)
            self.time_label.config(text=f"00:00.00 / {self.format_time(length)}")

    def toggle_play(self):
        state_playing = self.vlc_player.is_playing()
        if state_playing:
            self.vlc_player.pause()
            self.is_playing = False
            self.play_btn.config(text="▶")
            if self.update_slider_job:
                self.root.after_cancel(self.update_slider_job)
                self.update_slider_job = None
        else:
            try:
                state = self.vlc_player.get_state()
                if state == vlc.State.Ended:
                    self.vlc_player.set_time(0)
            except Exception:
                pass

            self.vlc_player.play()
            self.is_playing = True
            self.play_btn.config(text="⏸")
            if not self.update_slider_job:
                self.update_slider()

    def on_space_pressed(self, event):
        self.toggle_play()

    def update_slider(self):
        if self.vlc_player is None or self.vlc_media is None:
            return

        try:
            current = self.vlc_player.get_time()
            length = self.vlc_player.get_length()

            if length > 0 and not self.dragging:
                self.time_slider.config(to=length)
                self.time_slider.set(current)
                self.time_label.config(text=f"{self.format_time(current)} / {self.format_time(length)}")

            state = self.vlc_player.get_state()
            if state == vlc.State.Ended:
                self.vlc_player.stop()
                self.is_playing = False
                self.play_btn.config(text="▶")

        except Exception as e:
            print(f"[Slider Update Error] {e}")

        self.update_slider_job = self.root.after(50, self.update_slider)

    def on_slider_press(self, event):
        if self.is_playing:
            self.was_playing = True
            self.vlc_player.pause()
        else:
            self.was_playing = False

    def on_slider_drag(self, event):
        length = self.vlc_player.get_length()
        if length > 0:
            new_time = int(self.time_slider.get())
            new_time = max(0, min(new_time, length))
            self.vlc_player.set_time(new_time)
            self.time_label.config(text=f"{self.format_time(new_time)} / {self.format_time(length)}")

    def on_slider_release(self, event):
        length = self.vlc_player.get_length()
        if length > 0:
            new_time = int(self.time_slider.get())
            self.vlc_player.set_time(new_time)
            current = self.vlc_player.get_time()
            self.time_label.config(text=f"{self.format_time(current)} / {self.format_time(length)}")
            if self.was_playing:
                self.vlc_player.play()

    def on_slider_move(self, value):
        if hasattr(self, 'video_ready') and self.video_ready:
            self.vlc_player.set_time(int(float(value)))

    def on_media_parsed(self, event):
        self.video_ready = True

    def skip_time(self, milliseconds):
        current_time = self.vlc_player.get_time()
        new_time = max(0, current_time + milliseconds)
        self.vlc_player.set_time(int(new_time))

    def jump_to_fall_start(self):
        self.vlc_player.set_time(int(self.fall_start * 1000))

    def jump_to_fall_end(self):
        self.vlc_player.set_time(int(self.fall_end * 1000))

    def format_time(self, milliseconds):
        total_seconds = milliseconds / 1000
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        hundredths = int(((total_seconds - int(total_seconds)) * 100))
        return f"{mins:02d}:{secs:02d}.{hundredths:02d}"

    def analyse_video(self):
        if not self.current_file_path:
            self.class_value.config(text="No video selected")
            self.desc_value.config(text="Please upload or choose a video first.")
            return

        self.class_value.config(text="")
        self.desc_value.config(text=f"analyzing with {self.selected_detector}...")
        self.model_analysis_label.config(text="")
        self.root.update_idletasks()

        thread = threading.Thread(target=self.run_analysis_in_thread,
                                  args=(self.current_file_path, self.selected_detector))
        thread.start()

    def run_analysis_in_thread(self, video_path, detector_name):
        try:
            detector = get_detector(detector_name)
            result = detector.analyze_video(video_path, SYSTEM_PROMPT)
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            result = {"class": "ERROR", "confidence": 0.0, "reasoning": f"Error: {str(e)}",
                      "fall_start": 0, "fall_end": 0}
        self.root.after(0, lambda: self.display_analysis_result(result))

    def display_analysis_result(self, result):
        cls = result.get("class", "UNKNOWN")
        desc = result.get("reasoning", "No description available")
        self.fall_start = result.get("fall_start", 0)
        self.fall_end = result.get("fall_end", 0)

        self.class_value.config(text=cls)
        self.desc_value.config(text=desc)

        self.frames_value.config(state="normal")
        self.frames_value.delete("1.0", "end")
        self.frames_value.insert("end", f"Start: {self.fall_start:.2f}s\n", "center")
        self.frames_value.insert("end", f"End: {self.fall_end:.2f}s\n", "center")
        self.frames_value.config(state="disabled")

        self.detected_label.config(text="FALL DETECTED!" if cls == "FALL" else "NO FALL DETECTED")

        if self.fall_start > 0 or self.fall_end > 0:
            pass
        else:
            pass

        if 'total_cost_estimate' in result:
            cost_info = result['total_cost_estimate']
            confidence = result['confidence']
            if 'total_usd' in cost_info:
                text = f"Confidence: {confidence:.2%} | Model: {result.get('api_used', 'N/A')} | Cost: ${cost_info['total_usd']:.4f}"
                if 'images' in cost_info:
                    text += f" | Frames: {cost_info['images']}"
            else:
                text = f"Model: {result.get('api_used', 'N/A')} | {cost_info.get('note', 'Free')}"
            self.model_analysis_label.config(text=text)

    def refresh(self):
        if self.update_slider_job:
            self.root.after_cancel(self.update_slider_job)
        self.vlc_player.stop()
        self.current_file_path = None
        self.is_playing = False
        self.fall_start = 0
        self.fall_end = 0

        self.filename_label.config(text="")
        self.detected_label.config(text="")
        self.class_value.config(text="")
        self.desc_value.config(text="")
        self.model_analysis_label.config(text="")
        self.time_label.config(text="00:00 / 00:00")
        self.time_slider.set(0)
        self.play_btn.config(text="▶")

        self.frames_value.config(state="normal")
        self.frames_value.delete("1.0", "end")
        self.frames_value.config(state="disabled")

        for w in [self.video_frame, self.btn_frame, self.result_frame, self.file_label_frame, self. models_frame_h, self.controls_frame]:
            if w.winfo_ismapped():
                w.pack_forget()

        if not self.preview_frame.winfo_ismapped():
            self.preview_frame.pack(pady=10)
        if not self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack(pady=(5, 15))
        if not self.upload_btn.winfo_ismapped():
            self.upload_btn.pack(padx=15, pady=5)

        self.detector_var.set("")
        self.selected_detector = ""

    def hide_current_view(self):
        frames_to_hide = [
            self.welcome_frame,
            self.upload_btn,
            self.result_frame,
            self.selector_frame
        ]

        for w in frames_to_hide:
            if w.winfo_ismapped():
                w.pack_forget()

    def compare_view(self):
        self.hide_current_view()

        self.models_frame_h = tk.Frame(self.root, bg="#1f1f1f")
        self.models_frame_h.pack(fill="both", expand=True, padx=10, pady=10)

        self.model_boxes = []

        controls_frame = tk.Frame(self.root, bg="#1f1f1f")
        controls_frame.pack(pady=(0, 10))

        add_btn = tk.Button(controls_frame, text="+", font=("Arial", 12), bg="#2e2e2e", fg="#fff",
                            width=3, command=lambda: self.add_model_box())
        add_btn.pack(side="left", padx=5)

        remove_btn = tk.Button(controls_frame, text="-", font=("Arial", 12), bg="#2e2e2e", fg="#fff",
                               width=3, command=lambda: self.remove_model_box())
        remove_btn.pack(side="left", padx=5)

        for i in range(2):
            self.add_model_box()

    def compare_view(self):
        self.hide_current_view()

        self.models_frame_h = tk.Frame(self.root, bg="#1f1f1f")
        self.models_frame_h.pack(fill="both", expand=True, padx=10, pady=10)

        self.model_boxes = []

        self.controls_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.controls_frame.pack(fill="x", pady=(0, 5))

        tk.Frame(self.controls_frame, bg="#1f1f1f").pack(side="left", expand=True)

        self.add_btn = tk.Button(self.controls_frame, text="+", font=("Arial", 10),
                                 bg="#2b2b2b", fg="#ccc", width=2, bd=0, relief="flat",
                                 activebackground="#3a3a3a", activeforeground="#fff",
                                 command=self.add_model_box)
        self.add_btn.pack(side="right", padx=(0, 5))
        self.remove_btn = tk.Button(self.controls_frame, text="-", font=("Arial", 10),
                                    bg="#2b2b2b", fg="#ccc", width=2, bd=0, relief="flat",
                                    activebackground="#3a3a3a", activeforeground="#fff",
                                    command=self.remove_model_box)
        self.remove_btn.pack(side="right", padx=(0, 5))


        for _ in range(2):
            self.add_model_box()

    def add_model_box(self):
        idx = len(self.model_boxes)
        model_vbox = tk.Frame(self.models_frame_h, bg="#1f1f1f", relief="flat", bd=0)
        model_vbox.pack(side="left", fill="both", expand=True, padx=5)

        selector_frame = tk.Frame(model_vbox, bg="#1f1f1f")
        selector_frame.pack(pady=(0, 5))
        tk.Label(selector_frame, text=f"Model {idx + 1}:", font=("Courier New Bold", 9),
                 bg="#1f1f1f", fg="#e8e8e8").pack(side="left", padx=(0, 10))

        detector_var = tk.StringVar(value=self.selected_detector)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Subtle.TCombobox", fieldbackground="#1f1f1f", background="#1f1f1f",
                        foreground="#e8e8e8", arrowcolor="#fe3330", bordercolor="#555555",
                        lightcolor="#555555", darkcolor="#555555", borderwidth=1, padding=2, font=("Arial", 8))
        style.map("Subtle.TCombobox", fieldbackground=[('readonly', '#1f1f1f')],
                  foreground=[('readonly', '#e8e8e8')], background=[('readonly', '#1f1f1f')],
                  selectbackground=[('readonly', '#1f1f1f')], selectforeground=[('readonly', '#e8e8e8')])

        detector_dropdown = ttk.Combobox(selector_frame, textvariable=detector_var,
                                         values=self.detector_options, state="readonly", width=20,
                                         font=("Arial", 8), style="Subtle.TCombobox")
        detector_dropdown.pack(side="left")
        detector_dropdown.bind("<<ComboboxSelected>>", lambda e, idx=idx: self.on_model_selected(idx))

        results_frame = tk.Frame(model_vbox, bg="#141414", relief="solid", bd=1)
        results_frame.pack(fill="both", expand=True, pady=(5, 0))
        tk.Label(results_frame, text="", font=("Arial", 10), bg="#141414", fg="#fcfcfc").pack(expand=True)

        results_widgets = self.setup_model_results_vertical(results_frame)

        self.model_boxes.append({
            "frame": model_vbox,
            "selector_var": detector_var,
            "results_frame": results_frame
        })

    def remove_model_box(self):
        if not self.model_boxes:
            return

        if len(self.model_boxes) > 2:
            last_box = self.model_boxes.pop()
            last_box["frame"].destroy()
        else:
            self.show_video_screen()
            if hasattr(self, "models_frame_h") and self.models_frame_h.winfo_ismapped():
                self.models_frame_h.pack_forget()
            if hasattr(self, "controls_frame") and self.controls_frame.winfo_ismapped():
                self.controls_frame.pack_forget()
            self.model_boxes = []

        if hasattr(self, "remove_btn"):
            self.remove_btn.config(state="normal" if len(self.model_boxes) > 2 else "disabled")

    def setup_model_results_vertical(self, parent_frame):
        """
        Creates a vertical result area inside the given parent_frame (results_frame of each model box).
        Divides vertically into Classification, Description, and Timestamps.
        """
        parent_frame.update()  # make sure size info is updated
        parent_frame.pack_propagate(False)

        # Container for vertical subsections
        vertical_container = tk.Frame(parent_frame, bg="#1f1f1f")
        vertical_container.pack(fill="both", expand=True, padx=5, pady=5)

        # ---- Classification Section ----
        class_frame = tk.Frame(vertical_container, bg="#141414", relief="solid", bd=1)
        class_frame.pack(fill="both", expand=True, pady=(0, 5))
        tk.Label(class_frame, text="CLASSIFICATION", font=("Arial Bold", 10),
                 bg="#252525", fg="#ffffff", anchor="center", padx=10, pady=5).pack(fill="x")
        class_value = tk.Label(class_frame, text="", font=("Arial Bold", 16), bg="#141414",
                               fg="#ffffff", anchor="center", padx=10, pady=20, justify="center")
        class_value.pack(fill="both", expand=True)

        # ---- Description Section ----
        desc_frame = tk.Frame(vertical_container, bg="#141414", relief="solid", bd=1)
        desc_frame.pack(fill="both", expand=True, pady=(0, 5))
        tk.Label(desc_frame, text="DESCRIPTION", font=("Arial Bold", 10), bg="#252525",
                 fg="#ffffff", anchor="center", padx=10, pady=5).pack(fill="x")
        desc_value = tk.Label(desc_frame, text="", font=("Arial", 12), bg="#141414",
                              fg="#fcfcfc", anchor="center", padx=10, pady=20,
                              justify="center", wraplength=300)
        desc_value.pack(fill="both", expand=True)

        # ---- Timestamps Section ----
        frames_frame = tk.Frame(vertical_container, bg="#141414", relief="solid", bd=1)
        frames_frame.pack(fill="both", expand=True)
        tk.Label(frames_frame, text="TIMESTAMPS", font=("Arial Bold", 10), bg="#252525",
                 fg="#ffffff", anchor="center", padx=10, pady=5).pack(fill="x")

        frames_container = tk.Frame(frames_frame, bg="#141414")
        frames_container.pack(expand=True)
        frames_value = tk.Text(frames_container, font=("Courier New", 11), bg="#141414",
                               fg="#fcfcfc", wrap="none", relief="flat", padx=10, pady=20,
                               highlightthickness=0, takefocus=0)
        frames_value.pack(expand=True)
        frames_value.tag_configure("center", justify="center")
        frames_value.config(state="disabled")

        # Return references so you can update them later
        return {
            "class_value": class_value,
            "desc_value": desc_value,
            "frames_value": frames_value
        }


FallDetection()