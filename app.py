import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2, re, json, os, threading, itertools
from model_utils import load, prep_message, analyze_video
from huggingface_hub import snapshot_download


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
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        path = "./qwen2.5vl_snapshot"

        self.root = tk.Tk()
        self.root.geometry("900x880")
        self.root.title("Fall Detection")
        self.root.config(bg="#1f1f1f")

        if not os.path.isdir(path):
            snapshot_download(model_name, local_dir=path)

        self.model, self.processor = load()

        self.current_file_path = None
        self.spinner_running = False
        self.spinner_job = None
        self.video_fps = 30

        self.setup_navbar()
        self.setup_preview_area()
        self.setup_thumbnail_area()
        self.setup_welcome_section()
        self.setup_file_label()
        self.setup_analyse_button()
        self.setup_result_area()

        self.root.mainloop()

    # ------------------------ UI SETUP ------------------------ #
    def setup_navbar(self):
        navbar = tk.Frame(self.root, bg="#000000", height=120)
        navbar.pack(fill="x")

        title_frame = tk.Frame(navbar, bg="#000000")
        title_frame.pack(side="left", padx=20)
        tk.Label(title_frame, text="fall", font=("Arial Bold", 25),
                 bg="#000000", fg="#fe3330").pack(side="left")
        tk.Label(title_frame, text="Detection", font=("Arial Bold", 24),
                 bg="#000000", fg="#ffffff").pack(side="left")

        directory_path = os.path.dirname(__file__)
        refresh_path = os.path.join(directory_path, 'img/refresh.png')
        if os.path.exists(refresh_path):
            refresh_img = Image.open(refresh_path).resize((20, 20), Image.Resampling.LANCZOS)
            self.refresh_tk = ImageTk.PhotoImage(refresh_img)
            tk.Button(navbar, image=self.refresh_tk, bg="#000000", bd=0,
                      activebackground="#1a1a1a", cursor="hand2", command=self.refresh
                      ).pack(side="right", padx=(0, 15))

    def setup_preview_area(self):
        self.detected_label = tk.Label(self.root, text="", font=("fixedsys", 22),
                                       bg="#1f1f1f", fg="#fe3330")
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
        tk.Label(self.welcome_frame, text="Upload a Video", font=("Arial bold", 26),
                 bg="#1f1f1f", fg="#cccaca").pack()
        self.upload_btn = tk.Button(
            self.root, text="Upload", font=("Courier New Bold", 14),
            bg="#fe3330", fg="#ffffff", activebackground="#b32028",
            bd=0, padx=15, pady=5, cursor="hand2", command=self.open_file_locator
        )
        self.upload_btn.pack(padx=10, pady=(30, 0))

    def setup_file_label(self):
        self.file_label_frame = tk.Frame(self.root, bg="#1f1f1f")
        tk.Label(self.file_label_frame, text="Selected:", font=("Courier New", 10),
                 bg="#1f1f1f", fg="#e8e8e8").pack(side="left")
        self.filename_label = tk.Label(self.file_label_frame, text="",
                                       font=("Courier New Bold", 10),
                                       bg="#1f1f1f", fg="#99e695")
        self.filename_label.pack(side="left")

    def setup_analyse_button(self):
        self.btn_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.analyse_btn = tk.Button(
            self.btn_frame, text="Analyse", font=("Courier New Bold", 12),
            bg="#fe3330", fg="white", activebackground="#b32028",
            bd=0, padx=15, pady=5, cursor="hand2", command=self.analyse_video
        )
        self.analyse_btn.pack(side="left", padx=15)
        self.change_video_btn = tk.Button(
            self.btn_frame, text="Change Video", font=("Courier New", 12),
            bg="#333333", fg="#b8b8b8", activebackground="#555555",
            bd=0, padx=15, pady=5, cursor="hand2", command=self.open_file_locator
        )
        self.change_video_btn.pack(side="left")

    def setup_result_area(self):
        self.result_frame = tk.Frame(self.root, bg="#1f1f1f")
        subsections_container = tk.Frame(self.result_frame, bg="#1f1f1f")
        subsections_container.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        # Classification
        class_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1, width=150)
        class_frame.pack(side="left", fill="both", padx=(0, 5))
        tk.Label(class_frame, text="CLASSIFICATION", font=("Arial Bold", 10),
                 bg="#252525", fg="#ffffff").pack(fill="x")
        self.class_value = tk.Label(class_frame, text="", font=("Arial Bold", 16),
                                    bg="#141414", fg="#ffffff")
        self.class_value.pack(fill="both", expand=True)

        # Description
        desc_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1)
        desc_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        tk.Label(desc_frame, text="DESCRIPTION", font=("Arial Bold", 10),
                 bg="#252525", fg="#ffffff").pack(fill="x")
        self.desc_value = tk.Label(desc_frame, text="", font=("Arial", 12),
                                   bg="#141414", fg="#fcfcfc", wraplength=400)
        self.desc_value.pack(fill="both", expand=True)

        # Fall timestamps
        frames_frame = tk.Frame(subsections_container, bg="#141414", relief="solid", bd=1, width=150)
        frames_frame.pack(side="left", fill="both")
        tk.Label(frames_frame, text="FALL TIMESTAMPS", font=("Arial Bold", 10),
                 bg="#252525", fg="#ffffff").pack(fill="x")
        self.frames_value = tk.Text(frames_frame, font=("Courier New", 11),
                                    bg="#141414", fg="#fcfcfc", wrap="none",
                                    relief="flat", padx=10, pady=25)
        self.frames_value.pack(expand=True)
        self.frames_value.config(state="disabled")

    # ------------------------ FILE + ANALYSIS ------------------------ #
    def open_file_locator(self):
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if file_path:
            self.current_file_path = file_path
            self.filename_label.config(text=os.path.basename(file_path))
            self.show_thumbnail(file_path)
            self.show_analyse_btn()

    def show_thumbnail(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((650, 400), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            self.thumbnail_label.configure(image=imgtk)
            self.thumbnail_label.image = imgtk
            self.thumbnail_label.pack(padx=10, pady=10)
            self.thumbnail_frame.pack(pady=10)

    def show_analyse_btn(self):
        for w in [self.preview_frame, self.welcome_frame, self.upload_btn]:
            try:
                w.pack_forget()
            except Exception:
                pass
        if not self.file_label_frame.winfo_ismapped():
            self.file_label_frame.pack()
        if not self.btn_frame.winfo_ismapped():
            self.btn_frame.pack(pady=(20, 5))
        if not self.result_frame.winfo_ismapped():
            self.result_frame.pack(side="bottom", padx=10, pady=10, fill="both", expand=True)

    def analyse_video(self):
        if not self.current_file_path:
            return
        self.class_value.config(text="Processing...")
        self.desc_value.config(text="Analysing video on GPU, please wait...")
        self.root.update_idletasks()
        self.start_spinner()
        thread = threading.Thread(target=self.run_analysis_in_thread, args=(self.current_file_path,))
        thread.start()

    # ------------------------ SPINNER ------------------------ #
    def start_spinner(self):
        self.spinner_running = True
        spinner_chars = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

        def animate():
            if not self.spinner_running:
                return
            self.class_value.config(text=f"Processing {next(spinner_chars)}")
            self.spinner_job = self.root.after(150, animate)
        animate()

    def stop_spinner(self):
        self.spinner_running = False
        if self.spinner_job:
            self.root.after_cancel(self.spinner_job)
            self.spinner_job = None

    # ------------------------ ANALYSIS THREAD ------------------------ #
    def run_analysis_in_thread(self, video_path):
        try:
            message = prep_message(video_path, SYSTEM_PROMPT)
            result = analyze_video(self.model, self.processor, message)
            result = json.loads(re.sub(r'^```json\n|```$', '', result))
            result["fall_timestamps"] = [result["fall_start"], result["fall_end"]]
        except Exception as e:
            print(f"[ERROR] analysis failed: {e}")
            result = {"class": "UNKNOWN", "reasoning": "Analysis failed.", "fall_timestamps": []}
        self.root.after(0, lambda: self.display_analysis_result(result))

    def display_analysis_result(self, result):
        self.stop_spinner()
        cls = result.get("class", "UNKNOWN")
        desc = result.get("reasoning", "")
        timestamps = result.get("fall_timestamps", [])
        self.class_value.config(text=cls)
        self.desc_value.config(text=desc)
        self.frames_value.config(state="normal")
        self.frames_value.delete("1.0", "end")
        for t in timestamps:
            self.frames_value.insert("end", f"{t:.2f}s\n")
        self.frames_value.config(state="disabled")
        self.detected_label.config(text="FALL DETECTED!" if cls == "FALL" else "NO FALL DETECTED")

    def refresh(self):
        self.stop_spinner()
        self.current_file_path = None
        self.filename_label.config(text="")
        self.detected_label.config(text="")
        self.class_value.config(text="")
        self.desc_value.config(text="")
        self.frames_value.config(state="normal")
        self.frames_value.delete("1.0", "end")
        self.frames_value.config(state="disabled")
        for w in [self.thumbnail_frame, self.btn_frame, self.result_frame, self.file_label_frame]:
            if w.winfo_ismapped():
                w.pack_forget()
        self.preview_frame.pack(pady=10)
        self.welcome_frame.pack(pady=(5, 15))
        self.upload_btn.pack(padx=15, pady=5)


if __name__ == "__main__":
    FallDetection()
