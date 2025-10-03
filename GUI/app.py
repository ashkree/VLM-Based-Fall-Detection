import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from tkinterdnd2 import DND_FILES, TkinterDnD
import os

class FallDetection:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.geometry("900x850")
        self.root.title("Fall Detection")
        self.root.config(bg="#1f1f1f")
        
        self.current_file_path = None
        self.flash_job = None

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

        fall_label = tk.Label(title_frame, text="fall", font=("Arial Bold", 25),
                            bg="#000000", fg="#fe3330")
        fall_label.pack(side="left")

        detection_label = tk.Label(title_frame, text="Detection", font=("Arial Bold", 24),
                                bg="#000000", fg="#ffffff")
        detection_label.pack(side="left")

        right_frame = tk.Frame(navbar, bg="#000000")
        right_frame.pack(side="right", padx=(20, 0))

        logo_img = Image.open("./img/segfault_logo.png").resize((130, 75), Image.Resampling.LANCZOS)
        self.logo_tk = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(right_frame, image=self.logo_tk, bg="#000000")
        logo_label.pack(side="right", padx=(10, 0))

        refresh_img = Image.open("./img/refresh.png").resize((20, 20), Image.Resampling.LANCZOS)
        self.refresh_tk = ImageTk.PhotoImage(refresh_img)

        self.refresh_btn = tk.Button(
            right_frame, image=self.refresh_tk,
            bg="#000000", bd=0, activebackground="#1a1a1a",
            cursor="hand2", command=self.refresh
        )
        self.refresh_btn.pack(side="right", padx=(0, 10))


    def setup_preview_area(self):
        self.detected_label = tk.Label(self.root, text="", font=("Arial Bold", 16),
                                      bg="#1f1f1f", fg="#fe3330")
        self.detected_label.pack(side="top", pady=(20, 0))

        self.preview_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.preview_frame.pack(pady=10)

        self.preview_label = tk.Label(self.preview_frame, text="📁 .mp4 .avi .mov 📁",
                                    font=("Courier New", 12), relief="solid", 
                                    width=65, height=23, bg="#141414", fg="#cccaca")
        self.preview_label.pack(padx=10, pady=10)
        self.preview_label.drop_target_register(DND_FILES)
        self.preview_label.dnd_bind('<<Drop>>', self.handle_drop)
        self.preview_label.bind("<Button-1>", lambda e: self.open_file_locator())
        self.preview_label.bind("<Enter>", lambda e: self.preview_label.config(bg="#1a1a1a", cursor="hand2"))
        self.preview_label.bind("<Leave>", lambda e: self.preview_label.config(bg="#141414", cursor=""))

    def setup_thumbnail_area(self):
        self.thumbnail_frame = tk.Frame(self.root, bg="#1f1f1f")
        
        self.thumbnail_label = tk.Label(self.thumbnail_frame, text="", 
                                       bg="#141414", relief="solid", bd=1)

    def setup_welcome_section(self):
        self.welcome_frame = tk.Frame(self.root, bg="#1f1f1f")
        self.welcome_frame.pack(pady=(5, 15))

        self.welcome_label = tk.Label(self.welcome_frame, text="Upload or Drag a Video", 
                                     font=("Arial bold", 26),
                                     bg="#1f1f1f", fg="#cccaca")
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

        self.selected_label = tk.Label(self.file_label_frame, text="Selected:", 
                                      font=("Courier New", 10),
                                      bg="#1f1f1f", fg="#e8e8e8")
        self.selected_label.pack(side="left")

        self.filename_label = tk.Label(self.file_label_frame, text="", 
                                      font=("Courier New Bold", 10),
                                      bg="#1f1f1f", fg="#99e695")
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

        self.result_text = tk.Text(
           self.result_frame, font=("Courier New", 10),
            width=100, height=6, wrap="word",
            bg="#141414", fg="#fcfcfc", relief="solid", bd=1,
            highlightcolor="#141414", highlightbackground="#141414"
        )
        self.result_text.pack(fill="both", expand=True)
        self.result_text.insert("1.0", "")
        self.result_text.config(state="disabled")

    # ------------------------ FILE HANDLING ------------------------ #
    def open_file_locator(self):
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if file_path:
            self.handle_file(file_path)

    def handle_drop(self, event):
        file_path = event.data.strip('{}')
        self.handle_file(file_path)

    def handle_file(self, file_path):
        print(f"Selected file: {file_path}")
        self.current_file_path = file_path
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
        print("Analysing video...")
        self.detected_label.config(text="FALL DETECTED")
        
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "Fall detected on frame 4, elderly fell on their back.")
        self.result_text.config(state="disabled")

        self.flash_detected_label(True)

    def flash_detected_label(self, state):
        color = "#fe3330" if state else "#ffffff"
        self.detected_label.config(fg=color)
        self.flash_job = self.root.after(250, self.flash_detected_label, not state)

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
        
        self.current_file_path = None
        self.filename_label.config(text="")
        self.detected_label.config(text="", fg="#fe3330")
 
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.config(state="disabled")

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