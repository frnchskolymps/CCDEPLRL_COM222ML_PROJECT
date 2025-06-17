import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# Load model (update path if needed)
model = load_model("final proj app/best_yolo_model.keras")

class_names = ['glass', 'leaf', 'metal', 'paper', 'plastic']
waste_category = {
    "glass": ("Non-biodegradable", "Recyclable"),
    "leaf": ("Biodegradable", "Non-recyclable"),
    "metal": ("Non-biodegradable", "Recyclable"),
    "paper": ("Biodegradable", "Recyclable"),
    "plastic": ("Non-biodegradable", "Recyclable"),
}

image_size = (120, 120)

class WasteClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🗑️ Waste Classifier")
        self.root.geometry("900x720")
        self.root.configure(bg="#2e2e2e")  # Dark background

        # Setup ttk style for modern look
        self.style = ttk.Style(root)
        self.style.theme_use('clam')  # More modern theme on most platforms

        # Customize styles for buttons
        self.style.configure('TButton',
                             font=('Segoe UI', 12),
                             padding=8,
                             foreground='#f0f0f0',
                             background='#444444')
        self.style.map('TButton',
                       background=[('active', '#6c63ff')],
                       foreground=[('active', '#ffffff')])

        # Label style
        self.style.configure('TLabel',
                             background="#2e2e2e",
                             foreground="#e0e0e0",
                             font=('Segoe UI', 14))

        # Frame for video + result
        self.main_frame = ttk.Frame(root, padding=15, style="TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Video display frame with border
        self.video_frame = ttk.Frame(self.main_frame, width=700, height=480, relief='ridge')
        self.video_frame.pack(pady=15)
        self.video_frame.pack_propagate(False)  # Prevent resizing

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(expand=True)

        # Button frame
        self.btn_frame = ttk.Frame(self.main_frame)
        self.btn_frame.pack(pady=10)

        self.capture_btn = ttk.Button(self.btn_frame, text="Capture", command=self.capture_image)
        self.capture_btn.grid(row=0, column=0, padx=15)

        self.upload_btn = ttk.Button(self.btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=1, padx=15)

        self.again_btn = ttk.Button(self.btn_frame, text="Capture Again", command=self.reset_capture)
        self.again_btn.grid(row=0, column=2, padx=15)
        self.again_btn['state'] = 'disabled'

        # Result label with padding and wrap
        self.result_label = ttk.Label(self.main_frame, text="", justify='left', font=('Segoe UI', 16, 'bold'))
        self.result_label.pack(pady=20, anchor='w')

        # Loading spinner as progressbar
        self.progress = ttk.Progressbar(self.main_frame, mode='indeterminate', length=250)
        self.progress.pack(pady=10)
        self.progress.pack_forget()

        # Webcam capture
        self.cap = cv2.VideoCapture(0)
        self.is_capturing = True
        self.current_frame = None

        # Start updating video feed
        self.update_video()

    def preprocess(self, img):
        img_resized = cv2.resize(img, image_size)
        img_norm = img_resized.astype('float32') / 255.0
        return np.expand_dims(img_norm, axis=0)

    def update_video(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_img = self.preprocess(rgb_frame)

                preds = model.predict(input_img)
                class_probs_batch, bboxes_batch = preds

                class_probs = class_probs_batch[0]
                bboxes = bboxes_batch[0]

                if len(bboxes.shape) == 1:
                    bboxes = np.expand_dims(bboxes, axis=0)
                    class_probs = np.expand_dims(class_probs, axis=0)

                h, w, _ = frame.shape
                for i in range(len(bboxes)):
                    probs = class_probs[i]
                    bbox = bboxes[i]

                    class_idx = np.argmax(probs)
                    confidence = probs[class_idx]

                    if confidence < 0.3:
                        continue

                    pred_class = class_names[class_idx]

                    x_c, y_c, bw, bh = bbox * np.array([w, h, w, h])
                    x1 = max(int(x_c - bw / 2), 0)
                    y1 = max(int(y_c - bh / 2), 0)
                    x2 = min(int(x_c + bw / 2), w - 1)
                    y2 = min(int(y_c + bh / 2), h - 1)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (108, 99, 255), 3)  # Use purple-ish color

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)

    def process_prediction(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape
        input_img = self.preprocess(rgb_frame)
        preds = model.predict(input_img)

        class_probs_batch, bboxes_batch = preds
        class_probs = class_probs_batch[0]
        bbox = bboxes_batch[0]

        class_idx = np.argmax(class_probs)
        confidence = class_probs[class_idx]

        if confidence < 0.3:
            return frame, "⚠ No confident detection."

        pred_class = class_names[class_idx]
        bio, rec = waste_category[pred_class]

        x_c, y_c, bw, bh = bbox * np.array([w, h, w, h])
        x1, y1 = max(int(x_c - bw / 2), 0), max(int(y_c - bh / 2), 0)
        x2, y2 = min(int(x_c + bw / 2), w - 1), min(int(y_c + bh / 2), h - 1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (108, 99, 255), 4)
        label = f"{pred_class} {confidence*100:.1f}%"
        cv2.putText(frame, label, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (108, 99, 255), 3)

        result_text = (
            f"🗑 Waste Type: {pred_class}\n"
            f"♻ Classification: {bio}\n"
            f"🔄 Recyclable: {rec}\n"
            f"✅ Accuracy: {confidence * 100:.2f}%"
        )

        return frame, result_text

    def show_loading_then_result(self, frame, result_text):
        self.progress.pack(pady=10)
        self.progress.start(10)

        self.root.after(2000, lambda: self.display_result_after_loading(frame, result_text))

    def display_result_after_loading(self, frame, result_text):
        self.progress.stop()
        self.progress.pack_forget()
        self.display_result(frame, result_text)

    def display_result(self, frame, result_text):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.result_label.config(text=result_text)
        self.capture_btn['state'] = 'disabled'
        self.upload_btn['state'] = 'disabled'
        self.again_btn['state'] = 'normal'

    def capture_image(self):
        if self.current_frame is None:
            return
        self.is_capturing = False

        frame = self.current_frame.copy()
        result_frame, result_text = self.process_prediction(frame)

        self.show_loading_then_result(result_frame, result_text)

    def upload_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not filepath:
            return
        self.is_capturing = False

        img = cv2.imread(filepath)
        if img is None:
            self.result_label.config(text="⚠ Failed to load image.")
            return

        display_img = cv2.resize(img, (640, 480))
        result_frame, result_text = self.process_prediction(display_img)

        self.show_loading_then_result(result_frame, result_text)

    def reset_capture(self):
        self.result_label.config(text="")
        self.capture_btn['state'] = 'normal'
        self.upload_btn['state'] = 'normal'
        self.again_btn['state'] = 'disabled'
        self.is_capturing = True

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = WasteClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
