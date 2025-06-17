import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# Load model
model = load_model("2final_yolo_model.keras")

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
        self.root.geometry("960x740")
        self.root.configure(bg="#1e1e2f")

        self.style = ttk.Style(root)
        self.style.theme_use('clam')

        self.style.configure('TButton',
                             font=('Segoe UI', 12, 'bold'),
                             padding=10,
                             foreground='#f0f0f0',
                             background='#4b47a3',
                             borderwidth=0)
        self.style.map('TButton',
                       background=[('active', '#6c63ff')],
                       foreground=[('active', '#ffffff')])

        self.style.configure('TLabel',
                             background="#1e1e2f",
                             foreground="#e0e0e0",
                             font=('Segoe UI', 14))

        self.main_frame = ttk.Frame(root, padding=20, style="TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_frame = tk.Frame(self.main_frame, width=800, height=520, bg="#2e2e44", bd=2, relief='groove')
        self.video_frame.pack(pady=20)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg="#2e2e44")
        self.video_label.pack(expand=True)

        self.btn_frame = ttk.Frame(self.main_frame)
        self.btn_frame.pack(pady=10)

        self.upload_btn = ttk.Button(self.btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=20)

        self.result_label = ttk.Label(self.main_frame, text="", justify='left', font=('Segoe UI', 16, 'bold'), foreground="#6c63ff")
        self.result_label.pack(pady=20, anchor='w')

        self.cap = cv2.VideoCapture(0)
        self.current_frame = None

        self.update_video()

    def preprocess(self, img):
        img_resized = cv2.resize(img, image_size)
        img_norm = img_resized.astype('float32') / 255.0
        return np.expand_dims(img_norm, axis=0)

    def draw_label(self, frame, text, x1, y1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame,
                      (x1, max(y1 - text_height - 10, 0)),
                      (x1 + text_width + 10, y1),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def update_video(self):
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
            displayed_result = ""
            for i in range(len(bboxes)):
                probs = class_probs[i]
                bbox = bboxes[i]

                class_idx = np.argmax(probs)
                confidence = probs[class_idx]

                if confidence < 0.3:
                    continue

                pred_class = class_names[class_idx]
                bio, rec = waste_category[pred_class]

                x_c, y_c, bw, bh = bbox * np.array([w, h, w, h])
                x1 = max(int(x_c - bw / 2), 0)
                y1 = max(int(y_c - bh / 2), 0)
                x2 = min(int(x_c + bw / 2), w - 1)
                y2 = min(int(y_c + bh / 2), h - 1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (108, 99, 255), 3)
                label = f"{pred_class} {confidence*100:.1f}%"
                self.draw_label(frame, label, x1, y1)

                displayed_result = (
                    f"🗑 Waste Type: {pred_class}\n"
                    f"♻ Classification: {bio}\n"
                    f"🔄 Recyclable: {rec}\n"
                    f"✅ Accuracy: {confidence * 100:.2f}%"
                )

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.result_label.config(text=displayed_result if displayed_result else "⚠ No confident detection.")

        self.root.after(30, self.update_video)

    def upload_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not filepath:
            return

        img = cv2.imread(filepath)
        if img is None:
            self.result_label.config(text="⚠ Failed to load image.")
            return

        display_img = cv2.resize(img, (640, 480))
        result_frame, result_text = self.process_prediction(display_img)

        img_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.result_label.config(text=result_text)

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
        self.draw_label(frame, label, x1, y1)

        result_text = (
            f"🗑 Waste Type: {pred_class}\n"
            f"♻ Classification: {bio}\n"
            f"🔄 Recyclable: {rec}\n"
            f"✅ Accuracy: {confidence * 100:.2f}%"
        )

        return frame, result_text

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WasteClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
