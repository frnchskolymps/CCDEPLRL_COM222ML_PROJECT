import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# Load model (update path if needed)
model = load_model("final proj/best_yolo_model.keras")

# Waste categories and classification info
class_names = ['glass', 'leaf', 'metal', 'paper', 'plastic']
waste_category = {
    "glass": ("Non-biodegradable", "Recyclable"),
    "leaf": ("Biodegradable", "Non-recyclable"),
    "metal": ("Non-biodegradable", "Recyclable"),
    "paper": ("Biodegradable", "Recyclable"),
    "plastic": ("Non-biodegradable", "Recyclable"),
}

image_size = (120, 120)  # Match your model's input size

class WasteClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Classifier")
        self.root.geometry("900x700")

        self.cap = cv2.VideoCapture(0)
        self.is_capturing = True

        self.video_label = ttk.Label(root)
        self.video_label.pack()

        # Buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)

        self.capture_btn = ttk.Button(btn_frame, text="Capture", command=self.capture_image)
        self.capture_btn.grid(row=0, column=0, padx=10)

        self.upload_btn = ttk.Button(btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=1, padx=10)

        self.again_btn = ttk.Button(btn_frame, text="Capture Again", command=self.reset_capture)
        self.again_btn.grid(row=0, column=2, padx=10)
        self.again_btn['state'] = 'disabled'

        self.result_label = ttk.Label(root, text="", font=("Arial", 14), justify="left")
        self.result_label.pack(pady=10)

        self.current_frame = None
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

                # Just draw predicted bounding box without label or percentage
                input_img = self.preprocess(frame)
                preds = model.predict(input_img)

                class_probs_batch, bboxes_batch = preds
                bbox = bboxes_batch[0]

                h, w, _ = frame.shape
                x_c, y_c, bw, bh = bbox * [w, h, w, h]
                x1, y1 = int(x_c - bw / 2), int(y_c - bh / 2)
                x2, y2 = int(x_c + bw / 2), int(y_c + bh / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)

    def process_prediction(self, frame):
        h, w, _ = frame.shape
        input_img = self.preprocess(frame)
        preds = model.predict(input_img)

        class_probs_batch, bboxes_batch = preds
        class_probs = class_probs_batch[0]
        bbox = bboxes_batch[0]

        # DEBUG (optional): print prediction values
        print("Raw class probabilities:", class_probs)
        print("Bounding box:", bbox)

        class_idx = np.argmax(class_probs)
        confidence = class_probs[class_idx]

        if confidence < 0.3:
            return frame, "⚠ No confident detection."

        pred_class = class_names[class_idx]
        bio, rec = waste_category[pred_class]

        x_c, y_c, bw, bh = bbox * [w, h, w, h]
        x1, y1 = int(x_c - bw / 2), int(y_c - bh / 2)
        x2, y2 = int(x_c + bw / 2), int(y_c + bh / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{pred_class} {confidence*100:.1f}%"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        result_text = (
            f"🗑 Waste Type: {pred_class}\n"
            f"♻ Classification: {bio}\n"
            f"🔄 Recyclable: {rec}\n"
            f"✅ Accuracy: {confidence * 100:.2f}%"
        )

        return frame, result_text

    def capture_image(self):
        if self.current_frame is None:
            return
        self.is_capturing = False

        frame = self.current_frame.copy()
        result_frame, result_text = self.process_prediction(frame)

        img_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.result_label.config(text=result_text)
        self.capture_btn['state'] = 'disabled'
        self.upload_btn['state'] = 'disabled'
        self.again_btn['state'] = 'normal'

    def upload_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not filepath:
            return
        self.is_capturing = False

        img = cv2.imread(filepath)
        frame = cv2.resize(img, (640, 480))

        result_frame, result_text = self.process_prediction(frame)

        img_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.result_label.config(text=result_text)
        self.capture_btn['state'] = 'disabled'
        self.upload_btn['state'] = 'disabled'
        self.again_btn['state'] = 'normal'

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
