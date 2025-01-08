# 🧩 Real-Time Object Detection + TTS + Gradio UI

## 📄 **Project Overview**
This project is a **real-time object detection and text-to-speech system** that:
- Uses **OpenCV** to capture live video from the webcam.
- Detects objects using a **pretrained object detection model**.
- Converts detected object names into **speech output** using a text-to-speech (TTS) model.
- Provides a **Gradio UI** to start/stop the detection process and display results.

---

## 🛠️ **1. Imports and Model Loading**
```python
import warnings
warnings.filterwarnings("ignore")

import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection, VitsModel, AutoTokenizer
import torch
import gradio as gr
import threading
import time
import sounddevice as sd
import numpy as np
```
### 🔍 **Explanation:**
- `cv2` (OpenCV) — For accessing the webcam and processing video frames.
- `transformers` — For loading object detection and TTS models from the Hugging Face library.
- `torch` — For handling deep learning models.
- `gradio` — To create an interactive web-based UI for users.
- `threading` — To run tasks (like video capture and object detection) simultaneously.
- `sounddevice` — For playing audio output.
- `numpy` — For handling numerical data (like audio waveforms).

---

## 🚀 **2. Loading Pretrained Models**
```python
save_directory = "./local_model"
processor = AutoImageProcessor.from_pretrained(save_directory)
model = AutoModelForObjectDetection.from_pretrained(save_directory)

tts_model_path = "./mms-tts-eng/model"
tts_model = VitsModel.from_pretrained(tts_model_path)
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_path)
```
### 🔍 **Explanation:**
- **Object Detection Model** (`AutoModelForObjectDetection`): Detects objects in live video frames.
- **Text-to-Speech Model** (`VitsModel`): Converts detected object names into speech.
- **Tokenizer**: Processes text for the TTS model.

---

## 📹 **3. Capturing Live Video from the Webcam**
```python
stop_event = threading.Event()
frame = None

def live_camera_feed(camera_index):
    global frame
    cap = cv2.VideoCapture(camera_index)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    while not stop_event.is_set():
        success, img = cap.read()
        if success:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    cap.release()
```
### 🔍 **Explanation:**
- Opens the webcam using OpenCV.
- Continuously captures frames until `stop_event` is triggered.
- Converts frames from **BGR to RGB** (because OpenCV uses BGR but models require RGB).

---

## 📦 **4. Object Detection Logic**
```python
def detect_objects():
    global frame
    while not stop_event.is_set():
        if frame is not None:
            img = frame.copy()
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([img.shape[:2]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

            detected_objects = []

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.5 and len(detected_objects) < 2:
                    box = [int(i) for i in box.tolist()]
                    label_name = model.config.id2label[label.item()]
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(img, f"{label_name} {int(score * 100)}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_objects.append(f"{label_name} {int(score * 100)}%")

            if detected_objects:
                for obj in detected_objects:
                    tts_output = text_to_speech(obj)
                    play_audio(tts_output)

            yield img, detected_objects

            del inputs, outputs, results, img
            torch.cuda.empty_cache()
            time.sleep(2.5)
```
### 🔍 **Explanation:**
- Detects objects in the live frame using a pretrained model.
- Filters objects with **confidence score > 50%** and limits detection to **2 objects per frame**.
- Draws bounding boxes and labels on the detected objects.
- Calls **text-to-speech** for each detected object and plays the audio.
- **Clears GPU cache** to prevent memory issues.

---

## 🎙️ **5. Text-to-Speech Conversion**
```python
def text_to_speech(text):
    inputs = tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs).waveform
    return output

def play_audio(output):
    audio_data = output.squeeze().numpy()
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=1)
    sd.play(audio_data, samplerate=22050)
    sd.wait()
```
### 🔍 **Explanation:**
- **`text_to_speech(text)`**: Converts text to a speech waveform using the TTS model.
- **`play_audio(output)`**: Plays the generated audio using the `sounddevice` library.

---

## 🎛️ **6. Gradio Interface Setup**
```python
camera_options = ["Camera 0", "Camera 1", "Camera 2"]

with gr.Blocks() as iface:
    gr.Markdown("# Object Detection with Webcam and Text-to-Speech")
    camera_option = gr.Dropdown(choices=camera_options, label="Select Camera", value="Camera 0")
    start_button = gr.Button("Start Detection")
    stop_button = gr.Button("Stop Detection")
    live_output = gr.Image(type="numpy", label="Live Webcam")
    detection_output = gr.Textbox(label="Detection Results")

    def update_output(camera_option):
        for frame, detected_objects in start_detection(camera_option):
            yield frame, "\n".join(detected_objects)

    start_button.click(update_output, inputs=camera_option, outputs=[live_output, detection_output])
    stop_button.click(stop_detection)

iface.launch(share=True)
```
### 🔍 **Explanation:**
- **Dropdown** to select the camera.
- **Buttons** to start/stop detection.
- **Image display** for the live webcam feed.
- **Textbox** to display detection results.
- Launches the Gradio app with `iface.launch(share=True)` to share the app publicly.

---

## 🔥 **Summary of the Workflow**
| **Step**                     | **Description**                                                |
|------------------------------|----------------------------------------------------------------|
| **1. Camera Feed**            | Opens the webcam and captures live frames.                    |
| **2. Object Detection**       | Detects objects in each frame using a pretrained model.        |
| **3. Text-to-Speech**         | Converts detected objects' names into speech.                  |
| **4. Play Audio**             | Plays the generated speech using `sounddevice`.                |
| **5. Gradio UI**              | Provides a web interface for selecting the camera and viewing results. |

---

Would you like to optimize this further or add any more features?
