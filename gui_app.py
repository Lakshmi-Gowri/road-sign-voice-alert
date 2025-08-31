import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox, ttk
import torch
from PIL import Image, ImageTk
import cv2
from voice_alert.tts_engine import speak_alert, set_language, get_available_languages, get_current_language
from road_sign_utils.label_map import label_map
import threading
import time
import os
import subprocess
import platform
from tensorflow.keras.models import load_model
from sign_detection_logs.sign_logger import SignDetectionLogger
# pyright: ignore[reportMissingImports]

model = load_model('my_model.keras')
print("Custom Road sign model loaded successfully")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Initialize logger
logger = SignDetectionLogger()

# Global variables for camera
camera_active = False
cap = None
last_alert_time = 0
ALERT_COOLDOWN = 3  # seconds between alerts
video_active = False

def on_language_change(event=None):
    """Handle language selection change"""
    selected_language = language_var.get()
    set_language(selected_language)
    app.result_label.config(text=f"🌐 Language changed to: {selected_language}")
    print(f"Language changed to: {selected_language}")

def view_logs():
    """Open the log file in the default application"""
    try:
        log_file_path = logger.get_log_file_path()
        
        if not os.path.exists(log_file_path):
            messagebox.showinfo("No Logs", "No detection logs found for today.")
            return
        
        # Get file size for confirmation
        file_size = os.path.getsize(log_file_path)
        if file_size == 0:
            messagebox.showinfo("Empty Log", "Log file exists but is empty.")
            return
        
        # Open file with default application based on OS
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', log_file_path])
        elif platform.system() == 'Windows':  # Windows
            os.startfile(log_file_path)
        else:  # Linux
            subprocess.call(['xdg-open', log_file_path])
        
        # Show stats in a message box
        stats = logger.get_detection_stats()
        stats_message = f"""
📊 Today's Detection Statistics:

🔍 Total Detections: {stats['total_detections']}
🚏 Unique Signs: {stats['unique_signs']}
📈 Average Confidence: {stats['avg_confidence']:.2%}

📁 Log file opened in default application.
        """
        messagebox.showinfo("Log Statistics", stats_message.strip())
        
    except Exception as e:
        messagebox.showerror("Error", f"Could not open log file: {str(e)}")

def show_detection_stats():
    """Display detection statistics in a popup window"""
    try:
        stats = logger.get_detection_stats()
        
        # Create stats window
        stats_window = tk.Toplevel(app)
        stats_window.title("Detection Statistics")
        stats_window.geometry("400x300")
        stats_window.configure(bg="#f0f0f0")
        
        # Title
        title_label = tk.Label(stats_window, text="📊 Detection Statistics", 
                              font=("Arial", 16, "bold"), bg="#f0f0f0", fg="darkblue")
        title_label.pack(pady=10)
        
        # Stats content
        stats_text = f"""
🔍 Total Detections: {stats['total_detections']}
🚏 Unique Signs Detected: {stats['unique_signs']}
📈 Average Confidence: {stats['avg_confidence']:.2%}

📱 Detection Sources:
"""
        
        for source, count in stats.get('sources', {}).items():
            stats_text += f"  • {source}: {count}\n"
        
        stats_text += "\n🌐 Languages Used:\n"
        for language, count in stats.get('languages', {}).items():
            stats_text += f"  • {language}: {count}\n"
        
        stats_label = tk.Label(stats_window, text=stats_text, 
                              font=("Arial", 11), bg="#f0f0f0", 
                              justify=tk.LEFT, anchor="w")
        stats_label.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Close button
        close_btn = tk.Button(stats_window, text="Close", 
                             command=stats_window.destroy,
                             font=("Arial", 10), bg="#ff4444", fg="white")
        close_btn.pack(pady=10)
        
    except Exception as e:
        messagebox.showerror("Error", f"Could not display statistics: {str(e)}")

# Replace the detect_image function (around line 36)
def detect_image():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    
    # Display the uploaded image immediately before processing
    try:
        # Load and display the original image first
        display_img = Image.open(filepath)
        
        # Resize to 400x400 while maintaining aspect ratio
        display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        
        # Create a new 400x400 image with white background
        preview_img = Image.new("RGB", (400, 400), "white")
        
        # Calculate position to center the image
        x = (400 - display_img.width) // 2
        y = (400 - display_img.height) // 2
        
        # Paste the resized image onto the center
        preview_img.paste(display_img, (x, y))
        
        # Convert to PhotoImage and display
        img_tk = ImageTk.PhotoImage(preview_img)
        panel.config(image=img_tk)
        panel.image = img_tk
        
        # Update result label to show image is loaded
        app.result_label.config(text="Image loaded. Processing...", 
                               fg="blue", font=("Arial", 12))
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        return
    
    # Show processing message
    btn.config(text="Processing...", state="disabled")
    progress_bar.grid(row=5, column=0, pady=10, sticky="ew")
    progress_bar.start()
    
    # Run detection in separate thread to keep GUI responsive
    threading.Thread(target=process_image, args=(filepath,), daemon=True).start()

def upload_video():
    """Handle video file upload and processing"""
    filepath = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("All files", "*.*")
        ]
    )
    if not filepath:
        return
    
    # Show processing message
    btn.config(state="disabled")
    camera_btn.config(state="disabled")
    video_btn.config(text="Processing Video...", state="disabled")
    app.result_label.config(text="Loading video file...", fg="blue")
    
    # Start video processing in separate thread
    threading.Thread(target=process_video, args=(filepath,), daemon=True).start()

def process_video(filepath):
    """Process uploaded video file frame by frame"""
    global video_active
    video_active = True
    
    try:
        # Open video file
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            app.after(0, lambda: messagebox.showerror("Error", "Could not open video file"))
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        app.after(0, lambda: app.result_label.config(
            text=f"📹 Processing Video: {total_frames} frames at {fps:.1f} FPS", 
            fg="blue"
        ))
        
        frame_count = 0
        
        while video_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame for detection
            detection_result = process_frame(frame)
            
            # Draw detection results on frame
            if detection_result:
                class_name, confidence, bbox = detection_result
                
                # Log the detection
                logger.log_detection(
                    sign_name=class_name,
                    confidence=confidence,
                    source="Video",
                    language=get_current_language(),
                    location="Video File"
                )
                
                # Draw bounding box
                if bbox:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                    
                    # Draw label with background
                    label = f"{class_name}: {confidence:.2%}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                                (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Trigger voice alert (with cooldown)
                current_time = time.time()
                if current_time - last_alert_time > ALERT_COOLDOWN:
                    threading.Thread(target=speak_alert, 
                                   args=(f'{class_name} detected in video.',), 
                                   daemon=True).start()
                    globals()['last_alert_time'] = current_time
                
                # Update GUI with detection
                app.after(0, update_video_gui, frame, class_name, confidence, frame_count, total_frames)
            else:
                # Update GUI without detection
                app.after(0, update_video_gui, frame, "", 0.0, frame_count, total_frames)
            
            # Control playback speed (adjust as needed)
            time.sleep(0.1)  # ~10 FPS playback
        
        # Cleanup
        cap.release()
        app.after(0, video_processing_complete)
        
    except Exception as e:
        app.after(0, lambda: messagebox.showerror("Error", f"Error processing video: {str(e)}"))
        app.after(0, video_processing_complete)

def update_video_gui(frame, class_name="", confidence=0.0, frame_num=0, total_frames=0):
    """Update GUI with video frame and detection results"""
    try:
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set display size for video
        target_width, target_height = 640, 480
        
        # Maintain aspect ratio while scaling
        frame_height, frame_width = frame_rgb.shape[:2]
        aspect_ratio = frame_width / frame_height
        
        if target_width / target_height > aspect_ratio:
            display_height = target_height
            display_width = int(display_height * aspect_ratio)
        else:
            display_width = target_width
            display_height = int(display_width / aspect_ratio)
        
        frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
        
        img_pil = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update panel
        panel.config(image=img_tk, text="", compound=tk.CENTER)
        panel.image = img_tk
        
        # Update result label with detection and progress
        progress_text = f"Frame {frame_num}/{total_frames} ({(frame_num/total_frames)*100:.1f}%)"
        
        if class_name:
            app.result_label.config(
                text=f"🔍 Video Detection: {class_name}\n📊 Confidence: {confidence:.2%}\n📹 {progress_text}", 
                fg="green", font=("Arial", 12, "bold")
            )
        else:
            app.result_label.config(
                text=f"📹 Processing Video...\n{progress_text}", 
                fg="blue", font=("Arial", 12)
            )
            
    except Exception as e:
        print(f"Error updating video GUI: {e}")

def stop_video():
    """Stop video processing"""
    global video_active
    video_active = False
    video_processing_complete()

def video_processing_complete():
    """Reset GUI after video processing"""
    btn.config(state="normal")
    camera_btn.config(state="normal")
    video_btn.config(text="Upload Video", state="normal")
    app.result_label.config(text="Video processing completed", fg="green")

def start_camera():
    global camera_active, cap
    
    if not camera_active:
        # Start camera with higher resolution
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        # Set camera resolution for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        camera_active = True
        camera_btn.config(text="Stop Camera", bg="#f44336")
        btn.config(state="disabled")
        
        # Start camera thread
        threading.Thread(target=camera_loop, daemon=True).start()
    else:
        # Stop camera
        stop_camera()

def stop_camera():
    global camera_active, cap
    
    camera_active = False
    if cap:
        cap.release()
        cap = None
    
    camera_btn.config(text="Start Camera", bg="#2196F3")
    btn.config(state="normal")
    app.result_label.config(text="Camera stopped", fg="red")

def camera_loop():
    global camera_active, cap, last_alert_time
    
    while camera_active and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame for detection
        detection_result = process_frame(frame)
        
        # Draw detection results on frame
        if detection_result:
            class_name, confidence, bbox = detection_result
            
            # Log the detection
            logger.log_detection(
                sign_name=class_name,
                confidence=confidence,
                source="Camera",
                language=get_current_language(),
                location="Live Camera"
            )
            
            # Draw bounding box
            if bbox:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2%}"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Trigger voice alert with cooldown
            current_time = time.time()
            if current_time - last_alert_time > ALERT_COOLDOWN:
                threading.Thread(target=speak_alert, 
                               args=(f'{class_name} detected. Please be careful.',), 
                               daemon=True).start()
                last_alert_time = current_time
            
            # Update GUI
            app.after(0, update_camera_gui, frame, class_name, confidence)
        else:
            # Update GUI with no detection
            app.after(0, update_camera_gui, frame, "", 0.0)
        
        time.sleep(0.1)  # Limit FPS to ~10
    
    # Cleanup when loop ends
    if cap:
        cap.release()

def process_frame(frame):
    try:
        # Resize frame for processing
        frame_resized = cv2.resize(frame, (32, 32))
        
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        frame_normalized = frame_gray / 255.0
        
        # Reshape for model
        frame_input = frame_normalized.reshape(1, 32, 32, 1)
        
        # Make prediction
        prediction = model.predict(frame_input, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Only return detection if confidence is above threshold
        if confidence > 0.7:  # Adjust threshold as needed
            class_name = label_map.get(predicted_class, f"Class {predicted_class}")
            
            # Create a simple bounding box (center of frame)
            h, w = frame.shape[:2]
            bbox = (w//4, h//4, 3*w//4, 3*h//4)
            
            return class_name, confidence, bbox
        
        return None
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def update_camera_gui(frame, class_name="", confidence=0.0):
    try:
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set LARGE fixed camera display size for better visibility
        target_width, target_height = 640, 480
        
        # Maintain aspect ratio while scaling to target size
        frame_height, frame_width = frame_rgb.shape[:2]
        aspect_ratio = frame_width / frame_height
        
        if target_width / target_height > aspect_ratio:
            # Fit to height
            display_height = target_height
            display_width = int(display_height * aspect_ratio)
        else:
            # Fit to width
            display_width = target_width
            display_height = int(display_width / aspect_ratio)
        
        frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
        
        img_pil = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update panel - clear text and show large image
        panel.config(image=img_tk, text="", compound=tk.CENTER)
        panel.image = img_tk  # Keep a reference
        
        # Update result label
        if class_name:
            app.result_label.config(
                text=f"🔍 Live Detection: {class_name}\n📊 Confidence: {confidence:.2%}", 
                fg="green", font=("Arial", 14, "bold")
            )
        else:
            app.result_label.config(
                text="📹 Live Camera - Scanning for road signs...", 
                fg="blue", font=("Arial", 12)
            )
            
    except Exception as e:
        print(f"Error updating camera GUI: {e}")
        panel.config(text=f"Error: {str(e)}", image="", fg="red")

def process_image(filepath):
    try:
        # Load and preprocess image for your Keras model
        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to match model input (32x32 based on your notebook)
        img_resized = cv2.resize(img_rgb, (32, 32))

        # Convert to grayscale (your model uses grayscale input)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Normalize pixel values
        img_normalized = img_gray / 255.0
        
        # Reshape for model input (add batch dimension and channel dimension)
        img_input = img_normalized.reshape(1, 32, 32, 1)
        
        # Make prediction
        prediction = model.predict(img_input)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        detection_text = ""
        if predicted_class in label_map:
            class_name = label_map[predicted_class]
            
            # Log the detection
            logger.log_detection(
                sign_name=class_name,
                confidence=confidence,
                source="Image",
                language=get_current_language(),
                location=f"File: {os.path.basename(filepath)}"
            )
            
            speak_alert(f'{class_name} detected. Please be careful.')
            print(f"Detection: {class_name} with confidence {confidence:.4f}")
            detection_text = class_name
        else:
            speak_alert(f'Road sign detected. Please be careful.')
            print(f"Unknown road sign class {predicted_class} detected")
            detection_text = "Unknown road sign"

        # img_scaled = cv2.resize(img_resized, (220, 220), interpolation=cv2.INTER_NEAREST)

        
        # Display MUCH LARGER image with detection results (400x400 instead of 150x150)
        display_img = Image.open(filepath)
        # display_img = Image.fromarray(img_scaled)
        display_img.thumbnail((500, 500), Image.Resampling.LANCZOS)
        
        # Create final display image with detection results
        final_img = Image.new("RGB", (500, 500), "white")
        x = (500 - display_img.width) // 2
        y = (500 - display_img.height) // 2
        final_img.paste(display_img, (x, y))
        
        # Add colored border if detection found
        from PIL import ImageDraw
        draw = ImageDraw.Draw(final_img)
        border_color = "green" if confidence > 0.7 else "orange"
        
        img_left = x
        img_top = y  
        img_right = x + display_img.width
        img_bottom = y + display_img.height

        border_thickness = 4
        for i in range(border_thickness):
            draw.rectangle([img_left - i, img_top - i, img_right + i - 1, img_bottom + i - 1], 
                   outline=border_color, width=1)        
        # Add detection text overlay on the image
        if detection_text:
            from PIL import ImageFont
            try:
                # Try to use a larger font
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Add text background
            text_bbox = draw.textbbox((10, 10), f"{detection_text}\n{confidence:.2%}", font=font)
            draw.rectangle([8, 8, text_bbox[2]+4, text_bbox[3]+4], fill="black", outline=border_color)
            
            # Add text
            draw.text((10, 10), f"{detection_text}\n{confidence:.2%}", 
                     fill="white", font=font)
        
        img_tk = ImageTk.PhotoImage(final_img)
        
        # Update GUI in main thread
        app.after(0, update_gui, img_tk, detection_text, confidence)
        
        print("Image processed successfully")
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        app.after(0, show_error, error_msg)

def update_gui(img_tk, detection_text="", confidence=0.0):
    panel.config(image=img_tk)
    panel.image = img_tk
    
    # Update result label with detection information
    if detection_text:
        app.result_label.config(text=f"Detection: {detection_text}\nConfidence: {confidence:.2%}", 
                              fg="green", font=("Arial", 14, "bold"))
    else:
        app.result_label.config(text="No detection found", fg="red", font=("Arial", 12))
    
    btn.config(text="Select Image", state="normal")
    progress_bar.stop()
    progress_bar.grid_remove()

def show_error(error_msg):
    messagebox.showerror("Error", error_msg)
    btn.config(text="Select Image", state="normal")
    progress_bar.stop()
    progress_bar.grid_remove()

def on_closing():
    global camera_active, video_active
    if camera_active:
        stop_camera()
    if video_active:
        stop_video()
    app.destroy()

# GUI Setup
app = tk.Tk()
app.title("Road Sign Recognition with Voice Alert")
app.geometry("900x1100")  # Increased height for new buttons
app.configure(background="#bad5ff")
app.resizable(True, True)
app.protocol("WM_DELETE_WINDOW", on_closing)

# Configure grid weights for proper scaling
app.grid_rowconfigure(2, weight=1)
app.grid_columnconfigure(0, weight=1)

# Add title label
title_label = tk.Label(app, text="Road Sign Recognition with Voice Alert", 
                      font=("Arial", 16, "bold"), bg="#bad5ff", fg="darkblue")
title_label.grid(row=0, column=0, pady=10, sticky="ew")

# Language selection frame
language_frame = tk.Frame(app, bg="#bad5ff")
language_frame.grid(row=1, column=0, pady=10, sticky="ew")

language_label = tk.Label(language_frame, text="🌐 Language:", 
                         font=("Arial", 12, "bold"), bg="#bad5ff", fg="darkblue")
language_label.pack(side=tk.LEFT, padx=5)

# Language dropdown
language_var = tk.StringVar(value="English")
language_dropdown = ttk.Combobox(language_frame, textvariable=language_var, 
                                values=get_available_languages(), 
                                state="readonly", width=15, font=("Arial", 10))
language_dropdown.pack(side=tk.LEFT, padx=5)
language_dropdown.bind('<<ComboboxSelected>>', on_language_change)

# Image display panel
panel = tk.Label(app, bg="white", width=80, height=30,
                relief="sunken", borderwidth=3, 
                text="📷 upload image/video or camera input 📷",
                font=("Arial", 11), fg="gray",
                justify=tk.CENTER)
panel.grid(row=2, column=0, pady=20, padx=20, sticky="nsew")

# Add result label
app.result_label = tk.Label(app, text="🎯 Select an image or start camera to begin detection", 
                           font=("Arial", 12), bg="#bad5ff", fg="darkgreen",
                           wraplength=600, justify=tk.CENTER)
app.result_label.grid(row=3, column=0, pady=10, sticky="ew")

# Main buttons frame
button_frame = tk.Frame(app, bg="#bad5ff")
button_frame.grid(row=4, column=0, pady=20, sticky="ew")

# Center the buttons
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(4, weight=1)

# Image selection button
btn = tk.Button(button_frame, text="Select Image", command=detect_image, 
               font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", padx=20, pady=12)
btn.grid(row=0, column=1, padx=10)

# Camera button
camera_btn = tk.Button(button_frame, text="Start Camera", command=start_camera, 
                      font=("Arial", 12, "bold"), bg="#2196F3", fg="white", padx=20, pady=12)
camera_btn.grid(row=0, column=2, padx=10)

# Video upload button
video_btn = tk.Button(button_frame, text="Upload Video", command=upload_video, 
                     font=("Arial", 12, "bold"), bg="#FF9800", fg="white", padx=20, pady=12)
video_btn.grid(row=0, column=3, padx=10)

# Logging buttons frame
log_frame = tk.Frame(app, bg="#bad5ff")
log_frame.grid(row=5, column=0, pady=10, sticky="ew")

# Center the log buttons
log_frame.grid_columnconfigure(0, weight=1)
log_frame.grid_columnconfigure(3, weight=1)

# View logs button
view_logs_btn = tk.Button(log_frame, text="📄 View Logs", command=view_logs, 
                         font=("Arial", 11, "bold"), bg="#9C27B0", fg="white", padx=15, pady=8)
view_logs_btn.grid(row=0, column=1, padx=10)

# Show stats button
stats_btn = tk.Button(log_frame, text="📊 Statistics", command=show_detection_stats, 
                     font=("Arial", 11, "bold"), bg="#607D8B", fg="white", padx=15, pady=8)
stats_btn.grid(row=0, column=2, padx=10)

# Add progress bar
progress_bar = ttk.Progressbar(app, mode='indeterminate')

# Initialize language
set_language("English")

app.mainloop()