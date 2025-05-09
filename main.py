# =============================================  
# Hand Gesture Air Canvas with OCR & Image Recognition  
# Author: Ananthakrishnan(@Ananthucyher)  
# Date:24-04-2025  
# ============================================= 
import cv2
import numpy as np
import mediapipe as mp
import pytesseract
from tkinter import Tk, Button, Label, Frame
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
from collections import deque
import threading
import time
import pyttsx3
import os
from difflib import get_close_matches
import requests
import json
import tensorflow as tf
from PIL import ImageOps

# Initialize Mediapipe Hand Tracking with minimal settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# Camera Configuration
camera_width, camera_height = 1280, 720
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
cap.set(cv2.CAP_PROP_FPS, 30)

# Canvas Initialization
canvas = np.ones((camera_height - 20, camera_width - 20, 3), dtype="uint8") * 255
drawing_color = (0, 0, 0)  # Black drawing color
brush_thickness = 12

# Smoothing and Tracking Variables
coords_buffer = deque(maxlen=5)
prev_x, prev_y = None, None
current_cursor_pos = (-1, -1)

# For undo functionality
stroke_history = []
current_stroke = []

# Processing lock
processing_lock = threading.Lock()

class AutocorrectSystem:
    def __init__(self):
        # Enhanced word list with common short words
        self.word_list = {
            'hey', 'hello', 'hi', 'the', 'and', 'for', 'are', 'but', 'not', 
            'you', 'all', 'any', 'can', 'yes', 'no', 'ok', 'hey', 'how', 'what',
            'when', 'where', 'why', 'who', 'this', 'that', 'these', 'those'
        }
        
        # Try to load additional words
        try:
            with open('english_words.json', 'r') as f:
                additional_words = json.load(f)
                self.word_list.update(additional_words.keys())
        except:
            try:
                response = requests.get('https://raw.githubusercontent.com/dwyl/english-words/master/words_dictionary.json')
                additional_words = json.loads(response.text)
                self.word_list.update(additional_words.keys())
                with open('english_words.json', 'w') as f:
                    json.dump(additional_words, f)
            except:
                pass

    def correct_word(self, word):
        """Enhanced word correction with special handling for short words"""
        # Skip if empty or single character that's not a letter
        if len(word) <= 1:
            return word if word.isalpha() else word
            
        # First check if word is in our dictionary (case insensitive)
        lower_word = word.lower()
        if lower_word in self.word_list:
            return word
            
        # Special handling for 2-3 letter words that might be misrecognized
        if len(word) <= 3:
            # Try common substitutions for OCR errors
            common_ocr_errors = {
                're': ['he', 'we', 'me', 'be'],
                'te': ['he', 'me', 'we'],
                'ar': ['are'],
                'wh': ['who', 'why', 'what', 'when', 'where'],
                'th': ['the', 'this', 'that'],
                'he': ['hey', 'the', 'she', 'we'],
                'we': ['he', 'me', 'be']
            }
            
            if lower_word in common_ocr_errors:
                return common_ocr_errors[lower_word][0]  # Return most likely correction
            
            # Check if it's a prefix of common words
            for dict_word in self.word_list:
                if len(dict_word) >= 3 and dict_word.startswith(lower_word):
                    return dict_word
                    
        # Find closest matches with higher similarity threshold
        matches = get_close_matches(lower_word, self.word_list, n=1, cutoff=0.75)
        if matches:
            return matches[0]
            
        return word

    def correct_text(self, text):
        """Correct entire text with enhanced logic"""
        if not text.strip():
            return text
            
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Preserve original if it's a single character that's not a letter
            if len(word) == 1 and not word.isalpha():
                corrected_words.append(word)
                continue
                
            corrected = self.correct_word(word)
            
            # Capitalize if original was capitalized (for proper nouns)
            if word[0].isupper():
                corrected = corrected.capitalize()
                
            corrected_words.append(corrected)
            
        return ' '.join(corrected_words)

class ImageRecognizer:
    def __init__(self, model_path=r'C:\Users\ASUS\Downloads\quickdraw_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.categories = ['apple', 'banana', 'car', 'house', 'tree']  # Must match your training
        
    def preprocess_drawing(self, canvas_image):
        """Convert canvas drawing to model input format"""
        # Convert to grayscale and invert (QuickDraw uses white on black)
        pil_img = Image.fromarray(canvas_image).convert('L')
        pil_img = ImageOps.invert(pil_img)
        
        # Resize to 28x28 and convert to numpy array
        pil_img = pil_img.resize((28, 28))
        img_array = np.array(pil_img).astype('float32') / 255.0
        
        # Add batch and channel dimensions
        return np.expand_dims(img_array, axis=(0, -1))
    
    def recognize(self, canvas_image):
        """Recognize the drawn image and return top prediction"""
        processed = self.preprocess_drawing(canvas_image)
        predictions = self.model.predict(processed, verbose=0)
        top_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][top_idx])
        return self.categories[top_idx], confidence

class SimplifiedAirCanvas:
    def __init__(self, root):
        self.root = root
        self.root.title("Simplified Air Canvas with OCR")
        self.root.attributes('-fullscreen', True)
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Flag for grid display
        self.show_grid = False
        self.grid_size = 30
        
        # Main Frame Setup
        self.main_frame = Frame(root)
        self.main_frame.pack(fill="both", expand=True)
        
        # Canvas Frame (Whiteboard)
        self.canvas_frame = Frame(self.main_frame, width=camera_width - 20, height=camera_height - 20)
        self.canvas_frame.pack(side="left", fill="both", expand=True)
        
        # Canvas Display Label
        self.canvas_label = Label(self.canvas_frame)
        self.canvas_label.pack(fill="both", expand=True)
        
        # Side Panel Frame
        self.side_panel = Frame(self.main_frame, width=200)
        self.side_panel.pack(side="right", padx=10)
        
        # Recognized Text Display
        self.text_label = Label(self.side_panel, text="Recognized Text:", font=('Arial', 14, 'bold'))
        self.text_label.pack(pady=(10, 5))
        
        self.recognized_text = Label(self.side_panel,
                                    text="Draw text and click Recognize",
                                    font=('Arial', 12),
                                    wraplength=180,
                                    justify='center',
                                    height=10)
        self.recognized_text.pack(pady=(0, 20))
        
        # Control Buttons
        button_frame = Frame(self.side_panel)
        button_frame.pack(fill='x', pady=10)
        
        self.clear_button = Button(button_frame, text="Clear", command=self.clear_canvas,
                                 bg='#2ecc71', fg='white', font=('Arial', 12))
        self.clear_button.pack(side='left', expand=True, padx=2)
        
        self.undo_button = Button(button_frame, text="Undo", command=self.undo_last_stroke,
                                bg='#f39c12', fg='white', font=('Arial', 12))
        self.undo_button.pack(side='left', expand=True, padx=2)
        
        # Second row buttons
        button_frame2 = Frame(self.side_panel)
        button_frame2.pack(fill='x', pady=5)
        
        self.recognize_button = Button(button_frame2, text="Recognize", command=self.recognize_text,
                                     bg='#3498db', fg='white', font=('Arial', 12))
        self.recognize_button.pack(side='left', expand=True, padx=2)
        
        self.speak_button = Button(button_frame2, text="Speak", command=self.speak_text,
                                 bg='#9b59b6', fg='white', font=('Arial', 12))
        self.speak_button.pack(side='left', expand=True, padx=2)
        
        self.grid_button = Button(self.side_panel, text="Toggle Grid", command=self.toggle_grid,
                                bg='#3498db', fg='white', font=('Arial', 12))
        self.grid_button.pack(fill='x', pady=5)
        
        self.exit_button = Button(self.side_panel, text="Exit", command=self.exit_app,
                                bg='#e74c3c', fg='white', font=('Arial', 12))
        self.exit_button.pack(fill='x', pady=5)
        
        # Small camera preview
        self.camera_label = Label(self.main_frame)
        self.camera_label.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
        
        # Start application threads
        self.running = True
        self.latest_frame = None
        
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.ui_thread = threading.Thread(target=self.update_ui, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()
        self.ui_thread.start()
        
        # Initialize image recognizer
        self.image_recognizer = ImageRecognizer()
    
        # Add image recognition button
        self.recognize_img_button = Button(
            self.side_panel, 
            text="Recognize Image", 
            command=self.recognize_image,
            bg='#3498db',
            fg='white',
            font=('Arial', 12)
        )
        self.recognize_img_button.pack(fill='x', pady=5)

    def recognize_image(self):
        """Recognize the drawn image using the image recognizer"""
        global canvas
        
        try:
            self.recognized_text.config(text="Recognizing...", fg="blue")
            self.root.update()  # Force UI update
            
            with processing_lock:
                # Get current canvas (convert to RGB first)
                img_array = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            
            # Perform recognition
            label, confidence = self.image_recognizer.recognize(img_array)
            
            if confidence > 0.7:  # Confidence threshold
                result = f"Recognized: {label}"
                self.recognized_text.config(text=result, fg="green")
                  # Optional: speak the result
            else:
                self.recognized_text.config(
                    text=f"Best guess: {label} ()", 
                    fg="orange"
                )
                
        except Exception as e:
            self.recognized_text.config(text=f"Error: {str(e)}", fg="red")

    def toggle_grid(self):
        """Toggle grid overlay on canvas"""
        self.show_grid = not self.show_grid
    
    def draw_grid(self, img):
        """Draw grid lines on canvas"""
        if not self.show_grid:
            return img
            
        grid_img = img.copy()
        h, w = grid_img.shape[:2]
        
        # Draw horizontal lines
        for y in range(0, h, self.grid_size):
            cv2.line(grid_img, (0, y), (w, y), (200, 200, 200), 1)
            
        # Draw vertical lines
        for x in range(0, w, self.grid_size):
            cv2.line(grid_img, (x, 0), (x, h), (200, 200, 200), 1)
            
        return grid_img
    
    def draw_cursor(self, img, x, y):
        """Draw cursor at position"""
        if x == -1 or y == -1:
            return img
            
        cursor_radius = 20
        cursor_thickness = 3
        
        # Draw cursor background
        cv2.circle(img, (x, y), cursor_radius, (50, 50, 50), -1)
        
        # Draw white outline
        cv2.circle(img, (x, y), cursor_radius, (255, 255, 255), cursor_thickness)
        
        # Draw cross lines
        cv2.line(img, (x - cursor_radius, y), (x + cursor_radius, y), (255, 255, 255), 1)
        cv2.line(img, (x, y - cursor_radius), (x, y + cursor_radius), (255, 255, 255), 1)
        
        return img
    
    def undo_last_stroke(self):
        """Undo the last stroke"""
        global canvas, stroke_history
        
        with processing_lock:
            if stroke_history:
                canvas = stroke_history.pop().copy()
                current_stroke.clear()
    
    def clear_canvas(self):
        """Clear the canvas"""
        global canvas, prev_x, prev_y
        
        with processing_lock:
            stroke_history.append(canvas.copy())
            canvas = np.ones((camera_height - 20, camera_width - 20, 3), dtype="uint8") * 255
            prev_x, prev_y = None, None
            current_stroke.clear()
    
    def speak_text(self):
        """Speak the recognized text"""
        text = self.recognized_text.cget("text")
        
        if text in ["Draw text and click Recognize", "No text recognized", "Processing..."]:
            return
            
        try:
            self.engine.stop()
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Text-to-speech error: {e}")
    
    def optimize_image_for_ocr(self, img):
        """Optimize image for OCR recognition"""
        # Convert to PIL for enhanced processing
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2.5)
        
        # Convert to grayscale
        pil_img = pil_img.convert('L')
        
        # Apply threshold to make binary image
        pil_img = pil_img.point(lambda x: 0 if x < 170 else 255, '1')
        
        # Resize for better OCR (2x)
        width, height = pil_img.size
        pil_img = pil_img.resize((width*2, height*2), Image.LANCZOS)
        
        # Sharpen
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        
        # Convert back to OpenCV format
        processed = np.array(pil_img)
        
        # Invert for better OCR (black text on white background)
        if processed.dtype == np.bool_:
            processed = processed.astype(np.uint8) * 255
        else:
            processed = 255 - processed
            
        # Save debug image
        cv2.imwrite("optimized.png", processed)
        
        return processed
    
    def recognize_text(self):
        """OCR with autocorrect functionality"""
        global canvas
        
        try:
            self.recognized_text.config(text="Processing...", fg="blue")
            self.root.update()
            
            # Initialize autocorrect system if not exists
            if not hasattr(self, 'autocorrect'):
                self.autocorrect = AutocorrectSystem()
            
            # Get current canvas
            with processing_lock:
                img = cv2.cvtColor(canvas.copy(), cv2.COLOR_RGB2BGR)
            
            # Optimize image for OCR
            processed = self.optimize_image_for_ocr(img)
            
            # Try multiple Tesseract configurations and pick best result
            psm_modes = [6, 7, 8, 3]  # Different page segmentation modes
            results = []
            
            for psm in psm_modes:
                config = f'--oem 3 --psm {psm} -l eng --dpi 300'
                text = pytesseract.image_to_string(processed, config=config)
                
                # Get confidence data
                data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = sum(confidences)/len(confidences) if confidences else 0
                
                text = text.strip()
                if text:
                    results.append((text, avg_confidence))
            
            # Sort by confidence and pick best result
            if results:
                results.sort(key=lambda x: x[1], reverse=True)
                best_text, confidence = results[0]
                text_to_speak = best_text  # Default to recognized text
                
                # Apply autocorrect based on confidence
                if confidence < 85:  # Only correct if confidence isn't very high
                    corrected_text = self.autocorrect.correct_text(best_text)
                    
                    # Show both original and corrected text if they differ
                    if corrected_text != best_text:
                        display_text = f"Recognized: {best_text}\nCorrected: {corrected_text}"
                        self.recognized_text.config(
                            text=display_text,
                            fg="green" if confidence > 70 else "orange"
                        )
                        text_to_speak = corrected_text
                    else:
                        self.recognized_text.config(
                            text=best_text,
                            fg="black"
                        )
                else:
                    self.recognized_text.config(
                        text=best_text,
                        fg="black"
                    )
                
                # Speak the final text (either corrected or original)
                self.engine.say(text_to_speak)
                self.engine.runAndWait()
            else:
                self.recognized_text.config(text="No text recognized", fg="red")
                
        except Exception as e:
            self.recognized_text.config(text=f"Error: {str(e)}", fg="red")

    
    def process_frame(self, frame):
        """Process video frame and detect hand landmarks"""
        global prev_x, prev_y, canvas, current_stroke
        
        if frame is None:
            return frame, -1, -1, frame
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process for hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        is_drawing = False
        canvas_x, canvas_y = -1, -1
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip position
                index_tip = hand_landmarks.landmark[8]
                
                # Convert to pixel coordinates
                x = int(index_tip.x * camera_width)
                y = int(index_tip.y * camera_height)
                
                # Smooth coordinates
                coords_buffer.append((x, y))
                smoothed_x = int(np.mean([p[0] for p in coords_buffer]))
                smoothed_y = int(np.mean([p[1] for p in coords_buffer]))
                
                # Map to canvas coordinates
                canvas_x = int(smoothed_x * (camera_width - 20) / camera_width)
                canvas_y = int(smoothed_y * (camera_height - 20) / camera_height)
                
                # Check if finger is in drawing position (tip is above knuckle)
                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                    is_drawing = True
                    with processing_lock:
                        if prev_x is not None and prev_y is not None:
                            # Save first point of stroke for undo
                            if not current_stroke:
                                stroke_history.append(canvas.copy())
                                
                            # Draw line on canvas
                            cv2.line(canvas, (prev_x, prev_y), (canvas_x, canvas_y), drawing_color, brush_thickness)
                            
                            # Store line segment
                            current_stroke.append(((prev_x, prev_y), (canvas_x, canvas_y)))
                else:
                    # End current stroke
                    current_stroke.clear()
                    
                # Update previous positions
                prev_x, prev_y = canvas_x, canvas_y
        
        # Draw cursor on canvas copy for display
        canvas_display = canvas.copy()
        canvas_display = self.draw_cursor(canvas_display, canvas_x, canvas_y)
        
        # Draw grid if enabled
        canvas_display = self.draw_grid(canvas_display)
        
        return frame, canvas_x, canvas_y, canvas_display
    
    def capture_frames(self):
        """Thread to capture camera frames"""
        global cap
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.latest_frame = frame.copy()
            
            time.sleep(1/60)  # Limit frame rate
    
    def process_frames(self):
        """Thread to process captured frames"""
        global current_cursor_pos
        
        while self.running:
            if self.latest_frame is not None:
                frame, x, y, canvas_display = self.process_frame(self.latest_frame.copy())
                
                if x != -1 and y != -1:
                    current_cursor_pos = (x, y)
            
            time.sleep(1/30)  # Process at 30fps
    
    def update_ui(self):
        """Thread to update UI elements"""
        while self.running:
            try:
                # Get latest canvas with cursor
                with processing_lock:
                    canvas_copy = canvas.copy()
                
                # Draw cursor if position is valid
                if current_cursor_pos[0] != -1:
                    canvas_copy = self.draw_cursor(canvas_copy, 
                                                 current_cursor_pos[0], 
                                                 current_cursor_pos[1])
                
                # Apply grid if enabled
                canvas_copy = self.draw_grid(canvas_copy)
                
                # Convert to PIL format for Tkinter
                canvas_rgb = cv2.cvtColor(canvas_copy, cv2.COLOR_BGR2RGB)
                canvas_pil = Image.fromarray(canvas_rgb)
                canvas_tk = ImageTk.PhotoImage(image=canvas_pil)
                
                # Update canvas display
                self.canvas_label.config(image=canvas_tk)
                self.canvas_label.image = canvas_tk
                
                # Update camera preview
                if self.latest_frame is not None:
                    # Create small preview
                    thumb_size = (160, 120)
                    thumb = cv2.resize(self.latest_frame, thumb_size)
                    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                    thumb_pil = Image.fromarray(thumb_rgb)
                    thumb_tk = ImageTk.PhotoImage(image=thumb_pil)
                    
                    # Update preview
                    self.camera_label.config(image=thumb_tk)
                    self.camera_label.image = thumb_tk
                
                time.sleep(1/30)  # Update at 30fps
                
            except Exception as e:
                print(f"UI update error: {e}")
                time.sleep(0.1)
    
    def exit_app(self):
        """Exit application and clean up resources"""
        self.running = False
        time.sleep(0.5)  # Allow threads to finish
        
        # Release camera
        if cap is not None:
            cap.release()
        
        # Save canvas
        cv2.imwrite("last_drawing.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        self.root.destroy()
        os._exit(0)  # Force exit all threads

def main():
    root = Tk()
    app = SimplifiedAirCanvas(root)
    root.mainloop()

if __name__ == "__main__":
    main()
