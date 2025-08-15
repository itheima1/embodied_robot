# -*- coding: utf-8 -*-
# Required libraries: opencv-python, scikit-learn, numpy, Pillow
# Required files: haarcascade_frontalface_default.xml, a Chinese font file (.ttf/.otf)

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageDraw, ImageFont
import time
import os

# --- Configuration ---
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml' # Haar Cascade file path
# --- 重要: 请将 'simhei.ttf' 替换为你实际使用的字体文件名 (确保它在脚本同目录下) ---
# --- 例如 'msyh.ttf' (微软雅黑), 'wqy-zenhei.ttc' (文泉驿正黑) 等 ---
CHINESE_FONT_PATH = 'simhei.ttf'
RESIZE_DIM = (48, 48) # Resize face images to this dimension
MIN_SAMPLES_PER_CLASS = 10 # Minimum samples required per expression to train
KNN_NEIGHBORS = 3 # Number of neighbors for KNN classifier

# --- Font Sizes ---
FONT_SIZE_INFO = 18      # Font size for status info (top-left)
FONT_SIZE_LABEL = 24     # Font size for expression label near face
FONT_SIZE_MSG = 20       # Font size for bottom messages
FONT_SIZE_GUIDE = 16     # Font size for initial guide text

# --- Initialization ---
# Check if required files exist
if not os.path.exists(FACE_CASCADE_PATH):
    print(f"错误：找不到人脸检测模型文件 '{FACE_CASCADE_PATH}'。请下载并放到脚本同目录下。")
    exit()
if not os.path.exists(CHINESE_FONT_PATH):
    print(f"错误：找不到字体文件 '{CHINESE_FONT_PATH}'。请将字体文件放到脚本同目录下，或修改 CHINESE_FONT_PATH 变量。")
    # Attempt to load a very basic system font as fallback (might not support Chinese well)
    # This part is system dependent and might fail
    try:
        # Common locations (adjust if needed for your OS)
        if os.name == 'nt': # Windows
             common_fonts = ['msyh.ttc', 'simhei.ttf', 'simsun.ttc']
             font_dir = 'C:/Windows/Fonts'
        elif os.name == 'posix': # Linux/macOS
            common_fonts = ['wqy-zenhei.ttc', 'PingFang.ttc', 'STHeiti Light.ttc']
            font_dir = '/usr/share/fonts/truetype/wqy' # Example for WenQuanYi on Linux
            if not os.path.exists(font_dir): font_dir = '/System/Library/Fonts' # macOS
        else:
            common_fonts = []
            font_dir = ''

        found_fallback = False
        for font_name in common_fonts:
            fallback_path = os.path.join(font_dir, font_name)
            if os.path.exists(fallback_path):
                print(f"警告：将尝试使用系统字体 '{fallback_path}' 作为备选。")
                CHINESE_FONT_PATH = fallback_path
                found_fallback = True
                break
        if not found_fallback:
           print("警告：也未能找到备选系统字体。中文可能无法正确显示。")
           # We can proceed but Chinese text will likely fail to render.
           # Pillow might default to a basic font if the path is invalid later.

    except Exception as e:
        print(f"尝试加载系统字体时出错: {e}")
        print("警告：无法加载备选字体。中文可能无法正确显示。")
        # Let it continue, Pillow might handle the invalid path later

# Load face detector (our "existing model")
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    print(f"错误：无法加载人脸检测模型文件 '{FACE_CASCADE_PATH}' (即使文件存在)。请检查文件是否损坏或 OpenCV 安装是否完整。")
    exit()

# Initialize camera
cap = cv2.VideoCapture(1) # 0 usually represents the default camera
if not cap.isOpened():
    print("错误：无法打开摄像头。请检查摄像头是否连接并被系统识别。")
    exit()

# Store training data and labels
X_train = [] # Stores processed face image data (features)
y_train = [] # Stores corresponding labels ('happy', 'not_happy')

# Initialize classifier (the model we will "fine-tune"/train)
knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
model_trained = False
last_message_time = 0
message_display_duration = 3 # Seconds to display temporary messages
current_message = ""
show_guide = True # Flag to show initial instructions

# --- Helper Functions ---

def preprocess_face(face_img):
    """Preprocesses the detected face image: grayscale, resize, flatten"""
    try:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        flattened_face = resized_face.flatten()
        return flattened_face
    except cv2.error as e:
        print(f"预处理人脸时出错: {e}. 可能人脸区域太小或无效。")
        return None # Return None if preprocessing fails

def set_message(msg, duration=message_display_duration):
    """Sets a message to be displayed on the screen"""
    global current_message, last_message_time, message_display_duration
    current_message = msg
    last_message_time = time.time()
    message_display_duration = duration

def cv2_putText_chinese(img, text, position, font_path, font_size, color_bgr):
    """
    Draws Chinese text on an OpenCV image (using Pillow).
    :param img: OpenCV image (NumPy array, BGR format)
    :param text: Text to draw (can include Chinese)
    :param position: Top-left corner coordinate (x, y)
    :param font_path: Path to the ttf/otf font file
    :param font_size: Font size
    :param color_bgr: Text color (B, G, R) tuple, e.g., (255, 0, 0) for blue
    :return: OpenCV image with text drawn (NumPy array, BGR format)
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback if font loading fails even after initial checks/warnings
        print(f"运行时错误：无法加载字体 '{font_path}'。将尝试用 OpenCV 绘制英文替代。")
        cv2.putText(img, "Font Error", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        return img
    except Exception as e:
        print(f"加载字体时发生未知错误: {e}")
        cv2.putText(img, "Font Error", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        return img

    # Convert OpenCV image (BGR) to Pillow image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Pillow uses RGB color
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) # BGR to RGB

    # Draw text
    try:
        draw.text(position, text, font=font, fill=color_rgb)
    except Exception as e:
         print(f"Pillow 绘制文本时出错: {e}")
         # Draw an error indicator if text drawing fails
         draw.text(position, "Draw Error", font=ImageFont.load_default(), fill=(255,0,0))


    # Convert Pillow image (RGB) back to OpenCV image (BGR)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

# --- Main Loop ---
print("\n--- AI 表情识别小课堂 ---")
print("正在启动摄像头...")

mode = "等待操作" # Current mode

while True:
    # Read camera frame
    ret, frame = cap.read()
    if not ret:
        print("错误：无法从摄像头读取画面，退出。")
        set_message("无法读取摄像头画面!", 10)
        time.sleep(2) # Give time for user to see message if window is still up
        break

    # Flip the frame horizontally for a more intuitive mirror effect
    frame = cv2.flip(frame, 1)

    # Create a copy for drawing without affecting detection
    display_frame = frame.copy()

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # Adjust minSize and maxSize based on expected face size in the view
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60), # Don't detect faces smaller than 60x60 pixels
        maxSize=(300, 300) # Don't detect faces larger than 300x300 pixels
    )

    # Get pressed key
    key = cv2.waitKey(1) & 0xFF

    # If any key is pressed, hide the initial guide
    if key != 255: # 255 means no key pressed
        show_guide = False

    action_this_frame = False # Flag if data collection or training happened

    # Process each detected face
    # If multiple faces detected, we'll usually process the first one for collection/prediction
    # or base the mode update on whether *any* face was detected.
    face_detected_this_frame = len(faces) > 0

    if face_detected_this_frame:
        # Sort faces by size (area) in descending order, process the largest first
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (x, y, w, h) = faces[0] # Get coordinates of the largest face

        # Extract face Region of Interest (ROI)
        face_roi = frame[y:y+h, x:x+w]

        # --- Data Collection Mode ---
        if key == ord('h'): # Collect 'happy' sample
            if face_roi.size > 0:
                processed_face = preprocess_face(face_roi)
                if processed_face is not None:
                    X_train.append(processed_face)
                    y_train.append('happy')
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3) # Thick Green box
                    set_message(f"收集到 [开心] 样本! (共 {y_train.count('happy')} 个)")
                    action_this_frame = True
                    mode = "收集中..."
            else:
                set_message("脸部区域太小或无效，无法收集")

        elif key == ord('s'): # Collect 'not_happy' sample
             if face_roi.size > 0:
                processed_face = preprocess_face(face_roi)
                if processed_face is not None:
                    X_train.append(processed_face)
                    y_train.append('not_happy')
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 3) # Thick Red box
                    set_message(f"收集到 [不开心] 样本! (共 {y_train.count('not_happy')} 个)")
                    action_this_frame = True
                    mode = "收集中..."
             else:
                 set_message("脸部区域太小或无效，无法收集")

        # --- Prediction Mode (after model is trained) ---
        elif model_trained:
            if face_roi.size > 0:
                processed_face = preprocess_face(face_roi)
                if processed_face is not None:
                    # KNN expects a 2D array as input
                    prediction = knn.predict([processed_face])[0]
                    label = "开心 :)" if prediction == 'happy' else "不开心 :("
                    color = (0, 255, 0) if prediction == 'happy' else (0, 0, 255)

                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    # Use the Chinese drawing function for the label
                    display_frame = cv2_putText_chinese(display_frame, label, (x, y - FONT_SIZE_LABEL - 5), CHINESE_FONT_PATH, FONT_SIZE_LABEL, color)
                    if not action_this_frame: mode = "识别中..."
                else:
                     # Draw a gray box if preprocessing failed but face was detected
                     cv2.rectangle(display_frame, (x, y), (x+w, y+h), (128, 128, 128), 1)
                     if not action_this_frame: mode = "脸部处理错误"
            else:
                 # Draw a gray box if ROI is invalid
                 cv2.rectangle(display_frame, (x, y), (x+w, y+h), (128, 128, 128), 1)
                 if not action_this_frame: mode = "脸部区域错误"


        # --- Untrained or no action: Just draw a box ---
        else:
             cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 150, 0), 2) # Blue box indicates detected face
             if not action_this_frame: mode = "请按 H/S 收集数据, 或按 T 训练"


    # --- Training Mode ---
    if key == ord('t'):
        action_this_frame = True # 't' is an action
        happy_count = y_train.count('happy')
        not_happy_count = y_train.count('not_happy')
        if happy_count >= MIN_SAMPLES_PER_CLASS and not_happy_count >= MIN_SAMPLES_PER_CLASS:
            print("\n开始训练模型...")
            set_message("正在训练模型，请稍候...", 5)
            # Convert lists to NumPy arrays for scikit-learn
            X_train_np = np.array(X_train)
            y_train_np = np.array(y_train)
            # Train the KNN model (this is our "fine-tuning"!)
            try:
                 knn.fit(X_train_np, y_train_np)
                 model_trained = True
                 set_message(f"模型训练完成! (开心:{happy_count}, 不开心:{not_happy_count})", 5)
                 print("模型训练完成！现在将尝试识别表情。")
                 mode = "识别中..."
            except Exception as e:
                 print(f"训练时出错: {e}")
                 set_message(f"训练失败: {e}", 5)
                 mode = "训练失败"

        else:
            set_message(f"数据不足! 需开心/不开心各 {MIN_SAMPLES_PER_CLASS} 个。(当前 {happy_count}/{not_happy_count})", 5)
            print(f"数据不足! 当前 开心:{happy_count}, 不开心:{not_happy_count}。需要至少各 {MIN_SAMPLES_PER_CLASS} 个。")
            mode = "数据不足"

    # --- Quit Program ---
    if key == ord('q'):
        print("收到退出指令。")
        break

    # Update mode based on state if no specific action was taken this frame
    if not action_this_frame:
        if not face_detected_this_frame:
            mode = "未检测到人脸"
        elif not model_trained:
             mode = "请按 H/S 收集数据, 或按 T 训练"
        else:
             # If face detected and model trained, but no prediction happened (e.g., error)
             if mode not in ["识别中...", "脸部处理错误", "脸部区域错误"]:
                 mode = "识别中..." # Default back to recognizing if possible

    # --- Display Status Information on the frame ---
    y_offset = 30 # Starting Y position for text
    line_height = FONT_SIZE_INFO + 8 # Space between lines

    # Display collected sample counts
    info_color = (0, 255, 0) # Green
    display_frame = cv2_putText_chinese(display_frame, f"开心样本: {y_train.count('happy')}", (10, y_offset), CHINESE_FONT_PATH, FONT_SIZE_INFO, info_color)
    y_offset += line_height
    info_color = (0, 0, 255) # Red
    display_frame = cv2_putText_chinese(display_frame, f"不开心样本: {y_train.count('not_happy')}", (10, y_offset), CHINESE_FONT_PATH, FONT_SIZE_INFO, info_color)
    y_offset += line_height

    # Display current mode
    info_color = (255, 255, 255) # White
    display_frame = cv2_putText_chinese(display_frame, f"当前模式: {mode}", (10, y_offset), CHINESE_FONT_PATH, FONT_SIZE_INFO, info_color)
    y_offset += line_height

    # Display minimum samples needed
    if not model_trained:
        info_color = (0, 255, 255) # Yellow
        display_frame = cv2_putText_chinese(display_frame, f"(每种需 {MIN_SAMPLES_PER_CLASS} 个样本才能训练)", (10, y_offset), CHINESE_FONT_PATH, FONT_SIZE_INFO - 2, info_color)
        y_offset += line_height


    # Display temporary message at the bottom
    if current_message and (time.time() - last_message_time < message_display_duration):
        msg_color = (50, 255, 255) # Bright Yellow
        # Calculate position for bottom center (approximately)
        text_size, _ = cv2.getTextSize(current_message, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE_MSG/25.0, 1) # Estimate size roughly
        # Pillow's text rendering size differs, this is just a rough centering
        # We'll use the dedicated function which handles font specifics
        # Need Pillow's help to get accurate size, or just left-align simply.
        # Let's keep it simple and left-align at the bottom.
        display_frame = cv2_putText_chinese(display_frame, current_message, (10, display_frame.shape[0] - FONT_SIZE_MSG - 10), CHINESE_FONT_PATH, FONT_SIZE_MSG, msg_color)
    elif time.time() - last_message_time >= message_display_duration:
        current_message = "" # Clear expired message

    # Display initial guide text if needed
    if show_guide:
        guide_color = (255, 255, 0) # Cyan
        guide_y = display_frame.shape[0] // 2 - 50 # Center vertically roughly
        guide_line_height = FONT_SIZE_GUIDE + 8
        cv2_putText_chinese(display_frame, "--- 操作指南 ---", (10, guide_y), CHINESE_FONT_PATH, FONT_SIZE_GUIDE+2, guide_color)
        guide_y += guide_line_height + 5
        cv2_putText_chinese(display_frame, "[H] 键: 按住并做 [开心] 表情收集数据", (10, guide_y), CHINESE_FONT_PATH, FONT_SIZE_GUIDE, guide_color)
        guide_y += guide_line_height
        cv2_putText_chinese(display_frame, "[S] 键: 按住并做 [不开心] 表情收集数据", (10, guide_y), CHINESE_FONT_PATH, FONT_SIZE_GUIDE, guide_color)
        guide_y += guide_line_height
        cv2_putText_chinese(display_frame, "[T] 键: 收集足够数据后，按此键训练", (10, guide_y), CHINESE_FONT_PATH, FONT_SIZE_GUIDE, guide_color)
        guide_y += guide_line_height
        cv2_putText_chinese(display_frame, "[Q] 键: 退出程序", (10, guide_y), CHINESE_FONT_PATH, FONT_SIZE_GUIDE, guide_color)
        guide_y += guide_line_height
        cv2_putText_chinese(display_frame, "(按任意键开始)", (10, guide_y), CHINESE_FONT_PATH, FONT_SIZE_GUIDE-2, guide_color)

    # Display the resulting frame
    # The window title might still show garbled characters on some systems if it contains Chinese
    window_title = 'happy or not'
    try:
        # Try encoding title for robustness on different systems
        cv2.imshow(window_title.encode('gbk' if os.name == 'nt' else 'utf-8').decode(errors='ignore'), display_frame)
    except:
        cv2.imshow('AI Emotion Demo (Press Q to quit)', display_frame) # Fallback title


# --- Cleanup ---
print("清理资源...")
cap.release()
cv2.destroyAllWindows()
print("程序已退出。")