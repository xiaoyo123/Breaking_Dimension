import cv2
import mediapipe as mp
import tkinter as tk
import shutil
import time

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 開啟 webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 設置一個標誌來控制攝影機是否繼續運行
is_paused = False
captured_frame = None  # 儲存拍到的照片
close = False
init = 0
time1 = time.time()
time2 = time.time()
time3 = time.time()
time4 = time.time()
retake = False

def confirm_photo():
    """確認按鈕處理函數"""
    global captured_frame
    if captured_frame is not None:
        cv2.imwrite('input/photo.png', captured_frame)
        cv2.imwrite('examples/body/body.png', captured_frame)
        print("success！")
        close_webcam()  # 關閉 webcam 視窗並釋放攝像頭資源
    close_popup()

def retry_photo():
    """重拍按鈕處理函數"""
    cv2.destroyWindow("Completed")
    global is_paused, retake, time3
    is_paused = False
    close_popup()
    retake = True
    time3 = time.time()
    #shutil.copy("check.jpg","results/cancel.jpg")

def close_popup():
    """關閉彈出窗口"""
    popup.destroy()

def close_webcam():
    """關閉 webcam 視窗並釋放攝像頭"""
    global close
    close = True
    cap.release()  # 釋放攝像頭資源
    cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗

def show_popup():
    """顯示確認與重拍的彈出窗口"""
    global popup
    popup = tk.Tk()
    popup.title("選擇操作")
    label = tk.Label(popup, text="請選擇操作：")
    label.pack(pady=10)

    confirm_button = tk.Button(popup, text="確認", command=confirm_photo)
    confirm_button.pack(side=tk.LEFT, padx=20, pady=20)

    retry_button = tk.Button(popup, text="重拍", command=retry_photo)
    retry_button.pack(side=tk.RIGHT, padx=20, pady=20)

    popup.mainloop()

def testing(captured_frame):
    global is_paused
    cv2.imshow('Completed', captured_frame)
    cv2.waitKey(2000)

    # 暫停攝影，並顯示確認與重拍按鈕
    is_paused = True
    show_popup()

def detect(frame):
    global is_paused, captured_frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # 獲取人體關鍵點
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 檢測 "立正" 動作的條件（條件較寬鬆）
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x
        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x

    ##

        # 放寬條件
        t = 0.3
        if abs(left_shoulder - left_hip) < t and abs(right_shoulder - right_hip) < t:
            if abs(left_wrist - left_hip) < t and abs(right_wrist - right_hip) < t:
                if abs(left_foot - right_foot) < t:

                    # 計算包含人物的矩形框範圍
                    height, width, _ = frame.shape
                    min_x = min(landmark.x for landmark in landmarks) * width
                    max_x = max(landmark.x for landmark in landmarks) * width
                    min_y = min(landmark.y for landmark in landmarks) * height
                    max_y = max(landmark.y for landmark in landmarks) * height
                    
                    # 擴展邊界，確保矩形框不太小
                    padding = 30
                    min_x = max(0, int(min_x - padding))
                    max_x = min(width, int(max_x + padding))
                    min_y = max(0, int(min_y - padding))
                    max_y = min(height, int(max_y + padding))

                    # 截取人物附近的區域
                    cropped_frame = frame[min_y:max_y, min_x:max_x]

                    captured_frame = cropped_frame.copy()

                    # 儲存未繪製骨架的圖像
                    #captured_frame = frame.copy()
                    print("檢測到寬鬆條件下的立正姿勢，等待確認或重拍。")

                    # 顯示拍照完成的視窗
                    cv2.imshow('Completed', captured_frame)
                    cv2.waitKey(2000)

                    # 暫停攝影，並顯示確認與重拍按鈕
                    is_paused = True
                    show_popup()

while True:
    if not is_paused:
        ret, frame = cap.read()

        if not ret:
            break
        
        ### new
        time2 = time.time()
        if init == 3 and time2 - time1 > 5:
            if retake:
                time4 = time.time()
                if time4 - time3 > 3:
                    detect(frame)
            else:
                detect(frame)
        ### 
    
        ### testing
        # if init == 3 and time2 - time1 > 5:
        #     captured_frame = frame.copy()
        #     if retake:
        #         time4 = time.time()
        #         print(time4 - time3)
        #         if time4 - time3 > 3:
        #             testing(captured_frame)
        #     else:
        #         testing(captured_frame)

    # 顯示攝影機畫面
    if not is_paused:
        cv2.imshow('Webcam', frame)
        if init == 2:
            time1 = time.time()
            init += 1
        elif init < 2:
            init += 1

    if close:
        break
    # 按下 'q' 鍵退出攝影
    if cv2.waitKey(1) & 0xFF == ord('q'):
        shutil.copy("check.jpg","results/cancel.jpg")
        break

cap.release()
cv2.destroyAllWindows()
