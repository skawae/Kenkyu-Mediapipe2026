import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# --- 設定 ---
VIDEO_PATH = 'trampoline_side_top.mp4'
OUTPUT_CSV = 'analysis_results.csv'

# MediaPipe初期化（高精度モデルを使用）
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2, 
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_center_hip(landmarks):
    """左右の腰の中点を重心の代理として取得"""
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    return (l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2

def calculate_angle(a, b, c):
    """3点間の角度計算"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# データ保存用リスト
data_log = []

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    frame_count += 1
    
    # 解析実行
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # 1. 座標取得
        hip_x, hip_y = get_center_hip(lm)
        ankle_r = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        knee_r = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        
        # 2. 姿勢指標の計算
        # 重心(Hip)と接地足(Ankle)の水平方向のズレ（移動要因の仮説用）
        posture_offset_x = hip_x - ankle_r[0]
        
        # 膝の角度
        knee_angle = calculate_angle([lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y], 
                                     knee_r, ankle_r)

        # 3. データの蓄積
        data_log.append({
            'frame': frame_count,
            'time': frame_count / fps,
            'hip_x': hip_x, 'hip_y': hip_y,
            'ankle_x': ankle_r[0], 'ankle_y': ankle_r[1],
            'posture_offset_x': posture_offset_x,
            'knee_angle': knee_angle
        })

        # 可視化：重心と足首を結ぶ線を描画
        start_point = (int(hip_x * w), int(hip_y * h))
        end_point = (int(ankle_r[0] * w), int(ankle_r[1] * h))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(image, f"Offset: {posture_offset_x:.3f}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Traveling Analysis', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# CSV保存（後の相関分析用）
df = pd.DataFrame(data_log)
df.to_csv(OUTPUT_CSV, index=False)
print(f"解析完了。データは {OUTPUT_CSV} に保存されました。")