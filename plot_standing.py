import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# --- 骨格描画用カスタム関数 ---
def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if getattr(start, 'visibility', 1.0) < 0.5 or getattr(end, 'visibility', 1.0) < 0.5:
                continue
            p1 = (int(start.x * w), int(start.y * h))
            p2 = (int(end.x * w), int(end.y * h))
            cv2.line(image, p1, p2, (0, 255, 0), 2)
    for lm_point in landmarks:
        if getattr(lm_point, 'visibility', 1.0) < 0.5:
            continue
        cx, cy = int(lm_point.x * w), int(lm_point.y * h)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)


# --- DLT法関連の関数（変更なし） ---
def compute_dlt_coeffs(img_pts, real_pts):
    A = []
    for (u, v), (X, Y) in zip(img_pts, real_pts):
        A.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y])
        A.append([0, 0, 0, X, Y, 1, -v*X, -v*Y])
    A = np.array(A)
    B = np.array([pt for p in img_pts for pt in p])
    return np.linalg.solve(A, B)

def transform_coords(L, u, v):
    M = np.array([[L[6]*u - L[0], L[7]*u - L[1]], [L[6]*v - L[3], L[7]*v - L[4]]])
    rhs = np.array([L[2] - u, L[5] - v])
    return np.linalg.solve(M, rhs)

# --- 設定 ---
# 順序: 右上, 左上, 左下, 右下
real_points = [(0.5, 0.5), (0.0, 0.5), (0.0, 0.0), (0.5, 0.0)]
# 抽出済みの青枠画像座標（右上, 左上, 左下, 右下）
image_points = [(400, 250), (100, 250), (100, 550), (400, 550)] 

# ベッド（面）の高さの基準を計算（左下と右下のv座標の平均）
bed_v_threshold = (image_points[2][1] + image_points[3][1]) / 2

L_coeffs = compute_dlt_coeffs(image_points, real_points)

# --- MediaPipeのセットアップ (Tasks API) ---
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture('video/kenkyu-0414/kawae-04142054-240.MOV')

was_above_bed = True # 前のフレームでベッドより上にいたか

start_time = time.time()
last_timestamp_ms = 0

# 全フレームのDLT座標を保存するためのリスト
all_real_x = []
all_real_y = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    
    # Tasks API用にRGB変換し、mp.Imageを作成
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # タイムスタンプの計算 (推論用)
    video_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    current_time_ms = video_time_ms if video_time_ms > 0 else int((time.time() - start_time) * 1000)
    
    if current_time_ms <= last_timestamp_ms:
        current_time_ms = last_timestamp_ms + 1
    last_timestamp_ms = current_time_ms
    
    # Tasks API で推論
    results = landmarker.detect_for_video(mp_image, current_time_ms)

    # ベッドの基準線を可視化（青色）
    cv2.line(frame, (0, int(bed_v_threshold)), (w, int(bed_v_threshold)), (255, 0, 0), 2)

    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        lm = results.pose_landmarks[0] # 1人目のランドマーク
        
        # 画面上に骨格を描画
        draw_landmarks(frame, lm)
        
        # 踵(30)とつま先(32)の中間点を計算
        # Tasks APIではインデックス番号で直接アクセスします
        heel = lm[30] # RIGHT_HEEL
        toe = lm[32]  # RIGHT_FOOT_INDEX
        
        u = int(((heel.x + toe.x) / 2) * w)
        v = int(((heel.y + toe.y) / 2) * h)

        # 毎フレーム、足の座標をDLTで2次元実空間の座標に変換
        real_x, real_y = transform_coords(L_coeffs, u, v)

        # グラフ描画用に座標をリストに追加
        all_real_x.append(real_x)
        all_real_y.append(real_y)

        # --- 接地判定ロジック：DLTの実空間（Y座標）で判定 ---
        # 実座標の基準点はベッド面が Y=0.0、上空が Y=0.5 となっているので、 
        # real_y <= 0.0 になればベッドに接触（接地面より下）と判定
        is_below_bed = real_y <= 0.0

        if was_above_bed and is_below_bed:
            print(f"【接地】ベッド接触検知: 2次元DLT座標 X={real_x:.3f}, Y={real_y:.3f}")
            
            # 接地した瞬間に赤い大きめの円を表示
            cv2.circle(frame, (u, v), 15, (0, 0, 255), -1)

        # 状態の更新
        was_above_bed = not is_below_bed

        # デバッグ用：現在の中間点を表示（緑）
        cv2.circle(frame, (u, v), 5, (0, 255, 0), -1)

    cv2.imshow('Trampoline Bed Contact Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

# --- 動画終了後：算出したすべてのDLT座標をプロット ---
if len(all_real_x) > 0:
    print("グラフを生成しています...")
    plt.figure(figsize=(8, 6))
    
    # 軌跡を線と点でプロット
    plt.plot(all_real_x, all_real_y, marker='o', markersize=3, linestyle='-', color='b', alpha=0.7, label='Foot Trajectory')
    
    # ベッドの高さ基準 (Y=0.0) を赤の点線で表示
    plt.axhline(y=0.0, color='r', linestyle='--', label='Bed Surface (Y=0.0)')
    
    plt.title('Foot Trajectory based on 2D DLT')
    plt.xlabel('Real X Coordinate')
    plt.ylabel('Real Y Coordinate')
    plt.legend()
    plt.grid(True)
    # 実空間のスケールを保つため縦横比を揃える
    plt.axis('equal')
    
    plt.show()
else:
    print("有効な足の座標が検出されなかったため、グラフを描画できません。")