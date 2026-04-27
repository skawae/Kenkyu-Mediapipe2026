import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# =========================
# 設定
# =========================
VIDEO_PATH = "../kenkyu-0414/kawae-120/kawae000-120.MOV"
OUTPUT_VIDEO_PATH = "../video/demo_result120.mp4"
OUTPUT_CSV_PATH = "../video/output_final.csv"
HOMOGRAPHY_PATH = "../Kenkyu-result/kawae000-120/H-kawae000-120.npy"

HEIGHT_THRESHOLD = 0.8

# =========================
# 歪み補正（要キャリブレーション値）
# =========================
K = np.array([[2000, 0, 960],
              [0, 2000, 540],
              [0, 0, 1]], dtype=np.float32)

dist = np.array([-0.1, 0.01, 0, 0, 0], dtype=np.float32)

# =========================
# MediaPipe
# =========================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# =========================
# 関数定義
# =========================
def adaptive_savgol(data):
    data = np.array(data)

    # NaN除去
    if np.isnan(data).any():
        data = pd.Series(data).interpolate().fillna(method='bfill').fillna(method='ffill').values

    n = len(data)
    if n < 5:
        return data

    window = min(7, n if n % 2 == 1 else n - 1)
    if window < 3:
        return data

    return savgol_filter(data, window, 2)

def calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-6:
        return 0
    cos_theta = np.dot(ba, bc) / norm
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def midpoint(p1, p2):
    return (p1 + p2) / 2

def apply_homography(pt, H):
    pt_h = np.array([pt[0], pt[1], 1.0])
    dst = H @ pt_h
    dst /= dst[2]
    return dst[:2]

# =========================
# メイン
# =========================
def run():
    cap = cv2.VideoCapture(VIDEO_PATH)
    try:
        H = np.load(HOMOGRAPHY_PATH)
    except FileNotFoundError:
        print(f"⚠️ ホモグラフィ行列 {HOMOGRAPHY_PATH} が見つかりません。デフォルトの単位行列を使用します。")
        H = np.eye(3, dtype=np.float32)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    results = []
    frame_id = 0

    # リアルタイム用の変数
    prev_y = None
    prev_v = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 歪み補正
        frame = cv2.undistort(frame, K, dist)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        realtime_landing_flag = 0
        land_x, land_z = 0.0, 0.0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            def get_xy(i):
                return np.array([lm[i].x * w, lm[i].y * h])

            try:
                hip_l = get_xy(23)
                hip_r = get_xy(24)
                knee_l = get_xy(25)
                knee_r = get_xy(26)
                ankle_l = get_xy(27)
                ankle_r = get_xy(28)
                heel_l = get_xy(29)
                heel_r = get_xy(30)
                foot_l = get_xy(31)
                foot_r = get_xy(32)
                shoulder_l = get_xy(11)
                shoulder_r = get_xy(12)
                
                # 骨格描画
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 中点
                hip_mid = midpoint(hip_l, hip_r)
                shoulder_mid = midpoint(shoulder_l, shoulder_r)
                foot_mid = midpoint(midpoint(heel_l, heel_r), midpoint(foot_l, foot_r))

                # 角度（CSV用）
                knee_l_ang = calc_angle(hip_l, knee_l, ankle_l)
                knee_r_ang = calc_angle(hip_r, knee_r, ankle_r)
                hip_l_ang = calc_angle(shoulder_l, hip_l, knee_l)
                hip_r_ang = calc_angle(shoulder_r, hip_r, knee_r)
                ankle_l_ang = calc_angle(knee_l, ankle_l, foot_l)
                ankle_r_ang = calc_angle(knee_r, ankle_r, foot_r)
                trunk = calc_angle(hip_mid + np.array([0, -1]), hip_mid, shoulder_mid)

                # リアルタイム接地判定と動画描画（demo_plot を踏襲）
                y = foot_mid[1]
                if prev_y is not None:
                    v = y - prev_y
                    if prev_v is not None:
                        if y > HEIGHT_THRESHOLD * h and prev_v > 0 and v < 0:
                            realtime_landing_flag = 1
                    prev_v = v
                prev_y = y

                # リアルタイムでのホモグラフィ（動画表示用）
                world = apply_homography(foot_mid, H)
                land_x, land_z = world

                # 角度・着地点のリアルタイム表示
                cv2.putText(frame, f"Knee(L): {int(knee_l_ang)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.circle(frame, tuple(foot_mid.astype(int)), 8, (0, 0, 255), -1)

                # データリストへ追加
                results.append([
                    frame_id,
                    knee_l_ang, knee_r_ang,
                    hip_l_ang, hip_r_ang,
                    ankle_l_ang, ankle_r_ang,
                    trunk,
                    foot_mid[0], foot_mid[1]
                ])

            except Exception as e:
                # 取得失敗時は記録せずにスキップ
                pass

        # リアルタイムの着地フラッシュ
        if realtime_landing_flag == 1:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.putText(frame, f"Landing: ({land_x:.2f}, {land_z:.2f})", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # フレーム番号
        cv2.putText(frame, f"Frame: {frame_id}", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print("🎥 動画出力完了:", OUTPUT_VIDEO_PATH)

    # =========================
    # DataFrame 作成と CSV 保存
    # =========================
    if len(results) == 0:
        print("⚠️ データが取得できませんでした (CSVは出力されません)")
        return

    df = pd.DataFrame(results, columns=[
        "frame_id",
        "knee_L", "knee_R",
        "hip_L", "hip_R",
        "ankle_L", "ankle_R",
        "trunk",
        "foot_x", "foot_y"
    ])

    # 平滑化（改良版）
    for col in df.columns[1:]:
        df[col] = adaptive_savgol(df[col])

    # オフラインの厳密な接地判定
    landing = [0] * len(df)
    for i in range(1, len(df)-1):
        prev = df["foot_y"].iloc[i-1]
        curr = df["foot_y"].iloc[i]
        next_ = df["foot_y"].iloc[i+1]
        if curr > HEIGHT_THRESHOLD * h and prev < curr and next_ < curr:
            landing[i] = 1
    df["landing_flag"] = landing

    # ホモグラフィ適用
    coords = []
    for i in range(len(df)):
        pt = np.array([df["foot_x"].iloc[i], df["foot_y"].iloc[i]])
        coords.append(apply_homography(pt, H))

    coords = np.array(coords)
    df["land_x"] = coords[:, 0]
    df["land_z"] = coords[:, 1]

    # 保存
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("✅ CSV出力完了（平滑化済み）:", OUTPUT_CSV_PATH)

# =========================
if __name__ == "__main__":
    run()
