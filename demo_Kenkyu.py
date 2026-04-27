import cv2
import mediapipe as mp
import numpy as np

# =========================
# ■ 設定
# =========================
VIDEO_PATH = "../kenkyu-0414/kawae-120/kawae000-120.MOV"
OUTPUT_PATH = "../Kenkyu-result/kawae000-120/demo_Kenkyu-kawae000-120.mp4"
HOMOGRAPHY_PATH = "../Kenkyu-result/kawae000-120/H-kawae000-120.npy"

BED_WIDTH = 5.5   # [m]
BED_DEPTH = 3.0   # [m]

HEIGHT_THRESHOLD = 0.8

# =========================
# ■ 時間ベース設計（研究仕様）
# =========================
CONTACT_TIME = 0.15      # 接地観測時間 [秒]
COOLDOWN_TIME = 0.25     # 再検出防止 [秒]

# =========================
# ■ カメラ歪み補正（例）
# =========================
K = np.array([[2000, 0, 960],
              [0, 2000, 540],
              [0, 0, 1]], dtype=np.float32)

dist = np.array([-0.1, 0.01, 0, 0, 0], dtype=np.float32)

# =========================
# ■ MediaPipe
# =========================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# =========================
# ■ ユーティリティ
# =========================
def midpoint(p1, p2):
    return (p1 + p2) / 2

def apply_homography(pt, H):
    pt_h = np.array([pt[0], pt[1], 1.0])
    dst = H @ pt_h
    dst /= dst[2]
    return dst[:2]

def draw_bed_map(points):
    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (10, 10), (290, 190), (0, 0, 0), 2)

    for i, p in enumerate(points):
        x = int(10 + (p[0] / BED_WIDTH) * 280)
        z = int(10 + (p[1] / BED_DEPTH) * 180)

        cv2.circle(img, (x, z), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i+1),
                    (x+5, z-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)

    return img

# =========================
# ■ メイン処理
# =========================
def run():
    cap = cv2.VideoCapture(VIDEO_PATH)
    H = np.load(HOMOGRAPHY_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # 時間 → フレーム変換
    CONTACT_FRAMES = int(CONTACT_TIME * fps)
    COOLDOWN_FRAMES = int(COOLDOWN_TIME * fps)

    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] CONTACT_FRAMES: {CONTACT_FRAMES}")
    print(f"[INFO] COOLDOWN_FRAMES: {COOLDOWN_FRAMES}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(fps),
        (w, h)
    )

    landing_points = []

    prev_y = None
    prev_v = None

    state = "AIR"
    contact_buffer = []
    contact_timer = 0
    cooldown = 0

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 歪み補正
        frame = cv2.undistort(frame, K, dist)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        landing_flag = 0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            def get_xy(i):
                return np.array([lm[i].x * w, lm[i].y * h])

            try:
                heel_l = get_xy(29)
                heel_r = get_xy(30)
                foot_l = get_xy(31)
                foot_r = get_xy(32)
            except:
                continue

            # 両足の中点
            foot_mid = midpoint(
                midpoint(heel_l, heel_r),
                midpoint(foot_l, foot_r)
            )

            y = foot_mid[1]

            # =========================
            # 接地検出（速度反転）
            # =========================
            if prev_y is not None:
                v = y - prev_y

                if prev_v is not None:
                    if (y > HEIGHT_THRESHOLD * h and
                        prev_v > 0 and v < 0 and
                        cooldown == 0 and
                        state == "AIR"):

                        state = "CONTACT"
                        contact_buffer = []
                        contact_timer = 0
                        cooldown = COOLDOWN_FRAMES

                prev_v = v

            prev_y = y

            if cooldown > 0:
                cooldown -= 1

            # =========================
            # CONTACT状態（最下点抽出）
            # =========================
            if state == "CONTACT":
                contact_buffer.append((foot_mid.copy(), y))
                contact_timer += 1

                if contact_timer >= CONTACT_FRAMES:

                    ys = [p[1] for p in contact_buffer]
                    idx = np.argmax(ys)

                    best_point = contact_buffer[idx][0]
                    world = apply_homography(best_point, H)

                    landing_points.append(world)

                    state = "AIR"
                    landing_flag = 1

            # 描画
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.circle(frame, tuple(foot_mid.astype(int)), 5, (0, 0, 255), -1)

        # =========================
        # ミニマップ
        # =========================
        map_img = draw_bed_map(landing_points)
        mh, mw, _ = map_img.shape
        frame[10:10+mh, w-mw-10:w-10] = map_img

        # 着地フラッシュ
        if landing_flag == 1:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # フレーム表示
        cv2.putText(frame, f"Frame: {frame_id}",
                    (50, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    print("🎯 完了：研究用トランポリン着地点解析システム")


if __name__ == "__main__":
    run()