import cv2
import mediapipe as mp
import numpy as np
from collections import deque

VIDEO_PATH = "../kenkyu-0414/kawae-120/kawae000-120.MOV"
OUTPUT_PATH = "../Kenkyu-result/kawae000-120/landing_minpoint-kawae000-120.mp4"
HOMOGRAPHY_PATH = "../Kenkyu-result/kawae-120/H-kawae000-120.npy"

BED_WIDTH = 5.5
BED_DEPTH = 3.0
HEIGHT_THRESHOLD = 0.8

# 歪み補正
K = np.array([[2000, 0, 960],
              [0, 2000, 540],
              [0, 0, 1]], dtype=np.float32)
dist = np.array([-0.1, 0.01, 0, 0, 0], dtype=np.float32)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


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


def run():
    cap = cv2.VideoCapture(VIDEO_PATH)
    H = np.load(HOMOGRAPHY_PATH)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    landing_points = []

    # 🔥 バッファ（前後含める）
    buffer = deque(maxlen=15)

    prev_y = None
    prev_v = None

    landing_cooldown = 0

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

            # 両足中点
            foot_mid = midpoint(
                midpoint(heel_l, heel_r),
                midpoint(foot_l, foot_r)
            )

            y = foot_mid[1]

            # バッファ保存
            buffer.append((foot_mid.copy(), y))

            # 接地検出
            if prev_y is not None:
                v = y - prev_y

                if prev_v is not None:
                    if (y > HEIGHT_THRESHOLD * h and
                        prev_v > 0 and v < 0 and
                        landing_cooldown == 0):

                        landing_flag = 1
                        landing_cooldown = 15

                prev_v = v

            prev_y = y

            if landing_cooldown > 0:
                landing_cooldown -= 1

            # 🔥 最下点抽出
            if landing_flag == 1 and len(buffer) > 5:
                ys = [b[1] for b in buffer]
                idx = np.argmax(ys)

                best_point = buffer[idx][0]
                world = apply_homography(best_point, H)

                landing_points.append(world)

            # 描画
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.circle(frame, tuple(foot_mid.astype(int)), 6, (0, 0, 255), -1)

        # ミニマップ
        map_img = draw_bed_map(landing_points)
        mh, mw, _ = map_img.shape
        frame[10:10+mh, w-mw-10:w-10] = map_img

        # フレーム番号
        cv2.putText(frame, f"Frame: {frame_id}",
                    (50, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    print("🎯 最下点＋接地融合版 完成")


if __name__ == "__main__":
    run()