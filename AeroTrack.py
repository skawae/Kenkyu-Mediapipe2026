import cv2
import mediapipe as mp
import numpy as np

VIDEO_PATH = "../kenkyu-0414/kawae-120/kawae000-120.MOV"
OUTPUT_PATH = "../Kenkyu-result/kawae000-120/AeroTrack-kawae000-120.mp4"
HOMOGRAPHY_PATH = "../Kenkyu-result/kawae000-120/H-kawae000-120.npy"

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

    # 外枠
    cv2.rectangle(img, (10, 10), (290, 190), (0, 0, 0), 2)

    for i, p in enumerate(points):
        x = int(10 + (p[0] / BED_WIDTH) * 280)
        z = int(10 + (p[1] / BED_DEPTH) * 180)

        # 点
        cv2.circle(img, (x, z), 5, (0, 0, 255), -1)

        # 番号（時系列）
        cv2.putText(img, str(i+1),
                    (x + 5, z - 5),
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

    prev_y = None
    prev_v = None

    # ★ デバウンス用（超重要）
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
        world = None

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

            # 両足中点（重要改善）
            foot_mid = midpoint(
                midpoint(heel_l, heel_r),
                midpoint(foot_l, foot_r)
            )

            y = foot_mid[1]

            if prev_y is not None:
                v = y - prev_y

                if prev_v is not None:
                    if (y > HEIGHT_THRESHOLD * h and
                        prev_v > 0 and v < 0 and
                        landing_cooldown == 0):

                        landing_flag = 1
                        landing_cooldown = 10  # 約0.16秒無効化

                prev_v = v

            prev_y = y

            if landing_cooldown > 0:
                landing_cooldown -= 1

            # ホモグラフィ
            world = apply_homography(foot_mid, H)

            # 接地時のみ記録
            if landing_flag == 1:
                landing_points.append(world)

            # 可視化
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.circle(frame, tuple(foot_mid.astype(int)), 6, (0, 0, 255), -1)

        # ミニマップ
        map_img = draw_bed_map(landing_points)
        mh, mw, _ = map_img.shape
        frame[10:10+mh, w-mw-10:w-10] = map_img

        # 着地フラッシュ
        if landing_flag == 1:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    print("🎯 着地のみプロット版 完成")


if __name__ == "__main__":
    run()