import cv2
import numpy as np

# =========================
# 設定
# =========================
VIDEO_PATH = "../kenkyu-0414/kawae-120/kawae000-120.MOV"

BED_WIDTH = 5.5  # m
BED_DEPTH = 3.0  # m

points = []

# 順序固定（重要）
labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

# =========================
# マウスクリック
# =========================
def click_event(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            print(f"{labels[len(points)]}: ({x}, {y})")
            points.append([x, y])

# =========================
# メイン
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("動画が読み込めません")
    exit()

clone = frame.copy()

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", click_event)

while True:
    temp = clone.copy()

    # =========================
    # 既に選択された点の描画
    # =========================
    for i, p in enumerate(points):
        cv2.circle(temp, tuple(p), 6, (0, 0, 255), -1)
        cv2.putText(temp, labels[i],
                    (p[0] + 10, p[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    # =========================
    # 次にクリックすべき位置のガイド
    # =========================
    if len(points) < 4:
        guide_text = f"Click: {labels[len(points)]}"
    else:
        guide_text = "Press 's' to save, 'r' to reset"

    cv2.putText(temp, guide_text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Calibration", temp)

    key = cv2.waitKey(1)

    # ESC終了
    if key == 27:
        break

    # リセット
    elif key == ord('r'):
        print("Reset points")
        points = []

    # 保存
    elif key == ord('s'):
        if len(points) == 4:
            break
        else:
            print("4点選択してください")

cv2.destroyAllWindows()

# =========================
# ホモグラフィ計算
# =========================
if len(points) != 4:
    print("4点不足")
    exit()

src = np.array(points, dtype=np.float32)

dst = np.array([
    [0, 0],
    [BED_WIDTH, 0],
    [BED_WIDTH, BED_DEPTH],
    [0, BED_DEPTH]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(src, dst)

np.save("H-kawae000-120.npy", H)

print("✅ H-kawae000-120.npy を保存しました")