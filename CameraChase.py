import cv2
import camera_def as pos
from CameraSearcher import CameraSearcher  # CameraSearcherクラスをインポート

# 画像を指定した幅にリサイズする関数
def scale_to_width (frame, width):
    h, w = frame.shape[:2]
    height = round(h * (width / w))
    dst = cv2.resize(frame, dsize=(width, height))
    return dst

# トラッカー（物体追跡アルゴリズム）を選択する関数（利用可能なもののみ）
def select_tracker():
    print("Which Tracker API do you use?")
    available = []

    # トラッカーの特性説明
    tracker_info = {
        "MIL": "MIL: 比較的安定・高速。照明変化や部分的な遮蔽に強い。",
        "KCF": "KCF: 高速で精度も高いが、急激な動きや大きなスケール変化には弱い。",
        "MedianFlow": "MedianFlow: 追跡失敗時を自動検出。動きが緩やかな対象向き。"
    }

    # 利用可能なトラッカーを動的にリストアップ
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
        print("0: MIL      -", tracker_info["MIL"])
        available.append(("MIL", cv2.legacy.TrackerMIL_create))
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        print("1: KCF      -", tracker_info["KCF"])
        available.append(("KCF", cv2.legacy.TrackerKCF_create))
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMedianFlow_create"):
        print("2: MedianFlow -", tracker_info["MedianFlow"])
        available.append(("MedianFlow", cv2.legacy.TrackerMedianFlow_create))

    if not available:
        raise RuntimeError("利用可能なトラッカーがありません。opencv-contrib-pythonが必要です。")

    choice = input("Please select your tracker number: ")
    try:
        idx = int(choice)
        if 0 <= idx < len(available):
            tracker = available[idx][1]()
            return tracker
        else:
            print("範囲外の番号です。KCFを使用します。")
    except Exception:
        print("無効な入力です。KCFを使用します。")

    # デフォルトはKCF
    for name, creator in available:
        if name == "KCF":
            return creator()
    # それもなければ最初のもの
    return available[0][1]()

# カメラ選択
searcher = CameraSearcher()
camera_port = searcher.select_camera()
if camera_port is None:
    print("カメラが見つかりません。")
    exit()

# トラッカーを選択
tracker = select_tracker()
tracker_name = str(tracker).split()[0][1:]
print(tracker)

# 選択したカメラポートでカメラをオープン
cap = cv2.VideoCapture(camera_port)

import time
time.sleep(1)  # カメラ起動待ち

# 最初のフレームを取得し、ROI（追跡範囲）をユーザーに選択させる
ret, frame = cap.read()
roi = cv2.selectROI(frame, False)
ret = tracker.init(frame, roi)
cv2.destroyWindow('frame')

# メインループ：カメラ画像を取得し、物体追跡を実行
while True:
    ret, frame = cap.read()
    success, roi = tracker.update(frame)
    (x,y,w,h) = tuple(map(int,roi))

    if success:
        # 追跡成功時は矩形を描画
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
    else :
        # 追跡失敗時は警告テキストを表示
        cv2.putText(frame, "Tracking failed!!", (500,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

    # 実行しているトラッカー名を画面に表示
    cv2.putText(frame, tracker_name, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0),3)

    # フレームを指定幅にリサイズして表示
    frame = scale_to_width(frame,pos.FRAME_WIDTH)
    cv2.imshow(tracker_name, frame)

    key = cv2.waitKey( 1 ) & 0xFF
    if key == ord( ' ' ):
        break  # スペースキーで終了
cap.release()
cv2.destroyAllWindows()
