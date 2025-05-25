#import winsound
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pyautogui as pag
import pyperclip
from CameraSearcher import CameraSearcher  # CameraSearcherクラスをインポート

# カメラ選択
searcher = CameraSearcher()
camera_port = searcher.select_camera()
if camera_port is None:
    print("カメラが見つかりません。")
    exit()

# 選択したカメラでキャプチャ開始
cap = cv2.VideoCapture(camera_port)
cap.set(3,640)
cap.set(4,480)

while True:
    ret,frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    for barcode in decode(frame):
        # QRコードを読み取り、画面にバウンディングボックスと文字列を表示する
        myData = barcode.data.decode('utf-8')
        pts = np.array([barcode.polygon], np.int32)
        cv2.polylines(frame, [pts], True, (255,0,0), 5)
        pts2 = barcode.rect
        cv2.putText(frame, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        
        # クリップボードにQRコードの内容をコピー
        pyperclip.copy(myData)
        # ペーストする
        pag.hotkey('ctrl','v')
        #winsound.Beep(440,1000)  # 必要なら有効化

    cv2.imshow('test', frame)

    # qキーが押されたらループを終了する
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

