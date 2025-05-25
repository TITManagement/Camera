import cv2
import sys

# バージョンを表示
print("OpenCV version:", cv2.__version__)

# バージョンを数値タプルに変換
version = tuple(map(int, cv2.__version__.split(".")))

# 例：4.7.0以上かどうか判定
if version >= (4, 7, 0):
    print("OpenCV 4.7.0以上です")
else:
    print("")

# OAK-D-Lite用のimport
try:
    import depthai as dai
    OAKD_AVAILABLE = True
except ImportError:
    OAKD_AVAILABLE = False

def get_camera_name(cap):
    # OpenCV 4.7.0以上ならCAP_PROP_DEVICE_DESCRIPTIONを使う
    version = tuple(map(int, cv2.__version__.split(".")))
    if version >= (4, 7, 0) and hasattr(cv2, "CAP_PROP_DEVICE_DESCRIPTION"):
        try:
            desc = cap.get(cv2.CAP_PROP_DEVICE_DESCRIPTION)
            if desc:
                return str(desc)
        except Exception:
            pass
    return "Unknown"

class CameraSearcher:
    """
    カメラデバイスを検索し、利用可能なポート番号やOAK-D-Liteを選択できるクラス
    """
    def __init__(self, max_ports=10):
        self.max_ports = max_ports
        self.available_ports = []

    def search(self):
        self.available_ports = []
        for camera_number in range(self.max_ports):
            cap = self.create_capture(camera_number)
            ret, _ = cap.read()
            cap.release()
            if ret:
                self.available_ports.append(camera_number)
        return self.available_ports

    def create_capture(self, camera_number):
        if sys.platform.startswith('win'):
            return cv2.VideoCapture(camera_number, cv2.CAP_DSHOW)
        elif sys.platform == 'darwin':
            return cv2.VideoCapture(camera_number, cv2.CAP_AVFOUNDATION)
        else:
            return cv2.VideoCapture(camera_number)

    def is_oakd_connected(self):
        if not OAKD_AVAILABLE:
            return False
        devices = dai.Device.getAllAvailableDevices()
        return len(devices) > 0

    def select_camera(self):
        ports = self.search()
        oakd_available = self.is_oakd_connected()
        print("利用可能なカメラポート番号:")
        for idx, port in enumerate(ports):
            cap = self.create_capture(port)
            name = get_camera_name(cap)
            cap.release()
            print(f"{idx}: Camera port {port} ({name})")
        if oakd_available:
            print(f"{len(ports)}: OAK-D-Lite (DepthAI)")

        # 各カメラの画像をライブで表示して選択
        for idx, port in enumerate(ports):
            cap = self.create_capture(port)
            window_name = f"Camera Preview {idx} (port {port})"
            print(f"カメラ {idx} (port {port}) のライブ映像を表示中。何かキーを押すと次へ進みます。")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                disp_frame = frame.copy()
                cv2.putText(disp_frame, f"Port: {port}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow(window_name, disp_frame)
                if cv2.waitKey(1) != -1:
                    break
            cv2.destroyWindow(window_name)
            cap.release()

        if oakd_available:
            self.preview_oakd()

        while True:
            try:
                sel = int(input("使用するカメラの番号を選択してください（例: 0）: "))
                if 0 <= sel < len(ports):
                    return ports[sel]
                elif oakd_available and sel == len(ports):
                    return "OAKD"
                else:
                    print("範囲外の番号です。")
            except ValueError:
                print("数字を入力してください。")

    def preview_oakd(self):
        # OAK-D-Liteの画像を一時的に取得して表示
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            print("OAK-D-Lite (DepthAI) のプレビューを表示中。何かキーを押すと次へ進みます。")
            while True:
                in_rgb = q_rgb.tryGet()
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()
                    cv2.putText(frame, "OAK-D-Lite", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.imshow("OAK-D-Lite Preview", frame)
                if cv2.waitKey(1) != -1:
                    break
            cv2.destroyWindow("OAK-D-Lite Preview")

# テスト用
if __name__ == '__main__':
    searcher = CameraSearcher()
    selected = searcher.select_camera()

    if selected == "OAKD":
        # OAK-D-Lite用の処理
        import depthai as dai
        import cv2

        pipeline = dai.Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            while True:
                in_rgb = q_rgb.tryGet()
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()
                    cv2.imshow("OAK-D-Lite Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
    elif selected is not None:
        # 通常のWebカメラ用の処理
        cap = cv2.VideoCapture(selected)
        print(f"選択されたカメラポート: {selected}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("カメラから映像を取得できません。")
                break
            cv2.imshow("Web Camera Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()