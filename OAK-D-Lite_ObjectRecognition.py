import cv2  # OpenCVライブラリ
import depthai as dai  # DepthAIライブラリ
import numpy as np  # 数値計算ライブラリ
import blobconverter  # blob自動ダウンロード用
from pyzbar import pyzbar  # バーコード検出用

# --- QRコード検出クラス ---
class QRCodeFinder:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()  # OpenCVのQRコード検出器

    def find_and_draw(self, frame):
        data, points, _ = self.detector.detectAndDecode(frame)  # QRコード検出・デコード
        if points is not None and data:
            pts = points[0].astype(int)  # 頂点座標
            cv2.polylines(frame, [pts], True, (255,0,0), 2)  # 青色で枠線描画
            cv2.putText(frame, f"QR: {data}", (pts[0][0], pts[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)  # データ表示
        return frame

# --- バーコード検出クラス ---
class BarcodeFinder:
    def __init__(self):
        pass  # 特に初期化不要

    def find_and_draw(self, frame):
        barcodes = pyzbar.decode(frame)  # バーコード検出
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect  # バーコード領域
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 赤枠
            barcode_data = barcode.data.decode("utf-8")  # データ
            barcode_type = barcode.type  # 種類
            cv2.putText(frame, f"{barcode_type}: {barcode_data}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 表示
        return frame

# --- YOLO物体認識クラス ---
class YoloObjectDetector:
    def __init__(self, labels, model_blob, preview_size=(640, 352)):
        self.labels = labels
        self.model_blob = model_blob
        self.preview_size = preview_size
        self.pipeline = self._create_pipeline()
        self.device = None

    def _create_pipeline(self):
        pipeline = dai.Pipeline()
        # カメラノード
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(*self.preview_size)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        # NNノード
        detection_nn = pipeline.createYoloDetectionNetwork()
        detection_nn.setBlobPath(self.model_blob)
        detection_nn.setConfidenceThreshold(0.5)
        detection_nn.setNumClasses(len(self.labels))
        detection_nn.setCoordinateSize(4)
        detection_nn.setIouThreshold(0.5)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)
        # ノード接続
        cam_rgb.preview.link(detection_nn.input)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        xout_nn = pipeline.createXLinkOut()
        xout_nn.setStreamName("detections")
        detection_nn.out.link(xout_nn.input)
        return pipeline

    def __enter__(self):
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_nn = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device is not None:
            self.device.close()

    def get_frame_and_detections(self):
        in_rgb = self.q_rgb.get()
        in_nn = self.q_nn.get()
        frame = in_rgb.getCvFrame()
        detections = in_nn.detections
        return frame, detections

    def draw_detections(self, frame, detections):
        label_counts = {}
        for det in detections:
            label_idx = det.label
            label = self.labels[label_idx] if label_idx < len(self.labels) else str(label_idx)
            label_counts[label] = label_counts.get(label, 0) + 1
            x1 = int(det.xmin * frame.shape[1])
            y1 = int(det.ymin * frame.shape[0])
            x2 = int(det.xmax * frame.shape[1])
            y2 = int(det.ymax * frame.shape[0])
            conf = det.confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # 画面左上にラベルごとの検出数を表示
        y_offset = 30
        for label, count in label_counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
        return frame

# --- 定数 ---
MODEL_BLOB = blobconverter.from_zoo(
    name="yolov8n_coco_640x352",
    zoo_type="depthai",
    shaves=6
)

LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# --- メイン処理 ---
def main():
    qr_finder = QRCodeFinder()      # QRコード検出器
    barcode_finder = BarcodeFinder()  # バーコード検出器

    with YoloObjectDetector(LABELS, MODEL_BLOB) as detector:
        while True:
            frame, detections = detector.get_frame_and_detections()
            frame = detector.draw_detections(frame, detections)
            frame = qr_finder.find_and_draw(frame)
            frame = barcode_finder.find_and_draw(frame)
            cv2.imshow("OAK-D-Lite YOLOv8 Object Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()