import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

# カメラ種別選択: "oak" or "realsense"
CAMERA_TYPE = "oak"
if len(sys.argv) > 1:
    CAMERA_TYPE = sys.argv[1].lower()

class OAKDLiteUI:
    def __init__(self, parent, max_cm, camera_name=""):
        # UI部品の初期化
        self.parent = parent
        self.max_cm = max_cm
        self.camera_name = camera_name
        self.setup_ui()

    def setup_ui(self):
        # ウィンドウやラベル、スライダーなどUI部品の配置・初期化
        self.parent.setWindowTitle("Depth Viewer")
        self.parent.setFixedSize(1280, 720)

        # 原画像表示
        self.rgb_label = QtWidgets.QLabel(self.parent)
        self.rgb_label.setGeometry(0, 0, 640, 480)
        self.rgb_label.setStyleSheet("background-color: black;")

        # デプス画像表示
        self.depth_label = QtWidgets.QLabel(self.parent)
        self.depth_label.setGeometry(640, 0, 640, 480)
        self.depth_label.setStyleSheet("background-color: black;")

        # 下部ラベル（カメラ名も表示）
        self.info_label = QtWidgets.QLabel(self.parent)
        self.info_label.setGeometry(0, 500, 1280, 40)
        self.info_label.setText(
            f"カメラ: {self.camera_name}　|　RGB: 原画像 / Depth: 距離画像（1cm色分け）"
        )
        self.info_label.setStyleSheet("color: white; background-color: #222; font-size: 20px;")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)

        # スライダー（最大距離調整）
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.parent)
        self.slider.setGeometry(100, 600, 1080, 40)
        self.slider.setMinimum(10)
        self.slider.setMaximum(300)
        self.slider.setValue(self.max_cm)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

        # スライダー値表示
        self.slider_label = QtWidgets.QLabel(self.parent)
        self.slider_label.setGeometry(0, 640, 1280, 40)
        self.slider_label.setAlignment(QtCore.Qt.AlignCenter)
        self.slider_label.setStyleSheet("color: white; background-color: #333; font-size: 18px;")
        self.slider_label.setText(f"デプス色分け最大距離: {self.max_cm} cm")

class DepthViewer(QtWidgets.QWidget):
    def __init__(self, preview_size=(640, 480), max_cm=100):
        super().__init__()
        self.preview_size = preview_size
        self.max_cm = max_cm
        self.color_map = self._create_colormap(self.max_cm)
        self.over_color = (128, 128, 128)  # max_cm以上はグレー

        # カメラ名を決定
        if CAMERA_TYPE == "oak":
            camera_name = "OAK-D-Lite"
        elif CAMERA_TYPE == "realsense":
            camera_name = "Intel RealSense D415"
        else:
            camera_name = "Unknown"

        self.ui = OAKDLiteUI(self, self.max_cm, camera_name)
        self.ui.slider.valueChanged.connect(self.on_slider_changed)

        # カメラ種別ごとに初期化
        if CAMERA_TYPE == "oak":
            import depthai as dai
            self.dai = dai
            self.pipeline = self._create_pipeline_oak()
            self.device = self.dai.Device(self.pipeline)
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            self.get_frame = self._get_frame_oak
        elif CAMERA_TYPE == "realsense":
            import pyrealsense2 as rs
            self.rs = rs
            self.pipeline = self.rs.pipeline()
            config = self.rs.config()
            config.enable_stream(self.rs.stream.color, 640, 480, self.rs.format.bgr8, 30)
            config.enable_stream(self.rs.stream.depth, 640, 480, self.rs.format.z16, 30)
            self.profile = self.pipeline.start(config)
            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            self.get_frame = self._get_frame_realsense
        else:
            raise ValueError("カメラタイプは 'oak' または 'realsense' を指定してください")

    def _create_colormap(self, max_cm):
        # 1cmごとの色分け用カラーマップを作成
        color_map = np.zeros((max_cm, 3), dtype=np.uint8)
        for i in range(max_cm):
            hsv = np.uint8([[[int(i * 179 / (max_cm - 1)), 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            color_map[i] = bgr
        return color_map

    # --- OAK-D-Lite用 ---
    def _create_color_camera(self, pipeline):
        dai = self.dai
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(*self.preview_size)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        return cam_rgb

    def _create_mono_camera(self, pipeline, socket):
        dai = self.dai
        mono = pipeline.createMonoCamera()
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono.setBoardSocket(socket)
        return mono

    def _create_stereo_depth(self, pipeline, mono_left, mono_right):
        dai = self.dai
        stereo = pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        return stereo

    def _create_xlinkout(self, pipeline, name, source):
        xout = self.dai.Pipeline().createXLinkOut()
        xout.setStreamName(name)
        source.link(xout.input)
        return xout

    def _create_pipeline_oak(self):
        dai = self.dai
        pipeline = dai.Pipeline()
        cam_rgb = self._create_color_camera(pipeline)
        mono_left = self._create_mono_camera(pipeline, dai.CameraBoardSocket.LEFT)
        mono_right = self._create_mono_camera(pipeline, dai.CameraBoardSocket.RIGHT)
        stereo = self._create_stereo_depth(pipeline, mono_left, mono_right)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        return pipeline

    def _get_frame_oak(self):
        in_rgb = self.q_rgb.tryGet()
        in_depth = self.q_depth.tryGet()
        frame = None
        depth_frame = None
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        if in_depth is not None:
            depth_frame = in_depth.getFrame()  # mm単位
        return frame, depth_frame

    # --- RealSense用 ---
    def _get_frame_realsense(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        frame = None
        depth = None
        if color_frame:
            frame = np.asanyarray(color_frame.get_data())
        if depth_frame:
            # RealSenseはm単位なのでcm単位に変換
            depth = np.asanyarray(depth_frame.get_data()) * self.depth_scale * 100  # cm
            depth = depth.astype(np.uint16)
        return frame, depth

    def on_slider_changed(self, value):
        self.max_cm = value
        self.color_map = self._create_colormap(self.max_cm)
        self.ui.slider_label.setText(f"デプス色分け最大距離: {self.max_cm} cm")

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(50)

    def update_frames(self):
        frame, depth_frame = self.get_frame()

        if frame is not None:
            frame_resized = cv2.resize(frame, self.preview_size)
            rgb_img = QtGui.QImage(frame_resized.data, frame_resized.shape[1], frame_resized.shape[0], frame_resized.strides[0], QtGui.QImage.Format_BGR888)
            self.ui.rgb_label.setPixmap(QtGui.QPixmap.fromImage(rgb_img))

        if depth_frame is not None:
            # OAK: mm, RealSense: cm
            if CAMERA_TYPE == "oak":
                bins = np.clip(depth_frame // 10, 0, self.max_cm)
            else:
                bins = np.clip(depth_frame, 0, self.max_cm)
            depth_color = np.zeros((*depth_frame.shape, 3), dtype=np.uint8)
            mask_over = bins == self.max_cm
            mask_valid = bins < self.max_cm
            depth_color[mask_valid] = self.color_map[bins[mask_valid]]
            depth_color[mask_over] = self.over_color
            depth_color = cv2.resize(depth_color, self.preview_size)
            depth_img = QtGui.QImage(depth_color.data, depth_color.shape[1], depth_color.shape[0], depth_color.strides[0], QtGui.QImage.Format_BGR888)
            self.ui.depth_label.setPixmap(QtGui.QPixmap.fromImage(depth_img))

    def closeEvent(self, event):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if CAMERA_TYPE == "oak" and hasattr(self, 'device'):
            self.device.close()
        elif CAMERA_TYPE == "realsense" and hasattr(self, 'pipeline'):
            self.pipeline.stop()
        event.accept()

    def keyPressEvent(self, event):
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = DepthViewer(preview_size=(640, 480), max_cm=100)
    viewer.show()
    viewer.start()
    sys.exit(app.exec_())