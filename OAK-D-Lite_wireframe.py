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
        self.parent = parent
        self.brightness = max_cm  # 初期値100
        self.camera_name = camera_name
        self.setup_ui()

    def setup_ui(self):
        self.parent.setWindowTitle("Wireframe/Edge/PointCloud Depth Viewer")
        self.parent.setFixedSize(1280, 800)

        # 原画像表示
        self.rgb_label = QtWidgets.QLabel(self.parent)
        self.rgb_label.setGeometry(0, 0, 640, 480)
        self.rgb_label.setStyleSheet("background-color: black;")

        # ワイヤーフレーム/輪郭/点群画像表示
        self.depth_label = QtWidgets.QLabel(self.parent)
        self.depth_label.setGeometry(640, 0, 640, 480)
        self.depth_label.setStyleSheet("background-color: black;")

        # 下部ラベル（カメラ名も表示）
        self.info_label = QtWidgets.QLabel(self.parent)
        self.info_label.setGeometry(0, 500, 1280, 40)
        self.info_label.setText(
            f"カメラ: {self.camera_name}　|　RGB: 原画像 / Depth: ワイヤーフレーム画像"
        )
        self.info_label.setStyleSheet("color: white; background-color: #222; font-size: 20px;")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)

        # スライダー（輝度調整用に変更）
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.parent)
        self.slider.setGeometry(100, 600, 1080, 40)
        self.slider.setMinimum(10)
        self.slider.setMaximum(300)
        self.slider.setValue(self.brightness)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

        # スライダー値表示
        self.slider_label = QtWidgets.QLabel(self.parent)
        self.slider_label.setGeometry(0, 640, 1280, 40)
        self.slider_label.setAlignment(QtCore.Qt.AlignCenter)
        self.slider_label.setStyleSheet("color: white; background-color: #333; font-size: 18px;")
        self.slider_label.setText(f"輝度調整: {self.brightness}")

        # 表示切替ボタン（3状態: ワイヤーフレーム/輪郭/点群）
        self.toggle_button = QtWidgets.QPushButton("Wireframe表示", self.parent)
        self.toggle_button.setGeometry(440, 700, 400, 40)
        self.toggle_button.setCheckable(False)  # クリックごとに切替
        self.toggle_states = ["Wireframe", "Edge", "PointCloud"]
        self.toggle_index = 0
        self.toggle_button.clicked.connect(self.cycle_mode)

    def cycle_mode(self):
        self.toggle_index = (self.toggle_index + 1) % len(self.toggle_states)
        mode = self.toggle_states[self.toggle_index]
        if mode == "Wireframe":
            self.toggle_button.setText("Wireframe表示")
            self.info_label.setText(
                f"カメラ: {self.camera_name}　|　RGB: 原画像 / Depth: ワイヤーフレーム画像"
            )
        elif mode == "Edge":
            self.toggle_button.setText("Edge表示")
            self.info_label.setText(
                f"カメラ: {self.camera_name}　|　RGB: 原画像 / Depth: 輪郭抽出画像"
            )
        elif mode == "PointCloud":
            self.toggle_button.setText("PointCloud表示")
            self.info_label.setText(
                f"カメラ: {self.camera_name}　|　RGB: 原画像 / Depth: 点群画像"
            )

class DepthViewer(QtWidgets.QWidget):
    def __init__(self, preview_size=(640, 480), brightness=100):
        super().__init__()
        self.preview_size = preview_size
        self.brightness = brightness  # 輝度
        self.color_map = self._create_colormap(100)  # ダミー
        self.over_color = (128, 128, 128)  # max_cm以上はグレー

        # カメラ名を決定
        if CAMERA_TYPE == "oak":
            camera_name = "OAK-D-Lite"
        elif CAMERA_TYPE == "realsense":
            camera_name = "Intel RealSense D415"
        else:
            camera_name = "Unknown"

        self.ui = OAKDLiteUI(self, self.brightness, camera_name)
        self.ui.slider.valueChanged.connect(self.on_slider_changed)
        self.ui.toggle_button.clicked.connect(self.on_toggle_changed)
        self.show_mode = 0  # 0:ワイヤーフレーム, 1:輪郭, 2:点群

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
        # ダミー（未使用）
        return np.zeros((max_cm, 3), dtype=np.uint8)

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
        self.brightness = value
        self.ui.slider_label.setText(f"輝度調整: {self.brightness}")

    def on_toggle_changed(self):
        # UI側のインデックスを進めてラベル更新
        self.ui.cycle_mode()
        # UIのインデックスと同期
        self.show_mode = self.ui.toggle_index

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
            # 輝度調整
            depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = depth_norm.astype(np.uint8)
            alpha = self.brightness / 100.0  # 100で等倍
            depth_bright = cv2.convertScaleAbs(depth_norm, alpha=alpha, beta=0)

            if self.show_mode == 0:
                out_img = self._depth_to_wireframe(depth_bright)
            elif self.show_mode == 1:
                out_img = self._depth_to_edge(depth_bright)
            elif self.show_mode == 2:
                out_img = self._depth_to_pointcloud(depth_bright, frame)
            else:
                out_img = np.zeros((depth_bright.shape[0], depth_bright.shape[1], 3), dtype=np.uint8)

            out_img = cv2.resize(out_img, self.preview_size)
            out_qimg = QtGui.QImage(out_img.data, out_img.shape[1], out_img.shape[0], out_img.strides[0], QtGui.QImage.Format_BGR888)
            self.ui.depth_label.setPixmap(QtGui.QPixmap.fromImage(out_qimg))

    def _depth_to_wireframe(self, depth_img):
        """輝度調整済みデプス画像からワイヤーフレーム画像を生成"""
        edges = cv2.Canny(depth_img, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        wireframe = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(wireframe, contours, -1, (0, 255, 0), 1)  # 緑色でワイヤーフレーム
        return wireframe

    def _depth_to_edge(self, depth_img):
        """輝度調整済みデプス画像から鮮明な輪郭抽出画像を生成"""
        blurred = cv2.GaussianBlur(depth_img, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 30, 100)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        edge_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        edge_img[edges > 0] = (0, 0, 255)
        return edge_img

    def _depth_to_pointcloud(self, depth_img, rgb_img=None):
        """輝度調整済みデプス画像から点群画像を生成（2D投影の簡易可視化）"""
        # 点群の2D投影を画像上に描画（本格的な3D表示ではありません）
        points = np.column_stack(np.where(depth_img > 0))
        pc_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        if rgb_img is not None:
            rgb_small = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]))
            for y, x in points:
                color = rgb_small[y, x].tolist()
                pc_img[y, x] = color
        else:
            pc_img[points[:, 0], points[:, 1]] = (255, 255, 255)
        return pc_img

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
    viewer = DepthViewer(preview_size=(640, 480), brightness=100)
    viewer.show()
    viewer.start()
    sys.exit(app.exec_())