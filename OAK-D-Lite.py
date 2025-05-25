import sys
import depthai as dai
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

class OAKDLiteViewer(QtWidgets.QWidget):
    def __init__(self, preview_size=(640, 480), max_cm=100):
        super().__init__()
        self.preview_size = preview_size
        self.max_cm = max_cm
        self.color_map = self._create_colormap(self.max_cm)
        self.over_color = (128, 128, 128)  # max_cm以上はグレー
        self.pipeline = self._create_pipeline()
        self.init_ui()

    def _create_colormap(self, max_cm):
        color_map = np.zeros((max_cm, 3), dtype=np.uint8)
        for i in range(max_cm):
            hsv = np.uint8([[[int(i * 179 / (max_cm - 1)), 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            color_map[i] = bgr
        return color_map

    def _create_pipeline(self):
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(*self.preview_size)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

        mono_left = pipeline.createMonoCamera()
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

        mono_right = pipeline.createMonoCamera()
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo = pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        return pipeline

    def init_ui(self):
        self.setWindowTitle("OAK-D-Lite Viewer")
        self.setFixedSize(1280, 720)

        # 原画像表示
        self.rgb_label = QtWidgets.QLabel(self)
        self.rgb_label.setGeometry(0, 0, 640, 480)
        self.rgb_label.setStyleSheet("background-color: black;")

        # デプス画像表示
        self.depth_label = QtWidgets.QLabel(self)
        self.depth_label.setGeometry(640, 0, 640, 480)
        self.depth_label.setStyleSheet("background-color: black;")

        # 下部ラベル
        self.info_label = QtWidgets.QLabel(self)
        self.info_label.setGeometry(0, 500, 1280, 40)
        self.info_label.setText("RGB: 原画像 / Depth: 距離画像（1cm色分け）")
        self.info_label.setStyleSheet("color: white; background-color: #222; font-size: 20px;")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)

        # スライダー（最大距離調整）
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setGeometry(100, 600, 1080, 40)
        self.slider.setMinimum(10)
        self.slider.setMaximum(300)
        self.slider.setValue(self.max_cm)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.on_slider_changed)

        # スライダー値表示
        self.slider_label = QtWidgets.QLabel(self)
        self.slider_label.setGeometry(0, 640, 1280, 40)
        self.slider_label.setAlignment(QtCore.Qt.AlignCenter)
        self.slider_label.setStyleSheet("color: white; background-color: #333; font-size: 18px;")
        self.slider_label.setText(f"デプス色分け最大距離: {self.max_cm} cm")

    def on_slider_changed(self, value):
        self.max_cm = value
        self.color_map = self._create_colormap(self.max_cm)
        self.slider_label.setText(f"デプス色分け最大距離: {self.max_cm} cm")

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frames)

        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        self.timer.start(50)

    def update_frames(self):
        updated = False

        in_rgb = self.q_rgb.tryGet()
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            frame = cv2.resize(frame, self.preview_size)
            rgb_img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_BGR888)
            self.rgb_label.setPixmap(QtGui.QPixmap.fromImage(rgb_img))
            updated = True

        in_depth = self.q_depth.tryGet()
        if in_depth is not None:
            depth_frame = in_depth.getFrame()  # mm単位
            bins = np.clip(depth_frame // 10, 0, self.max_cm)
            depth_color = np.zeros((*depth_frame.shape, 3), dtype=np.uint8)
            mask_over = bins == self.max_cm
            mask_valid = bins < self.max_cm
            depth_color[mask_valid] = self.color_map[bins[mask_valid]]
            depth_color[mask_over] = self.over_color
            depth_color = cv2.resize(depth_color, self.preview_size)
            depth_img = QtGui.QImage(depth_color.data, depth_color.shape[1], depth_color.shape[0], depth_color.strides[0], QtGui.QImage.Format_BGR888)
            self.depth_label.setPixmap(QtGui.QPixmap.fromImage(depth_img))
            updated = True

        if updated:
            self.timer.start(50)

    def closeEvent(self, event):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'device'):
            self.device.close()
        event.accept()

    def keyPressEvent(self, event):
        # 任意のキーでウィンドウを閉じる
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = OAKDLiteViewer(preview_size=(640, 480), max_cm=100)
    viewer.show()
    viewer.start()
    sys.exit(app.exec_())