import sys
import cv2
import mediapipe as mp
import depthai as dai

# --- OAK-D-Liteカメラ取得クラス ---
class OakCamera:
    def __init__(self, width=640, height=480):
        self.pipeline = dai.Pipeline()
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(width, height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    def get_frame(self):
        in_rgb = self.q_rgb.tryGet()
        if in_rgb is not None:
            return in_rgb.getCvFrame()
        return None

    def release(self):
        self.device.close()

# --- Mediapipeハンドトラッキングクラス ---
class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_and_draw(self, frame):
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        tip_ids = [4, 8, 12, 16, 20]  # 各指のTIPランドマークID

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                h, w, _ = frame.shape
                for name, idx in zip(finger_names, tip_ids):
                    lm = hand_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(frame, name, (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return frame

    def release(self):
        self.hands.close()

# --- メイン処理 ---
def main():
    camera = OakCamera()
    hand_tracker = HandTracker()

    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                frame = hand_tracker.process_and_draw(frame)
                cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        hand_tracker.release()
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()