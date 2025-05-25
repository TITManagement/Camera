import sys
import cv2
import mediapipe as mp

# カメラ種別選択: "oak" のみ対応
CAMERA_TYPE = "oak"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

import depthai as dai
# OAK-D-Lite用パイプライン
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

device = dai.Device(pipeline)
q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
def get_frame():
    in_rgb = q_rgb.tryGet()
    if in_rgb is not None:
        return in_rgb.getCvFrame()
    return None

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        frame = get_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# リソース解放
cv2.destroyAllWindows()