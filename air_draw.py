import cv2

import mediapipe as mp

import numpy as np

import time

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

# カメラ設定

CAM_INDEX = 0

WIDTH  = 640

HEIGHT = 480

# 線の設定

DRAW_COLOR = (0, 255, 255)  # 線の色（BGR）

DRAW_THICKNESS = 4          # 線の太さ


def main():

    cap = cv2.VideoCapture(CAM_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():

        print("カメラが開けません")

        return

    # 描画用キャンバス（真っ黒なレイヤー）

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    prev_point = None        # 一個前の指先座標

    draw_mode = True         # とりあえず常にON、キーで切り替え

    show_skeleton = False    # 骨格描画のON/OFF

    prev_time = time.time()

    fps = 0.0

    with mp_hands.Hands(

        max_num_hands=1,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5,

        model_complexity=1

    ) as hands:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("フレーム取得失敗")

                break

            # 鏡モード

            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            # FPS 計測

            now = time.time()

            dt = now - prev_time

            if dt > 0:

                inst = 1.0 / dt

                fps = inst if fps == 0.0 else fps*0.9 + inst*0.1

            prev_time = now

            # MediaPipe で手検出

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb.flags.writeable = False

            results = hands.process(rgb)

            rgb.flags.writeable = True

            index_point = None

            if results.multi_hand_landmarks:

                hand_lm = results.multi_hand_landmarks[0]

                lm = hand_lm.landmark

                # 人差し指TIP

                ix = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x

                iy = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                ix_px = int(ix * w)

                iy_px = int(iy * h)

                index_point = (ix_px, iy_px)

                # 骨格表示

                if show_skeleton:

                    mp_drawing.draw_landmarks(

                        frame, hand_lm, mp_hands.HAND_CONNECTIONS

                    )

                # 指先にマーカー

                cv2.circle(frame, index_point, 6, (0, 0, 255), -1)

            # ===== 線を描く処理 =====

            if draw_mode and index_point is not None:

                if prev_point is not None:

                    # 前フレームの位置と線で結ぶ

                    cv2.line(canvas, prev_point, index_point,

                             DRAW_COLOR, DRAW_THICKNESS)

                prev_point = index_point

            else:

                # 手が見えない or draw_mode OFF のときは線をつなげない

                prev_point = None

            # キャンバスとカメラ映像を合成

            combined = frame.copy()

            # 線を上に重ねる（単純に加算 or 置き換えでもOK）

            mask = canvas > 0

            combined[mask] = canvas[mask]

            # UI表示

            draw_text = "ON" if draw_mode else "OFF"

            skel_text = "ON" if show_skeleton else "OFF"

            ui = f"FPS:{fps:.1f}  DRAW:{draw_text}  SKEL:{skel_text}"

            cv2.putText(combined, ui, (10, 25),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,

                        (255, 255, 255), 2)

            cv2.putText(

                combined,

                "q:quit  d:toggle draw  c:clear  l:toggle landmarks",

                (10, h - 10),

                cv2.FONT_HERSHEY_SIMPLEX, 0.5,

                (200, 200, 200), 1

            )

            cv2.imshow("air draw", combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):

                break

            elif key == ord('d'):

                draw_mode = not draw_mode

            elif key == ord('c'):

                canvas[:] = 0

            elif key == ord('l'):

                show_skeleton = not show_skeleton

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
 
