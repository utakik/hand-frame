import cv2

import mediapipe as mp

import math

import time

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

# ===== パラメータ =====

FRAME_PAD = 40          # グリップ点からどれだけ広げるか

MIN_FRAME_SIZE = 20     # これ未満の枠は無効

GRID_ROWS = 4

GRID_COLS = 4

SNAP_DIST_RATIO = 0.3   # セル対角に対するスナップ距離（小さいほど吸着弱）

IDLE_SEC = 0.4          # スナップ位置で何秒静止したら「置く」か

MAX_MOVE_FOR_IDLE = 8.0 # 静止判定に許容する動き（px）

RECT_SMOOTH_ALPHA = 0.3       # 枠の位置／サイズスムージング（0〜1）

# フレームに対するポータルの倍率（1.0より小さいと小さくなる）

PORTAL_SCALE = 0.6

# 選択＆削除まわり

SELECT_OVERLAP_THRESH = 0.6   # ライブ枠とポータルの重なり率しきい値

SELECT_TIMEOUT_SEC    = 1.5   # 最後に選択されてから何秒で選択解除するか

PINCH_START_MIN_DIST  = 25.0  # ピンチ開始とみなす最低距離(px) ← 少し緩め

PINCH_CLOSE_RATIO     = 0.7   # 開始距離の何倍以下になったら「閉じた」とみなすか ← 緩め

PINCH_HOLD_FRAMES     = 2     # 何フレーム連続で閉じ状態なら削除するか

# ピンチ削除後に勝手に新ポータルを置かないためのクールダウン

PLACE_COOLDOWN_SEC    = 1.0   # 削除から何秒は「置く」を無効化するか

# ライブ中ポータルの透明度（0〜1）

LIVE_ALPHA = 0.8


# ===== ユーティリティ =====

def to_px(lm, w, h):

    return int(lm.x * w), int(lm.y * h)

def dist(p1, p2):

    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def rect_overlap_ratio(rect_a, rect_b):

    """rect_a を基準にした重なり率"""

    ax1, ay1, ax2, ay2 = rect_a

    bx1, by1, bx2, by2 = rect_b

    iw = min(ax2, bx2) - max(ax1, bx1)

    ih = min(ay2, by2) - max(ay1, by1)

    if iw <= 0 or ih <= 0:

        return 0.0

    inter = iw * ih

    area_a = max(ax2 - ax1, 1) * max(ay2 - ay1, 1)

    return inter / area_a

def get_grip_point(hand_lm, w, h):

    """人差し指MCPと親指CMCの中点を「握りポイント」とする"""

    lm = hand_lm.landmark

    ix_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    th_cmc = lm[mp_hands.HandLandmark.THUMB_CMC]

    gx = (ix_mcp.x + th_cmc.x) / 2.0

    gy = (ix_mcp.y + th_cmc.y) / 2.0

    return int(gx * w), int(gy * h)

def line_intersection(p1, p2, p3, p4):

    """2直線の交点（UI用）。平行なら None"""

    x1, y1 = p1

    x2, y2 = p2

    x3, y3 = p3

    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

    if abs(denom) < 1e-6:

        return None

    num_px = (x1*y2 - y1*x2)

    num_py = (x3*y4 - y3*x4)

    px = (num_px*(x3 - x4) - (x1 - x2)*num_py) / denom

    py = (num_px*(y3 - y4) - (y1 - y2)*num_py) / denom

    return int(px), int(py)

def alpha_blend(bg, fg, alpha):

    """bg と fg（同サイズのBGR画像）を α で合成"""

    return cv2.addWeighted(fg, alpha, bg, 1 - alpha, 0)


# ===== メイン =====

def main():

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # ポータル用：背面カメラ（無ければ黒画面になる）

    sub_cap = cv2.VideoCapture(1)

    sub_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)

    sub_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():

        print("フロントカメラが開けません")

        return

    lm_show = False

    prev_time = time.time()

    fps = 0.0

    rect_center = None   # (cx, cy)

    rect_size   = None   # (w, h)

    portals = []         # {rect:(x1,y1,x2,y2), img, base_size:(w,h), cell_id:(row,col)}

    occupied_cells = set()   # (row, col) すでにポータルが置かれているグリッド

    snap_stable_frames = 0

    last_snap_cell = None

    prev_center_for_idle = None

    # 選択＆ピンチ削除用

    selected_portal_index = None

    last_select_time = 0.0

    pinch_mode = "idle"          # "idle" or "armed"

    pinch_start_dist = 0.0

    pinch_small_frames = 0

    last_delete_time = -1e9      # すごく昔に削除されたことにしておく

    with mp_hands.Hands(

        max_num_hands=2,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5,

        model_complexity=1

    ) as hands:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("カメラからのフレーム取得失敗")

                break

            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            # ===== プレイエリア（中央90%） =====

            PLAY_LEFT   = int(w * 0.05)

            PLAY_RIGHT  = int(w * 0.95)

            PLAY_TOP    = int(h * 0.05)

            PLAY_BOTTOM = int(h * 0.95)

            # FPS

            now = time.time()

            dt = now - prev_time

            if dt > 0:

                inst = 1.0 / dt

                fps = inst if fps == 0.0 else fps*0.9 + inst*0.1

            prev_time = now

            # グリッド情報（プレイエリア内だけ）

            play_w = PLAY_RIGHT - PLAY_LEFT

            play_h = PLAY_BOTTOM - PLAY_TOP

            cell_w = play_w / GRID_COLS

            cell_h = play_h / GRID_ROWS

            cell_diag = math.hypot(cell_w, cell_h)

            snap_dist = cell_diag * SNAP_DIST_RATIO

            # MediaPipe

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb.flags.writeable = False

            results = hands.process(rgb)

            rgb.flags.writeable = True

            # 手情報

            hand_lms = {}

            if results.multi_hand_landmarks and results.multi_handedness:

                for hand_lm, handed in zip(results.multi_hand_landmarks,

                                           results.multi_handedness):

                    label = handed.classification[0].label   # 'Left' or 'Right'

                    hand_lms[label] = hand_lm

            both_hands_present = ("Left" in hand_lms and "Right" in hand_lms)

            new_center = None

            new_size   = None

            live_rect  = None

            # ===== 両手フレームの矩形計算 =====

            if both_hands_present:

                gl = get_grip_point(hand_lms["Left"],  w, h)

                gr = get_grip_point(hand_lms["Right"], w, h)

                xs = [gl[0], gr[0]]

                ys = [gl[1], gr[1]]

                x_min = max(0, min(xs) - FRAME_PAD)

                x_max = min(w, max(xs) + FRAME_PAD)

                y_min = max(0, min(ys) - FRAME_PAD)

                y_max = min(h, max(ys) + FRAME_PAD)

                if (x_max - x_min) >= MIN_FRAME_SIZE and (y_max - y_min) >= MIN_FRAME_SIZE:

                    new_center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

                    new_size   = (float(x_max - x_min), float(y_max - y_min))

            # ===== 枠のスムージング =====

            if new_center is not None and new_size is not None:

                if rect_center is None:

                    rect_center = new_center

                    rect_size   = new_size

                else:

                    rect_center = (

                        rect_center[0] + RECT_SMOOTH_ALPHA * (new_center[0] - rect_center[0]),

                        rect_center[1] + RECT_SMOOTH_ALPHA * (new_center[1] - rect_center[1])

                    )

                    rect_size = (

                        rect_size[0] + RECT_SMOOTH_ALPHA * (new_size[0] - rect_size[0]),

                        rect_size[1] + RECT_SMOOTH_ALPHA * (new_size[1] - rect_size[1])

                    )

            # ===== グリッドスナップ ＋ ライブ枠矩形 =====

            snapped = False

            snapped_cell = None   # (row, col, cx_cell, cy_cell)

            if rect_center is not None and rect_size is not None and both_hands_present:

                cx, cy = rect_center

                rw, rh = rect_size

                # いちばん近いセル中心を探す（プレイエリア内）

                best_d = 1e9

                best_cell = None

                for row in range(GRID_ROWS):

                    for col in range(GRID_COLS):

                        cx_cell = PLAY_LEFT + (col + 0.5) * cell_w

                        cy_cell = PLAY_TOP  + (row + 0.5) * cell_h

                        d = math.hypot(cx - cx_cell, cy - cy_cell)

                        if d < best_d:

                            best_d = d

                            best_cell = (row, col, cx_cell, cy_cell)

                if best_cell is not None and best_d < snap_dist:

                    snapped = True

                    snapped_cell = best_cell

                    cx, cy = best_cell[2], best_cell[3]   # セル中心に寄せる

                # ★ ポータル用に少し縮めたサイズを使う

                rw_scaled = rw * PORTAL_SCALE

                rh_scaled = rh * PORTAL_SCALE

                # ライブ枠の矩形（縮小版）

                x1 = int(cx - rw_scaled / 2.0)

                x2 = int(cx + rw_scaled / 2.0)

                y1 = int(cy - rh_scaled / 2.0)

                y2 = int(cy + rh_scaled / 2.0)

                live_rect = (x1, y1, x2, y2)

            # ===== 既存ポータルの描画 =====

            for idx, p in enumerate(portals):

                px1, py1, px2, py2 = p["rect"]

                pw = px2 - px1

                ph = py2 - py1

                if pw <= 0 or ph <= 0:

                    continue

                img = cv2.resize(p["img"], (pw, ph))

                # 画面クリップ（フルフレーム基準）

                x1c = max(0, px1)

                y1c = max(0, py1)

                x2c = min(w, px2)

                y2c = min(h, py2)

                if x2c <= x1c or y2c <= y1c:

                    continue

                frame[y1c:y2c, x1c:x2c] = img[

                    (y1c - py1):(y2c - py1),

                    (x1c - px1):(x2c - px1)

                ]

                # ポータル枠

                color = (0, 200, 255)

                if idx == selected_portal_index:

                    color = (0, 255, 255)  # 選択中は明るいシアン寄り

                cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)

            # ===== ライブプレビューと「置く」「選択」を処理 =====

            if live_rect is not None and both_hands_present:

                # サブカメラの 1 フレームを取得（失敗したら黒）

                if sub_cap.isOpened():

                    ret2, sub_frame = sub_cap.read()

                    if not ret2:

                        sub_frame = None

                else:

                    sub_frame = None

                lx1, ly1, lx2, ly2 = live_rect

                # クリップ（フルフレーム基準）

                x1c = max(0, lx1)

                y1c = max(0, ly1)

                x2c = min(w, lx2)

                y2c = min(h, ly2)

                if x2c > x1c and y2c > y1c:

                    lw = lx2 - lx1

                    lh = ly2 - ly1

                    if lw > 0 and lh > 0:

                        if sub_frame is not None:

                            sub_resized = cv2.resize(sub_frame, (lw, lh))

                        else:

                            sub_resized = 0 * frame[0:lh, 0:lw]

                        # === ライブ表示：透明でブレンド ===

                        roi = frame[y1c:y2c, x1c:x2c]

                        patch = sub_resized[

                            (y1c - ly1):(y2c - ly1),

                            (x1c - lx1):(x2c - lx1)

                        ]

                        if roi.shape[:2] == patch.shape[:2]:

                            blended = alpha_blend(roi, patch, LIVE_ALPHA)

                            frame[y1c:y2c, x1c:x2c] = blended

                        # ===== スナップ＋静止で「置く」（1セル1ポータル＋削除直後クールダウン） =====

                        cx_now = (lx1 + lx2) / 2.0

                        cy_now = (ly1 + ly2) / 2.0

                        if snapped and snapped_cell is not None:

                            row, col, _, _ = snapped_cell

                            cell_id = (row, col)

                            if prev_center_for_idle is not None:

                                mv = dist((cx_now, cy_now), prev_center_for_idle)

                            else:

                                mv = 0.0

                            # すでにそのセルにポータルがある or クールダウン中なら「置く」カウント停止

                            if (cell_id in occupied_cells) or ((now - last_delete_time) < PLACE_COOLDOWN_SEC):

                                snap_stable_frames = 0

                            else:

                                if last_snap_cell == snapped_cell and mv < MAX_MOVE_FOR_IDLE:

                                    snap_stable_frames += 1

                                else:

                                    snap_stable_frames = 1

                            last_snap_cell = snapped_cell

                            prev_center_for_idle = (cx_now, cy_now)

                        else:

                            snap_stable_frames = 0

                            last_snap_cell = None

                            prev_center_for_idle = None

                        idle_frames_required = max(5, int(IDLE_SEC * fps))

                        if snapped_cell is not None and snap_stable_frames >= idle_frames_required:

                            # 今の live_rect サイズ・場所で静止画ポータルを置く

                            row, col, _, _ = snapped_cell

                            cell_id = (row, col)

                            if cell_id not in occupied_cells and (now - last_delete_time) >= PLACE_COOLDOWN_SEC:

                                snap_img = sub_resized.copy()

                                portals.append({

                                    "rect": (lx1, ly1, lx2, ly2),

                                    "img":  snap_img,

                                    "base_size": (lw, lh),

                                    "cell_id": cell_id,

                                })

                                occupied_cells.add(cell_id)

                            snap_stable_frames = 0

                            last_snap_cell = None

                            prev_center_for_idle = None

                        # ===== ライブ枠線 =====

                        color_live = (0, 255, 255) if snapped else (120, 120, 120)

                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c),

                                      color_live, 2)

                        # ===== ポータル選択処理（ライブ枠と一番重なっているもの） =====

                        best_idx = None

                        best_ov = 0.0

                        for i, p in enumerate(portals):

                            ov = rect_overlap_ratio(p["rect"], live_rect)

                            if ov > best_ov:

                                best_ov = ov

                                best_idx = i

                        if best_idx is not None and best_ov >= SELECT_OVERLAP_THRESH:

                            selected_portal_index = best_idx

                            last_select_time = now

                        else:

                            # 一定時間選択が続かないなら解除

                            if selected_portal_index is not None:

                                if (now - last_select_time) > SELECT_TIMEOUT_SEC:

                                    selected_portal_index = None

                                    pinch_mode = "idle"

                                    pinch_small_frames = 0

            else:

                snap_stable_frames = 0

                last_snap_cell = None

                prev_center_for_idle = None

            # ===== 選択済みポータルに対する「片手ピンチ削除」 =====

            if selected_portal_index is not None and hand_lms:

                p = portals[selected_portal_index]

                # どの手でもいいので、任意の片手のピンチをチェック

                pinch_detected_this_frame = False

                for hand_label, hand_lm in hand_lms.items():

                    lm = hand_lm.landmark

                    ix_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    th_tip = lm[mp_hands.HandLandmark.THUMB_TIP]

                    ix_px = to_px(ix_tip, w, h)

                    th_px = to_px(th_tip, w, h)

                    # ★ ポータルの中にいるかどうかは見ない（選択済みならどこでもピンチ有効）

                    d = dist(ix_px, th_px)

                    pinch_detected_this_frame = True

                    if pinch_mode == "idle":

                        # ある程度開いた状態からスタート

                        if d >= PINCH_START_MIN_DIST:

                            pinch_mode = "armed"

                            pinch_start_dist = d

                            pinch_small_frames = 0

                    else:  # "armed"

                        if d < pinch_start_dist * PINCH_CLOSE_RATIO:

                            pinch_small_frames += 1

                        else:

                            pinch_small_frames = 0

                    break  # どれかの手で処理したら抜ける

                if not pinch_detected_this_frame:

                    # 指が離れたらピンチはリセット

                    pinch_mode = "idle"

                    pinch_small_frames = 0

                # 一定フレーム以上「閉じた」状態が続いたら削除

                if pinch_small_frames >= PINCH_HOLD_FRAMES and pinch_mode == "armed":

                    cell_id = p.get("cell_id")

                    if cell_id in occupied_cells:

                        occupied_cells.remove(cell_id)

                    portals.pop(selected_portal_index)

                    selected_portal_index = None

                    pinch_mode = "idle"

                    pinch_small_frames = 0

                    last_delete_time = now  # ★ ここでクールダウン開始

                    # 削除後はスナップ系も一応リセット

                    snap_stable_frames = 0

                    last_snap_cell = None

                    prev_center_for_idle = None

            else:

                # 選択中でなければピンチ状態もリセット

                pinch_mode = "idle"

                pinch_small_frames = 0

            # ===== プレイエリア内のグリッド線描画 =====

            # 外枠

            cv2.rectangle(frame,

                          (PLAY_LEFT, PLAY_TOP),

                          (PLAY_RIGHT, PLAY_BOTTOM),

                          (80, 80, 80), 1)

            # 内側の区切り

            for gx in range(1, GRID_COLS):

                x = int(PLAY_LEFT + gx * cell_w)

                cv2.line(frame, (x, PLAY_TOP), (x, PLAY_BOTTOM), (80, 80, 80), 1)

            for gy in range(1, GRID_ROWS):

                y = int(PLAY_TOP + gy * cell_h)

                cv2.line(frame, (PLAY_LEFT, y), (PLAY_RIGHT, y), (80, 80, 80), 1)

            # ===== ここでフレームUIを「最前面」に再描画 =====

            for hand_label, hand_lm in hand_lms.items():

                lm = hand_lm.landmark

                ix_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                ix_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                th_tip = lm[mp_hands.HandLandmark.THUMB_TIP]

                th_ip  = lm[mp_hands.HandLandmark.THUMB_IP]

                ix_tip_px = to_px(ix_tip, w, h)

                ix_mcp_px = to_px(ix_mcp, w, h)

                th_tip_px = to_px(th_tip, w, h)

                th_ip_px  = to_px(th_ip,  w, h)

                # 線

                cv2.line(frame, ix_mcp_px, ix_tip_px, (0,255,0), 2)

                cv2.line(frame, th_ip_px,  th_tip_px, (255,0,0), 2)

                # 交点（フレームの角）

                corner = line_intersection(ix_mcp_px, ix_tip_px, th_ip_px, th_tip_px)

                if corner is not None:

                    cv2.circle(frame, corner, 4, (0,0,255), -1)

            # ===== UI =====

            sel_text = f"SEL:{selected_portal_index}" if selected_portal_index is not None else "SEL:-"

            ui = f"FPS:{fps:.1f}  PORTALS:{len(portals)}  {sel_text}"

            cv2.putText(

                frame,

                ui,

                (10, 25),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2

            )

            cv2.putText(

                frame,

                "q:quit  l:landmarks  （中央90%グリッド内でスナップ静止→置く / 選択 + 片手ピンチ→削除）",

                (10, h - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                (180, 180, 180),

                1

            )

            # ランドマーク表示（骨格）

            if lm_show and results.multi_hand_landmarks:

                for hand_lm in results.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("hand-grid (play area + pinch delete + alpha)", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):

                break

            elif key == ord('l'):

                lm_show = not lm_show

    cap.release()

    sub_cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
 
