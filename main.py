import cv2

import mediapipe as mp

import math

import time

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

# ====== パラメータ ======

Z_WEIGHT_ANGLE    = 0.5    # 3D角度のZ寄与（ポーズ判定用）

RECT_SMOOTH_ALPHA = 0.8    # 枠のなめらかさ（小ほどヌルヌル）←少し高めで追従よくする

STABLE_ON         = 3      # FRAME ON判定に必要な連続valid数

STABLE_OFF        = 6      # FRAME OFF判定の連続invalid数

MIN_FRAME_SIZE    = 15     # フレームの最小サイズ(px)

FRAME_PAD         = 0.8     # グリップ点からの余白（見かけの枠サイズ）

# === 加速度で「置く」用 ===

SPEED_SMOOTH_ALPHA = 0.7    # 速度のスムージング係数（0〜1）

PLACE_SPEED_THRESH = 350.0  # このスピード(px/sec)を超えたら置く

# === C案「縮めて消す」用 ===

GRAB_OVERLAP_THRESH   = 0.75  # フレームとポータルの重なり率がこれ以上なら「掴み」

SHRINK_DELETE_RATIO   = 0.15  # 元サイズの何割まで縮んだら消すか


# ========= ヘルパー =========

def to_px(lm, w, h):

    return int(lm.x * w), int(lm.y * h)


def dist(p1, p2):

    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def line_intersection(p1, p2, p3, p4):

    x1, y1 = p1

    x2, y2 = p2

    x3, y3 = p3

    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-6:

        return None

    num_px = (x1 * y2 - y1 * x2)

    num_py = (x3 * y4 - y3 * x4)

    px = (num_px * (x3 - x4) - (x1 - x2) * num_py) / denom

    py = (num_px * (y3 - y4) - (y1 - y2) * num_py) / denom

    return int(px), int(py)


def angle_between_3d(v1, v2):

    x1, y1, z1 = v1

    x2, y2, z2 = v2

    n1 = math.sqrt(x1*x1 + y1*y1 + z1*z1)

    n2 = math.sqrt(x2*x2 + y2*y2 + z2*z2)

    if n1 < 1e-6 or n2 < 1e-6:

        return 0.0

    cos_t = (x1*x2 + y1*y2 + z1*z2) / (n1 * n2)

    cos_t = max(-1.0, min(1.0, cos_t))

    return math.degrees(math.acos(cos_t))


def compute_hand_frame_info(hand_lm, handed_label, w, h):

    """

    ポーズが「フレームっぽいか」を判定する用。

    位置決めには後で MCP/CMC を使うので、

    ここでは TIPベースの角度＆長さだけ見る。

    """

    lm = hand_lm.landmark

    ix_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    ix_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    th_tip = lm[mp_hands.HandLandmark.THUMB_TIP]

    th_ip  = lm[mp_hands.HandLandmark.THUMB_IP]

    # 3Dベクトル（zはちょい弱め）

    v_index = (

        ix_tip.x - ix_mcp.x,

        ix_tip.y - ix_mcp.y,

        (ix_tip.z - ix_mcp.z) * Z_WEIGHT_ANGLE

    )

    v_thumb = (

        th_tip.x - th_ip.x,

        th_tip.y - th_ip.y,

        (th_tip.z - th_ip.z) * Z_WEIGHT_ANGLE

    )

    angle3d = angle_between_3d(v_index, v_thumb)

    # かなり広め（親指下向きも拾う）

    if not (10 <= angle3d <= 175):

        return {"valid": False}

    # ピクセル座標（可視化用）

    ix_tip_px = to_px(ix_tip, w, h)

    ix_mcp_px = to_px(ix_mcp, w, h)

    th_tip_px = to_px(th_tip, w, h)

    th_ip_px  = to_px(th_ip,  w, h)

    # 直線交点（角の可視化用）

    corner = line_intersection(ix_mcp_px, ix_tip_px, th_ip_px, th_tip_px)

    if corner is None:

        return {"valid": False}

    # 指の長さ判定 → 緩め

    hand_scale = max(dist(ix_mcp_px, th_ip_px), 1.0)

    if dist(ix_mcp_px, ix_tip_px) < hand_scale * 0.3:

        return {"valid": False}

    if dist(th_ip_px, th_tip_px) < hand_scale * 0.15:

        return {"valid": False}

    return {

        "valid": True,

        "corner": corner,

        "index_tip": ix_tip_px,

        "thumb_tip": th_tip_px,

        "index_mcp": ix_mcp_px,

        "thumb_ip": th_ip_px,

        "angle": angle3d,

    }


def get_grip_point(hand_lm, w, h):

    """

    剛体フレームの「握りポイント」を定義：

    人差し指MCPと親指CMCの中点。

    TIPよりブレが少ないので、フレームの芯に使う。

    """

    lm = hand_lm.landmark

    ix_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    th_cmc = lm[mp_hands.HandLandmark.THUMB_CMC]

    gx = (ix_mcp.x + th_cmc.x) / 2.0

    gy = (ix_mcp.y + th_cmc.y) / 2.0

    return int(gx * w), int(gy * h)


def rect_overlap_ratio(rect_a, rect_b):

    """

    2つの矩形の「Aに対する重なり割合」を返す。

    rect = (x1, y1, x2, y2)

    """

    ax1, ay1, ax2, ay2 = rect_a

    bx1, by1, bx2, by2 = rect_b

    iw = min(ax2, bx2) - max(ax1, bx1)

    ih = min(ay2, by2) - max(ay1, by1)

    if iw <= 0 or ih <= 0:

        return 0.0

    inter = iw * ih

    area_a = max(ax2 - ax1, 1) * max(ay2 - ay1, 1)

    return inter / area_a


# ========= メイン =========

def main():

    # Control Stream（フロントカメラ）

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Content Stream（ポータル用：背面カメラ or 別デバイス）

    sub_cap = cv2.VideoCapture(1)

    sub_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)

    sub_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():

        print("フロントカメラが開けません")

        return

    if not sub_cap.isOpened():

        print("背面カメラが開けません（小窓は黒のまま）")

    lm_show = False

    prev_time = time.time()

    fps = 0.0

    frame_on = False

    stable_count = 0

    lost_count = 0

    rect_center = None   # (cx, cy)

    rect_size   = None   # (w, h)

    # ポータルは 1個だけ管理

    portal = None        # {"rect":(x1,y1,x2,y2), "img":..., "base_size":(w0,h0), "grabbed":bool}

    # 加速度っぽいもの（スピード）用

    prev_center = None

    prev_center_t = None

    speed_smooth = 0.0

    # 「置ける状態かどうか」

    place_armed = False

    with mp_hands.Hands(

        max_num_hands=2,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5,

        model_complexity=1

    ) as hands:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("カメラからフレーム取得失敗")

                break

            # FPS

            now = time.time()

            dt = now - prev_time

            if dt > 0:

                inst = 1.0 / dt

                fps = inst if fps == 0.0 else fps*0.9 + inst*0.1

            prev_time = now

            # 鏡モード

            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            # MediaPipe

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb.flags.writeable = False

            results = hands.process(rgb)

            rgb.flags.writeable = True

            hands_info = {}

            hand_lms   = {}

            both_valid = False

            # 現在フレームの live_rect（両手フレームの矩形）。後で使い回す。

            live_rect = None

            # ==== 両手の解析 ====

            if results.multi_hand_landmarks and results.multi_handedness:

                for hand_lm, handed in zip(results.multi_hand_landmarks,

                                           results.multi_handedness):

                    label = handed.classification[0].label  # 'Left', 'Right'

                    lm = hand_lm.landmark

                    hand_lms[label] = hand_lm

                    # 軸線表示（TIPベース）

                    ix_tip_px = to_px(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)

                    ix_mcp_px = to_px(lm[mp_hands.HandLandmark.INDEX_FINGER_MCP], w, h)

                    th_tip_px = to_px(lm[mp_hands.HandLandmark.THUMB_TIP],      w, h)

                    th_ip_px  = to_px(lm[mp_hands.HandLandmark.THUMB_IP],       w, h)

                    cv2.line(frame, ix_mcp_px, ix_tip_px, (0,255,0), 2)

                    cv2.line(frame, th_ip_px, th_tip_px, (255,0,0), 2)

                    info = compute_hand_frame_info(hand_lm, label, w, h)

                    if info["valid"]:

                        hands_info[label] = info

                both_valid = ("Left" in hands_info and "Right" in hands_info

                              and "Left" in hand_lms and "Right" in hand_lms)

                # ==== 剛体フレーム：グリップ点ベースで枠候補 ====

                new_center = None

                new_size   = None

                if both_valid:

                    grip_left  = get_grip_point(hand_lms["Left"],  w, h)

                    grip_right = get_grip_point(hand_lms["Right"], w, h)

                    xs = [grip_left[0], grip_right[0]]

                    ys = [grip_left[1], grip_right[1]]

                    x_min = max(0, min(xs) - FRAME_PAD)

                    x_max = min(w, max(xs) + FRAME_PAD)

                    y_min = max(0, min(ys) - FRAME_PAD)

                    y_max = min(h, max(ys) + FRAME_PAD)

                    if (x_max - x_min) > MIN_FRAME_SIZE and (y_max - y_min) > MIN_FRAME_SIZE:

                        new_center = (

                            (x_min + x_max) / 2.0,

                            (y_min + y_max) / 2.0

                        )

                        new_size = (

                            float(x_max - x_min),

                            float(y_max - y_min)

                        )

                # ==== 枠スムージング ====

                if new_center is not None and new_size is not None:

                    if rect_center is None:

                        rect_center = new_center

                        rect_size   = new_size

                    else:

                        rect_center = (

                            rect_center[0] + RECT_SMOOTH_ALPHA*(new_center[0]-rect_center[0]),

                            rect_center[1] + RECT_SMOOTH_ALPHA*(new_center[1]-rect_center[1])

                        )

                        rect_size = (

                            rect_size[0] + RECT_SMOOTH_ALPHA*(new_size[0]-rect_size[0]),

                            rect_size[1] + RECT_SMOOTH_ALPHA*(new_size[1]-rect_size[1])

                        )

                    # この時点で live_rect を作っておく

                    cx, cy = rect_center

                    rw, rh = rect_size

                    x1 = int(cx - rw/2.0)

                    x2 = int(cx + rw/2.0)

                    y1 = int(cy - rh/2.0)

                    y2 = int(cy + rh/2.0)

                    live_rect = (x1, y1, x2, y2)

                # ==== ヒステリシス ====

                previous_frame_on = frame_on

                if both_valid and rect_center and rect_size:

                    stable_count += 1

                    lost_count = 0

                else:

                    stable_count = 0

                    lost_count += 1

                if stable_count >= STABLE_ON:

                    frame_on = True

                if lost_count >= STABLE_OFF:

                    frame_on = False

                # ONになった瞬間 → 置き動作の受付開始

                if frame_on and not previous_frame_on:

                    place_armed = True

                    prev_center = None

                    prev_center_t = None

                    speed_smooth = 0.0

                    # 掴み状態はリセット

                    if portal is not None:

                        portal["grabbed"] = False

                # OFFになった瞬間 → リセット

                if not frame_on and previous_frame_on:

                    place_armed = False

                    prev_center = None

                    prev_center_t = None

                    speed_smooth = 0.0

                    if portal is not None:

                        portal["grabbed"] = False

                # ランドマーク描画

                if lm_show and results.multi_hand_landmarks:

                    for hand_lm in results.multi_hand_landmarks:

                        mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                # 角点の可視化（TIPベース）

                for side, info in hands_info.items():

                    cx, cy = info["corner"]

                    cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)

                    cv2.circle(frame, info["index_tip"], 3, (0,255,0), -1)

                    cv2.circle(frame, info["thumb_tip"], 3, (255,0,0), -1)

            else:

                # 手が見えないとき

                previous_frame_on = frame_on

                lost_count += 1

                if lost_count >= STABLE_OFF:

                    frame_on = False

                if not frame_on and previous_frame_on:

                    place_armed = False

                    prev_center = None

                    prev_center_t = None

                    speed_smooth = 0.0

                    if portal is not None:

                        portal["grabbed"] = False

            # ======= まず「置かれているポータル」を描画 =======

            if portal is not None:

                x1, y1, x2, y2 = portal["rect"]

                ph = y2 - y1

                pw = x2 - x1

                if ph > 0 and pw > 0:

                    # クリップ

                    x1c = max(0, x1)

                    y1c = max(0, y1)

                    x2c = min(w, x2)

                    y2c = min(h, y2)

                    if x2c > x1c and y2c > y1c:

                        img = cv2.resize(portal["img"], (pw, ph))

                        frame[y1c:y2c, x1c:x2c] = img[

                            (y1c - y1):(y2c - y1),

                            (x1c - x1):(x2c - x1)

                        ]

                        # 掴んでいるときは枠をちょっと赤く縁取る

                        color = (0, 200, 255) if not portal.get("grabbed", False) else (0, 0, 255)

                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)

            # ======= C案：「ポータルを掴んで縮めて消す」ロジック =======

            if portal is not None and frame_on and live_rect is not None:

                # まだ掴んでいない → フレームがポータルを十分覆ったら掴む

                if not portal.get("grabbed", False):

                    overlap = rect_overlap_ratio(portal["rect"], live_rect)

                    if overlap > GRAB_OVERLAP_THRESH:

                        portal["grabbed"] = True

                        # 掴んだ瞬間にベースサイズを記録

                        x1, y1, x2, y2 = portal["rect"]

                        portal["base_size"] = (max(x2 - x1, 1), max(y2 - y1, 1))

                # 掴んでいる間はポータルの矩形を live_rect に追従させる

                if portal.get("grabbed", False):

                    # live_rect をそのまま使う（必要ならスムージングしても良い）

                    px1, py1, px2, py2 = live_rect

                    portal["rect"] = (px1, py1, px2, py2)

                    # どれくらい縮んだかチェック

                    bw, bh = portal.get("base_size", (1, 1))

                    w_now = max(px2 - px1, 1)

                    h_now = max(py2 - py1, 1)

                    size_ratio = min(w_now / bw, h_now / bh)

                    # 元サイズの一定割合以下になったら消す

                    if size_ratio < SHRINK_DELETE_RATIO:

                        portal = None  # 消滅

                        place_armed = False

                        prev_center = None

                        prev_center_t = None

                        speed_smooth = 0.0

            # ======= ライブのポータル描画＋加速度トリガー（置く） =======

            current_speed = 0.0

            if frame_on and rect_center and rect_size and sub_cap.isOpened():

                ret2, sub_frame = sub_cap.read()

                if not ret2:

                    sub_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    ret2, sub_frame = sub_cap.read()

                if ret2 and live_rect is not None:

                    x1, y1, x2, y2 = live_rect

                    # クリップ

                    x1c = max(0, x1)

                    y1c = max(0, y1)

                    x2c = min(w, x2)

                    y2c = min(h, y2)

                    if x2c > x1c and y2c > y1c:

                        sub_resized = cv2.resize(sub_frame, (x2 - x1, y2 - y1))

                        frame[y1c:y2c, x1c:x2c] = sub_resized[

                            (y1c - y1):(y2c - y1),

                            (x1c - x1):(x2c - x1)

                        ]

                        live_img = sub_resized.copy()

                        # ===== 枠中心のスピードを計算 =====

                        if prev_center is not None and prev_center_t is not None:

                            dt_center = now - prev_center_t

                            if dt_center > 0:

                                dx = rect_center[0] - prev_center[0]

                                dy = rect_center[1] - prev_center[1]

                                speed = math.hypot(dx, dy) / dt_center  # px/sec

                                # スムージング

                                speed_smooth = (1.0 - SPEED_SMOOTH_ALPHA) * speed_smooth + SPEED_SMOOTH_ALPHA * speed

                                current_speed = speed_smooth

                        prev_center   = rect_center

                        prev_center_t = now

                        # ===== 一定スピードを超えたら「置く」 =====

                        if place_armed and current_speed > PLACE_SPEED_THRESH:

                            # ポータルは 1個に限定するので上書き

                            portal = {

                                "rect": live_rect,

                                "img":  live_img,

                                "grabbed": False

                            }

                            # base_size は掴んだ瞬間にセットする

                            place_armed = False

                        # ライブ枠線（置く前のプレビュー）

                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0,255,255), 2)

            # ===== UI =====

            lm_text   = "ON" if lm_show else "OFF"

            fr_text   = "ON" if frame_on else "OFF"

            arm_text  = "ARMED" if place_armed else "OFF"

            has_portal = "YES" if portal is not None else "NO"

            grabbed_text = "ON" if (portal is not None and portal.get("grabbed", False)) else "OFF"

            ui = f"FPS:{fps:.1f}  LM:{lm_text}  FRAME:{fr_text}  PLACE:{arm_text}  PORTAL:{has_portal}  GRAB:{grabbed_text}  SPD:{current_speed:.0f}"

            cv2.putText(frame, ui, (10,25),

                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.putText(

                frame,

                "q:quit  l:landmarks  （両手フレーム→素早く動かすと置く / かぶせて縮めると消える）",

                (10,h-10),

                cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1

            )

            cv2.imshow("hand-frame portal C-mode (shrink to delete)", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('l'):

                lm_show = not lm_show

            elif key == ord('q'):

                break

    cap.release()

    sub_cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
 
