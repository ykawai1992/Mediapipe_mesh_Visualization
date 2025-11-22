import cv2
import mediapipe as mp
import sys
import time
import numpy as np

# MediaPipeとOpenCVのバージョン確認 (オプション)
print(f"Pythonバージョン: {sys.version}")
print(f"MediaPipeバージョン: {mp.__version__}")
print(f"OpenCVバージョン: {cv2.__version__}")
print("---------------------------------------")

# ----------------------------------------------------
# 0. 初期設定と関数定義
# ----------------------------------------------------

# MediaPipe Face Meshの初期設定
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# max_num_faces=1で1つの顔のみを検出
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, # 精度向上
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 貼り付けるイラスト画像を読み込むと、白背景透過処理
try:
    # 透明度(アルファチャネル)付きのPNGを優先して読み込む
    sticker_img = cv2.imread('sticker.png', cv2.IMREAD_UNCHANGED)
    if sticker_img is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("エラー: 'sticker.png' が見つかりません。")
    print("顔に貼り付けるための画像を準備し、'sticker.png'として保存してください。")
    sys.exit(1)

# 画像が3チャンネルの場合、白背景透過処理を行う
if sticker_img.shape[2] == 3:
    print("ステッカー画像にアルファチャンネルがないため、白背景を透過処理します。")
    
    # BGR画像をHSVに変換
    hsv = cv2.cvtColor(sticker_img, cv2.COLOR_BGR2HSV)

    # 白色の範囲を定義 (HSV: H=色相, S=彩度, V=明度)
    # ほぼ白（彩度が低く、明度が高い）の範囲を定義
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255]) 

    # 白色のマスクを作成
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # マスクを反転 (白以外の部分が不透明、白の部分が透明)
    alpha_channel = cv2.bitwise_not(mask)
    
    # 元のBGR画像にアルファチャンネルを追加して4チャンネル画像にする
    sticker_img = cv2.merge([sticker_img[:,:,0], sticker_img[:,:,1], sticker_img[:,:,2], alpha_channel])
    print(f"白背景透過処理後のチャンネル数: {sticker_img.shape[2]}")
elif sticker_img.shape[2] == 4:
    print("ステッカー画像は既にアルファチャンネルを含んでいます。")
else:
    print(f"警告: 'sticker.png' は予期せぬチャンネル数 ({sticker_img.shape[2]}) です。")
    sys.exit(1)


# ステッカー画像の幅と高さを取得 (透過処理後の4チャンネル画像から取得)
sticker_h, sticker_w, sticker_c = sticker_img.shape

# ランドマークからピクセル座標に変換する関数
def normalized_to_pixel_coordinates(normalized_x, normalized_y, width, height):
    return int(normalized_x * width), int(normalized_y * height)

# アルファブレンドで画像を合成する関数（背景が透明なPNG用）
def overlay_transparent_image(image, sticker, x, y, w, h):
    # ステッカーを指定サイズにリサイズ
    sticker = cv2.resize(sticker, (w, h), interpolation=cv2.INTER_AREA)

    # 貼り付け先の範囲を定義
    y1, y2 = max(0, y), min(image.shape[0], y + h)
    x1, x2 = max(0, x), min(image.shape[1], x + w)

    # ステッカーの範囲を定義 (画像がフレームからはみ出す場合に対応)
    sticker_y1 = max(0, -y)
    sticker_y2 = sticker_y1 + (y2 - y1)
    sticker_x1 = max(0, -x)
    sticker_x2 = sticker_x1 + (x2 - x1)

    if y2 <= y1 or x2 <= x1 or sticker_y2 <= sticker_y1 or sticker_x2 <= sticker_x1:
        return

    # 領域を切り出し
    frame_roi = image[y1:y2, x1:x2]
    sticker_roi = sticker[sticker_y1:sticker_y2, sticker_x1:sticker_x2]

    # ステッカーのアルファチャンネルとBGRチャンネルを取得
    alpha = sticker_roi[:, :, 3] / 255.0
    color = sticker_roi[:, :, :3]

    # 背景（フレーム）と前景（ステッカー）をアルファブレンド
    for c in range(0, 3):
        frame_roi[:, :, c] = (alpha * color[:, :, c] +
                              (1.0 - alpha) * frame_roi[:, :, c])


# ----------------------------------------------------
# 1. Webカメラのキャプチャを開始
# ----------------------------------------------------
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"エラー: インデックス {CAMERA_INDEX} のカメラを開けませんでした。")
    sys.exit(1)

print(f"カメラ (インデックス {CAMERA_INDEX}) を開きました。リアルタイム顔フィルターを開始します。")
print("'q'キーまたはESCキーを押すと終了します。")

# FPS計算用変数
pTime = 0
SCALING_FACTOR = 2.0 # ★ イラストのサイズ調整（顔の幅の2倍に設定）

# ----------------------------------------------------
# 2. 映像処理のメインループ
# ----------------------------------------------------
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("カメラフレームの読み込みに失敗しました。")
        break

    # 左右反転
    image = cv2.flip(image, 1)

    # 3. 処理のための画像準備 (BGR -> RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # 処理高速化
    
    # 4. MediaPipeによる顔検出の実行
    results = face_mesh.process(image_rgb)

    # 5. 結果の描画と表示

    # メッシュ描画用の画像を用意 (2つ目のウィンドウ用)
    mesh_image = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    # イラスト合成のために元のフレームをBGR形式に戻す
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # ----------------------------------------------------
            # (A) イラスト合成に必要な座標計算
            # ----------------------------------------------------
            
            # 鼻の先端 (4) を中心点、目の外側 (33, 263) を顔の幅の基準とする
            nose_tip = face_landmarks.landmark[4]
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]

            # ピクセル座標に変換
            nose_tip_x, nose_tip_y = normalized_to_pixel_coordinates(nose_tip.x, nose_tip.y, img_w, img_h)
            left_x, _ = normalized_to_pixel_coordinates(left_eye_outer.x, left_eye_outer.y, img_w, img_h)
            right_x, _ = normalized_to_pixel_coordinates(right_eye_outer.x, right_eye_outer.y, img_w, img_h)

            # 顔の幅を推定
            face_width_pixel = abs(left_x - right_x)
            
            # イラストのサイズと位置を決定
            sticker_display_w = int(face_width_pixel * SCALING_FACTOR)
            sticker_display_h = int(sticker_display_w * sticker_h / sticker_w)
            
            # 貼り付け開始座標 (イラストの中心を鼻の先端に合わせる)
            sticker_x = nose_tip_x - sticker_display_w // 2
            sticker_y = nose_tip_y - sticker_display_h // 2
            
            # ----------------------------------------------------
            # (B) 1つ目のウィンドウ: イラストを合成
            # ----------------------------------------------------
            overlay_transparent_image(image, sticker_img, sticker_x, sticker_y, sticker_display_w, sticker_display_h)
            
            # ----------------------------------------------------
            # (C) 2つ目のウィンドウ: MediaPipe Meshを描画
            # ----------------------------------------------------
            mp_drawing.draw_landmarks(
                image=mesh_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION, 
                landmark_drawing_spec=None, 
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

    # FPS表示
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    fps_text = f'FPS: {int(fps)}'
    
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(mesh_image, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # 6. 映像を画面に表示 (2つのウィンドウ)
    cv2.imshow('1. Sticker Overlay (Press "q" or ESC to quit)', image)
    cv2.imshow('2. MediaPipe Face Mesh Debug', mesh_image)

    # 'q'キーまたは ESCキーが押されたらループを終了
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27:
        break

# ----------------------------------------------------
# 7. 後処理
# ----------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("プログラムを終了しました。")