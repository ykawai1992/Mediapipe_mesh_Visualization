import cv2
import mediapipe as mp
import sys
import time
import numpy as np

# MediaPipeとOpenCVのバージョン確認 (省略)
# ...

# 0. 貼り付けるイラスト画像を読み込む
# 透明度(アルファチャネル)付きのPNG画像を読み込む (cv2.IMREAD_UNCHANGED)
try:
    sticker_img = cv2.imread('sticker.png', cv2.IMREAD_UNCHANGED)
    if sticker_img is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("エラー: 'sticker.png' が見つかりません。")
    print("顔に貼り付けるための透明な背景を持つPNG画像を準備し、'sticker.png'として同じディレクトリに保存してください。")
    sys.exit(1)

# ★★★★★ 修正点: 白背景の透明化処理を追加 ★★★★★
if sticker_img.shape[2] == 3: # 3チャンネル画像（BGR）の場合、透明度がないので白背景を透過させる
    print("ステッカー画像にアルファチャンネルがないため、白背景を透過処理します。")
    
    # BGR画像をHSVに変換
    hsv = cv2.cvtColor(sticker_img, cv2.COLOR_BGR2HSV)

    # 白色の範囲を定義 (HSV)
    # H: 色相 (Hue), S: 彩度 (Saturation), V: 明度 (Value)
    # 白は彩度が低く (0-30くらい)、明度が高い (200-255くらい)
    # 一般的な白の範囲
    lower_white = np.array([0, 0, 200])   # 彩度が非常に低く、明度が非常に高い
    upper_white = np.array([180, 30, 255]) # 彩度が少しあり、明度が最高の状態

    # 白色のマスクを作成
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # マスクを反転 (白以外の部分が前景、白が背景)
    mask_inv = cv2.bitwise_not(mask)
    
    # アルファチャンネルとして利用するために255に変換
    alpha_channel = mask_inv

    # 元のBGR画像にアルファチャンネルを追加して4チャンネル画像にする
    sticker_img = cv2.merge([sticker_img[:,:,0], sticker_img[:,:,1], sticker_img[:,:,2], alpha_channel])

    print(f"白背景透過処理後のチャンネル数: {sticker_img.shape[2]}")

elif sticker_img.shape[2] == 4: # 4チャンネル画像（BGRA）の場合
    print("ステッカー画像は既にアルファチャンネルを含んでいます。")
else:
    print(f"警告: 'sticker.png' は予期せぬチャンネル数です: {sticker_img.shape[2]}")
    sys.exit(1)
# ★★★★★ 修正点 終わり ★★★★★

# ステッカー画像の幅と高さを取得
sticker_h, sticker_w, sticker_c = sticker_img.shape


# MediaPipe Face Meshの初期設定に変更
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# max_num_faces=1 で処理を軽くし、検出数上限を1に設定
# refine_landmarks=True は精度向上に役立ちます
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ランドマークからピクセル座標に変換する関数
def normalized_to_pixel_coordinates(normalized_x, normalized_y, width, height):
    return int(normalized_x * width), int(normalized_y * height)

# アルファブレンドで画像を合成する関数（背景が透明なPNG用）
# image: 合成先のフレーム
# sticker: 合成するステッカー画像（4チャンネル: BGRA）
# x, y: 貼り付けを開始する左上の座標
# w, h: 貼り付け後のステッカーの幅と高さ
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

    # 貼り付け先の領域とステッカーの領域が有効かチェック
    if y2 <= y1 or x2 <= x1 or sticker_y2 <= sticker_y1 or sticker_x2 <= sticker_x1:
        return

    # 領域を切り出し
    frame_roi = image[y1:y2, x1:x2]
    sticker_roi = sticker[sticker_y1:sticker_y2, sticker_x1:sticker_x2]

    # ステッカーのアルファチャンネル（透明度）とBGRチャンネルを取得
    alpha = sticker_roi[:, :, 3] / 255.0
    color = sticker_roi[:, :, :3]

    # 背景（フレーム）と前景（ステッカー）をブレンド
    for c in range(0, 3):
        frame_roi[:, :, c] = (alpha * color[:, :, c] +
                              (1.0 - alpha) * frame_roi[:, :, c])


# 1. Webカメラのキャプチャを開始 (既存コードと同じ)
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"エラー: インデックス {CAMERA_INDEX} のカメラを開けませんでした。")
    print("カメラが接続されているか、他のアプリケーションで使用されていないか確認してください。")
    sys.exit(1)

print(f"カメラ (インデックス {CAMERA_INDEX}) を開きました。リアルタイム顔フィルターを開始します。")
print("'q'キーまたはESCキーを押すと終了します。")

# FPS計算用変数
pTime = 0

# 2. 映像処理のメインループ
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

    # 5. 結果の描画（イラストの貼り付け）
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 顔のランドマークから、目の外側（ランドマーク 33と263）と口の中心（ランドマーク 13）の座標を取得する

            # 鼻のランドマーク（例：鼻の先端 4, 額の中心 10）や目のランドマーク（33, 263）などを利用し、顔の中心位置を決定します。
            # 例: 左目外側 (33) と右目外側 (263) を基に顔の幅を推定
            # MediaPipe Face Meshのインデックスは以下の通り：
            # 左目外側: 33, 右目外側: 263
            # 鼻の先端: 4
            # 顎の中心: 152
            
            # 鼻の先端の座標（イラストの中心点として使用）
            nose_tip = face_landmarks.landmark[4]
            # 左目外側の座標
            left_eye_outer = face_landmarks.landmark[33]
            # 右目外側の座標
            right_eye_outer = face_landmarks.landmark[263]

            # ピクセル座標に変換
            nose_tip_x, nose_tip_y = normalized_to_pixel_coordinates(nose_tip.x, nose_tip.y, img_w, img_h)
            left_x, left_y = normalized_to_pixel_coordinates(left_eye_outer.x, left_eye_outer.y, img_w, img_h)
            right_x, right_y = normalized_to_pixel_coordinates(right_eye_outer.x, right_eye_outer.y, img_w, img_h)

            # 顔の幅を推定 (左右の目の外側の距離)
            face_width_pixel = np.linalg.norm(np.array([left_x, left_y]) - np.array([right_x, right_y]))

            # イラストの貼り付けサイズを顔の幅から決定（例：顔幅の1.5倍の幅にする）
            # ★★★ 変更点: スケーリングファクターを 2.0 に変更 ★★★
            SCALING_FACTOR = 2.0
            
            # イラストの貼り付けサイズを顔の幅から決定
            sticker_display_w = int(face_width_pixel * SCALING_FACTOR)
            # 縦横比を維持して高さを計算
            sticker_display_h = int(sticker_display_w * sticker_h / sticker_w)

            # 貼り付け開始座標 (イラストの中心を鼻の先端に合わせる)
            sticker_x = nose_tip_x - sticker_display_w // 2
            sticker_y = nose_tip_y - sticker_display_h // 2  # 鼻の先端から少し上に調整しても良い

            # イラストを合成
            overlay_transparent_image(image, sticker_img, sticker_x, sticker_y, sticker_display_w, sticker_display_h)


    # FPS（フレームレート）の表示 (既存コードと同じ)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # 6. 映像を画面に表示
    cv2.imshow('MediaPipe Face Sticker (Press "q" or ESC to quit)', image)

    # 'q'キーまたは ESCキーが押されたらループを終了 (5ms待機)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27: # 27はESCキーのASCIIコード
        break

# 7. 後処理 (既存コードと同じ)
cap.release()
cv2.destroyAllWindows()
print("プログラムを終了しました。")