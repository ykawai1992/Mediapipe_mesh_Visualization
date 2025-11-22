import cv2
import mediapipe as mp
import sys
import time

# MediaPipeとOpenCVのバージョン確認
print(f"Pythonバージョン: {sys.version}")
print(f"MediaPipeバージョン: {mp.__version__}")
print(f"OpenCVバージョン: {cv2.__version__}")
print("---------------------------------------")

# MediaPipe Poseの初期設定
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# model_complexity=1 (デフォルト) は精度と速度のバランスが良い設定
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 1. Webカメラのキャプチャを開始
# 動作確認済みのインデックス '0' と Windows推奨の CAP_DSHOW を使用
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"エラー: インデックス {CAMERA_INDEX} のカメラを開けませんでした。")
    print("カメラが接続されているか、他のアプリケーションで使用されていないか確認してください。")
    sys.exit(1)

print(f"カメラ (インデックス {CAMERA_INDEX}) を開きました。リアルタイム骨格検出を開始します。")
print("'q'キーまたはESCキーを押すと終了します。")

# FPS計算用変数
pTime = 0

# 2. 映像処理のメインループ
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("カメラフレームの読み込みに失敗しました。")
        break

    # 左右反転 (鏡のように表示するために推奨)
    image = cv2.flip(image, 1)

    # 3. 処理のための画像準備 (BGR -> RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # 処理高速化
    
    # 4. MediaPipeによる姿勢推定の実行
    results = pose.process(image_rgb)

    # 5. 結果の描画 (RGB -> BGR)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # ランドマークとその接続を描画
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # ランドマークの描画設定
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
            # 接続線の描画設定
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  
        )

    # FPS（フレームレート）の表示 (オプション)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # 6. 映像を画面に表示
    cv2.imshow('MediaPipe Pose Estimation (Press "q" or ESC to quit)', image)

    # 'q'キーまたは ESCキーが押されたらループを終了 (5ms待機)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27: # 27はESCキーのASCIIコード
        break

# 7. 後処理
cap.release()
cv2.destroyAllWindows()
print("プログラムを終了しました。")