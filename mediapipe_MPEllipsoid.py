import cv2
import mediapipe as mp
import numpy as np # 新たに追加
import sys
import time
import math

# MediaPipe Poseの初期設定
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------------------------------
# 関数 1: 3点座標から関節角度を計算（度数）
# p1: 基点(肩), p2: 関節点(肘), p3: 末端(手首)
# -----------------------------------------------------------
def calculate_angle(p1, p2, p3):
    # NumPy配列に変換
    p1 = np.array([p1.x, p1.y, p1.z])
    p2 = np.array([p2.x, p2.y, p2.z])
    p3 = np.array([p3.x, p3.y, p3.z])
    
    # リンクベクトルを計算
    vector_a = p1 - p2 
    vector_b = p3 - p2
    
    # 内積とノルムから角度（ラジアン）を計算
    cosine_angle = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    
    # 浮動小数点誤差によるクリッピング（-1.0から1.0の範囲に収める）
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# -----------------------------------------------------------
# 関数 2: 2リンクアームのキネマティクス計算と可操作性楕円のパラメータ導出
# -----------------------------------------------------------
def calculate_ellipsoid(shoulder, elbow, wrist, frame_width, frame_height):
    # MediaPipeの正規化座標をピクセル座標に変換
    S = np.array([shoulder.x * frame_width, shoulder.y * frame_height]).astype(int)
    E = np.array([elbow.x * frame_width, elbow.y * frame_height]).astype(int)
    W = np.array([wrist.x * frame_width, wrist.y * frame_height]).astype(int)

    # リンクベクトル (2D)
    V1 = E - S  # 肩 -> 肘
    V2 = W - E  # 肘 -> 手首
    
    # リンク長 L1, L2
    L1 = np.linalg.norm(V1)
    L2 = np.linalg.norm(V2)

    # 2Dキネマティクスのための角度計算
    # リンク1の絶対角度 (X軸基準)
    theta1 = math.atan2(V1[1], V1[0]) 
    # リンク2の相対角度 (リンク1基準)
    theta2 = math.atan2(V2[1], V2[0]) - theta1 
    
    # ロボティクス的な関節角の表現 (純粋な回転角)
    # これらはアームの姿勢を厳密に表すために使用されます
    
    # --- 1. ヤコビアン行列 J の構築 (2リンク平面アームの公式) ---
    # J = [[ dX/d(theta1), dX/d(theta2) ],
    #      [ dY/d(theta1), dY/d(theta2) ]]
    
    # 微分項の共通部分
    s1 = math.sin(theta1)
    c1 = math.cos(theta1)
    s12 = math.sin(theta1 + theta2)
    c12 = math.cos(theta1 + theta2)

    J = np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],  # dX/d(theta1), dX/d(theta2)
        [ L1 * c1 + L2 * c12,  L2 * c12]   # dY/d(theta1), dY/d(theta2)
    ])
    
    # --- 2. 可操作性行列 W の計算 ---
    # W = J * J.T
    W = J @ J.T

    # --- 3. W の固有値と固有ベクトルから楕円パラメータを取得 ---
    # 固有値(lambda)が楕円の軸の長さの二乗、固有ベクトル(eigenvectors)が軸の方向
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    
    # 軸の長さ (特異値の平方根)
    a = math.sqrt(eigenvalues[1]) * 1.5 # 描画のためにスケールアップ
    b = math.sqrt(eigenvalues[0]) * 1.5 # 描画のためにスケールアップ

    # 回転角: 固有ベクトル e1 が水平軸と成す角度 (度数)
    # 楕円の長軸は最大固有値に対応する固有ベクトルが示す
    rotation_vector = eigenvectors[:, 1]
    angle_rad = math.atan2(rotation_vector[1], rotation_vector[0])
    angle_deg = np.degrees(angle_rad)

    return W[0, 0], W[1, 1], (a, b), angle_deg, W

# -----------------------------------------------------------
# メイン処理
# -----------------------------------------------------------
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"エラー: インデックス {CAMERA_INDEX} のカメラを開けませんでした。")
    sys.exit(1)

print(f"カメラ (インデックス {CAMERA_INDEX}) を開きました。リアルタイム運動学解析を開始します。")
print("'q'キーまたはESCキーを押すと終了します。")

# FPS計算用変数
pTime = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("カメラフレームの読み込みに失敗しました。")
        break

    image_h, image_w, _ = image.shape
    image = cv2.flip(image, 1) # 左右反転
    
    # BGR -> RGB 変換と姿勢推定
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 描画と計算
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 右腕のランドマークを抽出
        R_SHOULDER = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        R_ELBOW = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        R_WRIST = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # ランドマークの検出信頼度が低い場合はスキップ
        if R_WRIST.visibility > 0.6 and R_ELBOW.visibility > 0.6 and R_SHOULDER.visibility > 0.6:
            # ------------------------------------
            # 1. 関節角度の計算
            # ------------------------------------
            elbow_angle = calculate_angle(R_SHOULDER, R_ELBOW, R_WRIST)
            
            # ------------------------------------
            # 2. リンクベクトルとその描画
            # ------------------------------------
            # ピクセル座標に変換
            S_px = (int(R_SHOULDER.x * image_w), int(R_SHOULDER.y * image_h))
            E_px = (int(R_ELBOW.x * image_w), int(R_ELBOW.y * image_h))
            W_px = (int(R_WRIST.x * image_w), int(R_WRIST.y * image_h))

            # リンクの描画（元のMediaPipe描画に上書き）
            cv2.line(image, S_px, E_px, (255, 255, 0), 3) # リンク1: 黄色
            cv2.line(image, E_px, W_px, (0, 255, 255), 3) # リンク2: シアン

            # ------------------------------------
            # 3. 可操作性楕円の計算と描画
            # ------------------------------------
            try:
                # キネマティクス計算
                Wxx, Wyy, (a, b), angle_deg, W = calculate_ellipsoid(R_SHOULDER, R_ELBOW, R_WRIST, image_w, image_h)
                
                # 楕円の描画 (中心: 手首 W_px, 軸: a, b, 角度: angle_deg)
                # 楕円は末端（手首）を中心に描画されます
                cv2.ellipse(
                    img=image,
                    center=W_px,
                    axes=(int(a), int(b)),
                    angle=angle_deg,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 0, 255), # 赤色
                    thickness=2
                )
                
                # 可操作性のテキスト表示 (画像右下)
                cv2.putText(image, f'W_det: {np.linalg.det(W):.2e}', (image_w - 200, image_h - 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                
            except Exception as e:
                # 計算中にエラーが発生した場合（例: リンク長がゼロに近い場合など）
                cv2.putText(image, "Ellipsoid Calc Error", (image_w // 2 - 100, image_h // 2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


            # 関節角度とリンクベクトルのテキスト表示 (画像左上)
            cv2.putText(image, f'Angle (Elbow): {int(elbow_angle)} deg', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ベクトル情報の簡易表示 (正規化座標)
            V1_norm = np.array([R_ELBOW.x - R_SHOULDER.x, R_ELBOW.y - R_SHOULDER.y])
            V2_norm = np.array([R_WRIST.x - R_ELBOW.x, R_WRIST.y - R_ELBOW.y])
            cv2.putText(image, f'V1 (Shoulder->Elbow) Length: {np.linalg.norm(V1_norm):.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(image, f'V2 (Elbow->Wrist) Length: {np.linalg.norm(V2_norm):.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 描画の残り（ランドマークの円など）
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  
        )

    # FPS表示
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # 映像表示
    cv2.imshow('Kinematics and Manipulability (Press "q" or ESC to quit)', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27:
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
print("プログラムを終了しました。")