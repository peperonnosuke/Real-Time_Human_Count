# ライブラリをインポート
from flask import render_template, Flask, Response
import cv2
from ultralytics import YOLO  

app = Flask(__name__,  static_folder="../static")

# 学習済みモデルをロード
model = YOLO("../yolov8n.pt")

# カメラを読み込む(引数はノートパソコンのWebカメラならば0、外付けのカメラならば1にする)
camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture(1)

def gen_frames():
   while True:
       success, frame = camera.read()
       if not success:
           break
       else:
           # フレームデータを推論
           results = model(frame, classes=0)    # 人だけを検知する
           annotated_frame = results[0].plot()  # バウンディングボックスを付与

           # 検知数を保存
           person_count = str(results[0].boxes.data.shape[0])

           # 画像の高さと幅
           height, width = annotated_frame.shape[:2]
           
           # 塗りつぶす領域の高さ
           top_height = int(height * 0.1)  # 上部10%を塗りつぶす
           
           # 塗りつぶす色の指定
           fill_color = (0, 0, 0) 
           
           # 上部領域を指定の色で塗る
           annotated_frame[0:top_height, :] = fill_color

           # 検出数を書き込む
           cv2.putText(annotated_frame, f'Human Count: {person_count}', (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (204, 153, 51), 3)
           
           # 推論データをjpgに圧縮 (画像をメモリ上に圧縮して保存する)
           ret, buffer = cv2.imencode('.jpg', annotated_frame)

           # bytesデータ化
           frame = buffer.tobytes()
           yield (
               b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
               )

@app.route('/video_feed')
def video_feed():
   # imgタグに埋め込まれるResponseオブジェクトを返す
   return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
@app.route('/index')
def index(): 
   user = {'username': 'User'}
   return render_template('index.html', title='home', user=user)

if __name__ == "__main__":
    app.debug = True
    app.run(host="localhost", port=5000)