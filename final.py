from flask import Flask, request, jsonify, send_file
import os
import cv2
import json
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # 로컬에서 이미지 업로드
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400
    
    file = request.files['image']
    file.save('uploaded_image.jpg')
    
    img = cv2.imread('uploaded_image.jpg')
    results = model.predict(img)

    # 객체 탐지 결과 이미지 저장
    for pred in results:
        boxes = pred.boxes  # Detection boxes
        for box in boxes:
            # Bounding box 좌표 얻기
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # numpy 배열로 변환
            # Bounding box 그리기
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    output_path = 'output_with_boxes.jpg'
    cv2.imwrite(output_path, img)

    # 결과 이미지 반환
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
