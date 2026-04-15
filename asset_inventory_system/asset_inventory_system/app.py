import os

# 解决某些环境下的OpenMP重复加载错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from flask import Flask, render_template, request, jsonify, Response
import sqlite3
import random
from datetime import datetime
import json
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import threading
from threading import Lock
import time
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# ==========================================
# 1. 初始化 AI 模型与全局变量
# ==========================================

# --- 关键修复：定义全局变量 ---
outputFrame = None
lock = Lock()
current_mode = 'qr'
yolo_model = None

# 加载模型
try:
    # 这里的路径请根据你的实际情况修改
    model_path = r'E:\Desktop\asset_inventory_system\asset_inventory_system\yolov8n.pt'
    yolo_model = YOLO(model_path)
    print(f"YOLO 模型加载成功: {model_path}")
except Exception as e:
    print(f"YOLO 模型加载失败: {e}")
    yolo_model = None


# ==========================================
# 2. 后台独立线程：负责读取摄像头 + AI计算
# ==========================================
# ==========================================
# 2. 后台独立线程：负责读取摄像头 + AI计算
# ==========================================
# ==========================================
# 辅助函数：解决 OpenCV 不支持中文的问题
# ==========================================
def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)

    # 字体的格式，Windows系统通常内置了黑体 (simhei.ttf)
    # 如果报错 "OSError: cannot open resource"，请检查 C:/Windows/Fonts/ 下是否有 simhei.ttf
    try:
        font_style = ImageFont.truetype("simhei.ttf", text_size, encoding="utf-8")
    except:
        # 如果找不到黑体，尝试使用默认字体（可能还是不支持中文，但至少不会崩）
        font_style = ImageFont.load_default()

    # 绘制文本
    draw.text(position, text, fill=text_color, font=font_style)

    # 转换回 OpenCV 格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ==========================================
# 2. 后台独立线程：负责读取摄像头 + AI计算
# ==========================================
def process_video_feed():
    global outputFrame, lock, current_mode, yolo_model

    # 打开摄像头
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    time.sleep(2.0)

    while True:
        success, frame = camera.read()
        if not success:
            camera.release()
            time.sleep(1)
            camera = cv2.VideoCapture(0)
            continue

        frame = cv2.resize(frame, (640, 480))

        try:
            # ---------------------------------------------------------
            # 模式 A: AR 信息增强模式 (二维码)
            # ---------------------------------------------------------
            if current_mode == 'qr':
                # 界面提示 (英文可以直接用 cv2.putText)
                cv2.putText(frame, "Mode: AR Asset Info", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                decoded_objects = decode(frame)
                for obj in decoded_objects:
                    rfid_data = obj.data.decode('utf-8')

                    # 绘制绿色的定位框
                    pts = np.array([obj.polygon], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

                    # 查询数据库
                    asset_info = get_asset_info_by_tag(rfid_data)

                    rect = obj.rect
                    x, y = rect.left, rect.top

                    if asset_info:
                        name, status, rate = asset_info

                        # 设定颜色 (OpenCV使用BGR格式: 蓝,绿,红)
                        # 正常=绿色(0, 255, 0)，其他=红色(0, 0, 255)
                        text_color = (0, 255, 0) if status == '正常' else (255, 0, 0)

                        # 绘制背景黑块 (让文字更清晰)
                        # 为了容纳中文，稍微把框画高一点
                        cv2.rectangle(frame, (x, y - 90), (x + 250, y - 5), (0, 0, 0), -1)

                        # --- 关键修改：使用自定义函数绘制中文 ---
                        # 注意：PIL使用RGB颜色，所以(255, 0, 0)是红，(0, 255, 0)是绿

                        # 第一行：资产名称 (白色)
                        frame = cv2_add_chinese_text(frame, f"资产: {name}", (x + 5, y - 85), (255, 255, 255), 20)

                        # 第二行：状态 (根据状态变色)
                        # PIL颜色是RGB，所以红色是(255, 0, 0)，绿色是(0, 255, 0)
                        pil_color = (0, 255, 0) if status == '正常' else (255, 0, 0)
                        frame = cv2_add_chinese_text(frame, f"状态: {status}", (x + 5, y - 55), pil_color, 20)

                        # 第三行：利用率 (灰色)
                        frame = cv2_add_chinese_text(frame, f"利用率: {rate:.0%}", (x + 5, y - 25), (200, 200, 200), 18)

                    else:
                        cv2.putText(frame, f"Unknown: {rfid_data}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 255), 2)

            # ---------------------------------------------------------
            # 模式 B: YOLO 物体检测
            # ---------------------------------------------------------
            elif current_mode == 'yolo' and yolo_model:
                cv2.putText(frame, "Mode: YOLO Object Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 165, 255), 2)
                results = yolo_model(frame, verbose=False, conf=0.5, imgsz=320)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = f"{yolo_model.names[cls]} {float(box.conf[0]):.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        except Exception as e:
            print(f"AI Error: {e}")
            pass

        with lock:
            outputFrame = frame.copy()


# ==========================================
# 3. Web 视频流生成器：只负责取图发送
# ==========================================
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            # 编码图片
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        # 发送字节流
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

        # 控制发送频率，防止网页卡死
        time.sleep(0.03)


# ==========================================
# 4. 数据库工具函数
# ==========================================
def get_db_connection():
    conn = sqlite3.connect('asset_inventory.db')
    conn.row_factory = None
    return conn


def get_asset_info_by_tag(rfid_tag):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT name, status, utilization_rate FROM assets WHERE rfid_tag = ?', (rfid_tag,))
    result = c.fetchone()
    conn.close()
    return result


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS assets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  type TEXT NOT NULL,
                  location TEXT NOT NULL,
                  status TEXT NOT NULL,
                  rfid_tag TEXT UNIQUE NOT NULL,
                  purchase_date DATE,
                  last_inventory DATE,
                  utilization_rate REAL,
                  model_url TEXT DEFAULT 'https://via.placeholder.com/150')''')

    c.execute('''CREATE TABLE IF NOT EXISTS inventory_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  asset_id INTEGER,
                  inventory_time DATETIME,
                  operator TEXT,
                  result TEXT,
                  accuracy REAL,
                  FOREIGN KEY(asset_id) REFERENCES assets(id))''')

    c.execute("SELECT COUNT(*) FROM assets")
    if c.fetchone()[0] == 0:
        test_assets = [
            ("生产设备A", "机械设备", "车间1号区域", "正常", "RFID001", "2023-01-15", "2024-10-01", 0.85,
             "https://via.placeholder.com/150/00FF00"),
            ("电脑主机B", "电子设备", "办公室3楼", "正常", "RFID002", "2023-03-20", "2024-10-01", 0.92,
             "https://via.placeholder.com/150/0000FF"),
            ("货架C", "仓储设备", "仓库2区", "闲置", "RFID003", "2022-11-05", "2024-09-15", 0.23,
             "https://via.placeholder.com/150/FFFF00"),
            ("打印机D", "电子设备", "行政办公室", "维修中", "RFID004", "2023-05-10", "2024-09-20", 0.45,
             "https://via.placeholder.com/150/FF0000"),
            ("叉车E", "运输设备", "仓库1区", "正常", "RFID005", "2022-08-25", "2024-10-02", 0.78,
             "https://via.placeholder.com/150/800080")
        ]
        c.executemany(
            "INSERT INTO assets (name, type, location, status, rfid_tag, purchase_date, last_inventory, utilization_rate, model_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            test_assets)
    conn.commit()
    conn.close()


# ==========================================
# 5. Web 路由
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')


# --- 关键路由：视频流 ---
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/api/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    global current_mode
    if mode in ['qr', 'yolo']:
        current_mode = mode
        return jsonify({"status": "success", "current_mode": current_mode})
    return jsonify({"status": "error", "msg": "Invalid mode"})


@app.route('/assets')
def asset_list():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM assets')
    assets = c.fetchall()
    conn.close()
    assets_dict = []
    for asset in assets:
        assets_dict.append({
            "id": asset[0], "name": asset[1], "type": asset[2], "location": asset[3],
            "status": asset[4], "rfid_tag": asset[5], "purchase_date": asset[6],
            "last_inventory": asset[7], "utilization_rate": asset[8], "model_url": asset[9]
        })
    return render_template('assets.html', assets=assets_dict)


@app.route('/assets/<int:asset_id>')
def asset_detail(asset_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM assets WHERE id = ?', (asset_id,))
    asset = c.fetchone()
    if not asset:
        conn.close()
        return "资产不存在", 404
    c.execute('SELECT * FROM inventory_records WHERE asset_id = ? ORDER BY inventory_time DESC LIMIT 5', (asset_id,))
    records = c.fetchall()
    conn.close()

    asset_dict = {
        "id": asset[0], "name": asset[1], "type": asset[2], "location": asset[3],
        "status": asset[4], "rfid_tag": asset[5], "purchase_date": asset[6],
        "last_inventory": asset[7], "utilization_rate": asset[8], "model_url": asset[9]
    }
    records_dict = []
    for record in records:
        records_dict.append({
            "id": record[0], "asset_id": record[1], "inventory_time": record[2],
            "operator": record[3], "result": record[4], "accuracy": record[5]
        })
    return render_template('asset_detail.html', asset=asset_dict, records=records_dict)


@app.route('/inventory/simulate', methods=['POST'])
def simulate_inventory():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id, rfid_tag FROM assets')
    assets = c.fetchall()
    results = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    accuracy = 0.995
    for asset in assets:
        asset_id = asset[0]
        rfid = asset[1]
        if random.random() < accuracy:
            result = "匹配成功"
            c.execute("UPDATE assets SET last_inventory = ? WHERE id = ?",
                      (datetime.now().strftime("%Y-%m-%d"), asset_id))
        else:
            result = "数据不匹配"
        c.execute(
            "INSERT INTO inventory_records (asset_id, inventory_time, operator, result, accuracy) VALUES (?, ?, ?, ?, ?)",
            (asset_id, now, "系统自动盘点", result, accuracy))
        results.append({"asset_id": asset_id, "rfid": rfid, "result": result, "accuracy": accuracy})
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "time": now, "total_assets": len(assets),
                    "success_count": len([r for r in results if r["result"] == "匹配成功"]), "results": results})


@app.route('/inventory/records')
def inventory_records():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''SELECT ir.id, a.name, a.rfid_tag, ir.inventory_time, ir.operator, ir.result, ir.accuracy
        FROM inventory_records ir JOIN assets a ON ir.asset_id = a.id ORDER BY ir.inventory_time DESC''')
    records = c.fetchall()
    conn.close()
    records_dict = []
    for record in records:
        records_dict.append({
            "id": record[0], "asset_name": record[1], "rfid_tag": record[2],
            "inventory_time": record[3], "operator": record[4], "result": record[5], "accuracy": record[6]
        })
    return render_template('inventory_records.html', records=records_dict)


@app.route('/assets/optimization')
def asset_optimization():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, name, type, location, status, utilization_rate FROM assets WHERE utilization_rate < 0.3 OR status IN ('维修中', '闲置')")
    low_util_assets = c.fetchall()
    conn.close()
    suggestions = []
    for asset in low_util_assets:
        asset_id, asset_name, asset_type, current_location, status, rate = asset

        def get_target_location(type):
            location_map = {"机械设备": "生产车间", "电子设备": "研发办公室", "仓储设备": "物流仓库",
                            "运输设备": "货运码头"}
            return location_map.get(type, "核心业务区域")

        if rate < 0.3:
            suggestion = f"利用率仅{rate:.0%}，建议转移至{get_target_location(asset_type)}或启动报废流程"
        elif status == "闲置":
            suggestion = "当前处于闲置状态，建议调配至需求部门提高利用率"
        else:
            suggestion = "设备维修中，建议加快维修进度或评估替换方案"
        suggestions.append({
            "asset_id": asset_id, "asset_name": asset_name, "current_location": current_location,
            "utilization_rate": f"{rate:.0%}", "status": status, "suggestion": suggestion
        })
    return render_template('optimization.html', suggestions=suggestions)


@app.route('/data/visualization')
def data_visualization():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT status, COUNT(*) FROM assets GROUP BY status')
    status_data = dict(c.fetchall())
    c.execute('SELECT type, COUNT(*) FROM assets GROUP BY type')
    type_data = dict(c.fetchall())
    c.execute('SELECT COUNT(*) FROM assets WHERE utilization_rate > 0.7')
    high_util = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM assets WHERE utilization_rate BETWEEN 0.3 AND 0.7')
    mid_util = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM assets WHERE utilization_rate < 0.3')
    low_util = c.fetchone()[0]
    util_data = {"高利用率(>70%)": high_util, "中利用率(30%-70%)": mid_util, "低利用率(<30%)": low_util}
    c.execute(
        'SELECT DATE(inventory_time), AVG(accuracy) FROM inventory_records GROUP BY DATE(inventory_time) ORDER BY DATE(inventory_time) DESC LIMIT 10')
    accuracy_trend = c.fetchall()
    conn.close()
    trend_labels = [row[0] for row in accuracy_trend][::-1] if accuracy_trend else []
    trend_values = [round(row[1] * 100, 2) for row in accuracy_trend][::-1] if accuracy_trend else []
    return render_template('visualization.html', status_data=status_data, type_data=type_data, util_data=util_data,
                           trend_labels=json.dumps(trend_labels), trend_values=json.dumps(trend_values))


@app.route('/ar/recognize', methods=['POST'])
def ar_recognize():
    data = request.json
    rfid_tag = data.get('rfid')
    if not rfid_tag: return jsonify({"status": "fail", "msg": "请提供RFID标签"})
    asset = get_asset_info_by_tag(rfid_tag)
    if asset:
        return jsonify({"status": "success", "msg": "建议使用视频流模式查看详情"})
    else:
        return jsonify({"status": "fail", "msg": "未识别到对应资产"})


@app.route('/ar/scan')
def ar_scan():
    return render_template('ar_scan.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


# ==========================================
# 6. 启动
# ==========================================

if __name__ == '__main__':
    init_db()

    # 启动后台线程处理摄像头
    t = threading.Thread(target=process_video_feed)
    t.daemon = True  # 设置为守护线程
    t.start()

    # 启动 Flask
    app.run(debug=False, host='0.0.0.0', port=8000, threaded=True)