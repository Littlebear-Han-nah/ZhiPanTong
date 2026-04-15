import sqlite3
import qrcode
import os


def generate_qr_from_db():
    # 1. 连接数据库
    db_path = 'asset_inventory.db'
    if not os.path.exists(db_path):
        print(f"错误：找不到数据库文件 {db_path}，请先运行一次 app.py 初始化数据库！")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. 查询所有资产的关键信息
    # 我们只需要 RFID TAG 写入二维码，其他信息生成在文件名上方便你区分
    cursor.execute('SELECT name, rfid_tag, type, status FROM assets')
    assets = cursor.fetchall()
    conn.close()

    # 3. 创建保存文件夹
    save_dir = "asset_qr_codes"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在从数据库读取 {len(assets)} 个资产...\n")

    for asset in assets:
        name, tag, asset_type, status = asset

        # --- 核心逻辑 ---
        # 二维码内容只写入 'tag' (例如 RFID001)
        # 这样 app.py 里的 get_asset_info_by_tag(rfid_tag) 才能正确查到数据
        qr_content = tag

        # 生成二维码
        qr = qrcode.QRCode(box_size=10, border=4)
        qr.add_data(qr_content)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        # 文件名包含详细信息，方便你演示时分辨
        # 格式：[RFID001] 生产设备A (正常).png
        filename = f"[{tag}] {name} ({status}).png"
        # 清理文件名非法字符
        filename = filename.replace("/", "_").replace("\\", "_")

        save_path = os.path.join(save_dir, filename)
        img.save(save_path)

        print(f"✅ 已生成: {filename}")

    print(f"\n所有二维码已保存在 '{save_dir}' 文件夹中。")
    print("请使用手机拍照或直接在屏幕上打开这些图片进行 AR 识别测试。")


if __name__ == "__main__":
    generate_qr_from_db()