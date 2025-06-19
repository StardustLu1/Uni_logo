# -*- coding: utf-8 -*-

import cv2
import json
import numpy as np
import typer
import supervision as sv
from PIL import Image
from io import BytesIO
import requests
from bs4 import BeautifulSoup
from zidian_uni import LABEL_TO_SCHOOL_NAME
import web_service
from ultralytics import YOLO
from tencentcloud.common import credential
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


app = typer.Typer()

def safe_print(s):
    import sys
    try:
        print(s.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
    except Exception:
        print(s)

# 初始化 YOLO 模型
# yolo_model = YOLO("./runs/detect/school_logo_yolov125/weights/best.pt")
yolo_model = YOLO("./runs/train/school_logo_yolov124/weights/best.pt")

def ask_hunyuan(school_abbr: str):
    school_name = LABEL_TO_SCHOOL_NAME.get(school_abbr.lower(), school_abbr)

    prompt_text = f"""
        你是一个大学信息查询助手。请根据提供的大学名称“{school_name}”返回如下结构化信息：
        - 学校名（中文）
        - 所属国家
        - QS 世界大学排名（若无请写“无”）
        - 软科大学排名
        - 官网链接
        
        请严格按如下格式输出：
        学校名：...
        国家：...
        QS排名：...
        软科排名：...
        官网链接：...
        
        此外，还请给出这个学校的相关百科、今年来的获得奖项、杰出校友，
        并请你查询后给出该大学近年高考浙江省投档线分数；
        
        最后，请配上几句你对这个大学的评价。
        """

    cred = credential.Credential("sdkid", "sdkmima")
    client = hunyuan_client.HunyuanClient(cred, "ap-guangzhou")

    req = models.ChatCompletionsRequest()
    params = {
        "Messages": [{"Role": "user", "Content": prompt_text}],
        "Model": "hunyuan-pro",
        "Temperature": 0.7
    }
    req.from_json_string(json.dumps(params))

    resp = client.ChatCompletions(req)
    return resp.Choices[0].Message.Content


@app.command()
def detect_camera():
    typer.echo("启动摄像头，检测大学校徽...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    target_width, target_height = 840, 440
    frame_rate = 10

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_count = 0
    detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            typer.echo("无法读取摄像头帧")
            break

        resized_frame = cv2.resize(frame, (target_width, target_height))
        results = yolo_model(resized_frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        annotated_frame = bounding_box_annotator.annotate(scene=resized_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imshow("摄像头校徽识别", annotated_frame)

        if frame_count % frame_rate == 0 and len(detections) > 0 and not detected:
            for i in range(len(detections)):
                class_id = int(detections.class_id[i])
                school_label = yolo_model.model.names[class_id]
                typer.echo(f"检测到校徽：{school_label}，准备识别...")

                try:
                    result = ask_hunyuan(school_label)
                    typer.echo("识别结果：")
                    typer.echo(result)
                except Exception as e:
                    typer.echo(f"识别失败：{str(e)}")

            detected = True

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            typer.echo("用户请求退出摄像头识别。")
            break

        if detected:
            typer.echo("识别完成，关闭摄像头。")
            break

    cap.release()
    cv2.destroyAllWindows()


@app.command()
def detect_image(path: str):
    typer.echo(f"识别图像文件: {path}")
    image = cv2.imread(path)
    results = yolo_model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
    annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

    cv2.imshow("图像识别结果", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i in range(len(detections)):
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        cropped = image[y1:y2, x1:x2]
        temp_path = f"image_logo_{i}.jpg"
        cv2.imwrite(temp_path, cropped)
        class_id = int(detections.class_id[i])
        school_label = yolo_model.model.names[class_id]
        try:
            result = ask_hunyuan(school_label)
            typer.echo("识别结果：")
            typer.echo(result)
        except Exception as e:
            typer.echo(f"识别失败：{str(e)}")

@app.command()
def detect_video(video_path: str):
    typer.echo(f"打开视频并实时检测：{video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_rate = 10  # 每10帧检测一次
    frame_count = 0
    seen_schools = set()
    last_detections = None  # 保存上一轮的检测结果

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        display_frame = frame.copy()

        if frame_count % frame_rate == 0:
            results = yolo_model(frame)[0]
            last_detections = sv.Detections.from_ultralytics(results)

            for i in range(len(last_detections)):
                class_id = int(last_detections.class_id[i])
                school_label = yolo_model.model.names[class_id]
                seen_schools.add(school_label)

        # 始终使用最近一次的检测框来标注
        if last_detections is not None:
            bbox_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            display_frame = bbox_annotator.annotate(scene=display_frame, detections=last_detections)
            display_frame = label_annotator.annotate(scene=display_frame, detections=last_detections)

        # 缩放窗口大小（宽度缩放为800）
        display_frame = cv2.resize(display_frame, (1200, int(display_frame.shape[0] * 1200 / display_frame.shape[1])))

        cv2.imshow("Logo Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    typer.echo("\n识别到的学校Logo有：")
    for school in seen_schools:
        typer.echo(f"- {school}")

@app.command()
def detect_url_images(url: str):
    """
    从网页中提取所有图片进行大学Logo识别和混元信息查询。
    """
    typer.echo(f"开始提取网页图片：{url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "lxml")
        img_tags = soup.find_all("img")
        img_urls = []

        for img in img_tags:
            src = img.get("src")
            if not src:
                continue
            # 补全相对路径为绝对路径
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                base_url = "/".join(url.split("/")[:3])
                src = base_url + src
            elif not src.startswith("http"):
                base_url = "/".join(url.split("/")[:3])
                src = base_url + "/" + src
            img_urls.append(src)

        typer.echo(f"共提取到 {len(img_urls)} 张图片，开始识别...")

        for idx, img_url in enumerate(img_urls):
            typer.echo(f"\n[{idx+1}/{len(img_urls)}] 处理图片：{img_url}")
            try:
                img_resp = requests.get(img_url, headers=headers, timeout=10)
                image = Image.open(BytesIO(img_resp.content)).convert("RGB")
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                results = yolo_model(image_cv)[0]
                detections = sv.Detections.from_ultralytics(results)

                if len(detections) == 0:
                    typer.echo("未检测到大学Logo")
                    continue

                for i in range(len(detections)):
                    x1, y1, x2, y2 = map(int, detections.xyxy[i])
                    cropped = image_cv[y1:y2, x1:x2]
                    class_id = int(detections.class_id[i])
                    school_label = yolo_model.model.names[class_id]
                    typer.echo(f"检测到大学：{school_label}")

                    try:
                        result = ask_hunyuan(school_label)
                        typer.echo("识别结果：")
                        typer.echo(result)
                        # print(type(result))  <class 'str'>
                    except Exception as e:
                        typer.echo(f"混元识别失败：{str(e)}")

            except Exception as e:
                typer.echo(f"图片加载失败：{str(e)}")

    except Exception as e:
        typer.echo(f"网页加载失败：{str(e)}")

@app.command()
def run_web():
    web_service.run_web()

if __name__ == "__main__":
    app()
