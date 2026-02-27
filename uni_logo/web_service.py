from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import cv2
import typer
import json
import uvicorn
import supervision as sv
from ultralytics import YOLO
from zidian_uni import LABEL_TO_SCHOOL_NAME
from tencentcloud.common import credential
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

app = typer.Typer()

web_app = FastAPI()

web_app.mount("/static", StaticFiles(directory="."), name="static")

yolo_model = YOLO("./runs/train/../best.pt")

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

        最后，请配上几句你对这个大学的评价，以及预测一下这个大学在未来十年中的发展趋势。
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

def detect_logos(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    output_texts = []
    annotator = sv.BoxAnnotator()
    annotated_image = annotator.annotate(scene=image, detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        school_label = yolo_model.model.names[class_id]
        try:
            result = ask_hunyuan(school_label)
            output_texts.append(f"\U0001F393 校徽：{school_label}\n\n{result}")
        except Exception as e:
            output_texts.append(f"⚠️ 识别失败：{str(e)}")

    return annotated_image, output_texts

@web_app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>中国最强大学 Logo 识别系统</title>
        <style>
            body {
                font-family: "Helvetica Neue", sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f6f8;
                color: #333;
            }
            header {
                background-color: #007acc;
                color: white;
                padding: 20px;
                text-align: center;
            }
            .container {
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            input[type="file"] {
                display: block;
                margin: 20px 0;
            }
            input[type="submit"] {
                background-color: #007acc;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #005fa3;
            }
            footer {
                text-align: center;
                font-size: 13px;
                margin-top: 50px;
                color: #777;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>🎓 中国最强大学 Logo 识别系统</h1>
            <p>上传一张图像，我们将识别其中的大学校徽并进行展示</p>
        </header>
        <div class="container">
            <form action="/upload" method="post" enctype="multipart/form-data">
                <label for="file">选择图像文件（支持 JPG/PNG）:</label>
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="开始识别">
            </form>
        </div>
        <footer>
            &copy; 2025 Zening Li & 中国最强大学识别项目
        </footer>
    </body>
    </html>
    """

@web_app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    upload_path = "uploaded_image.jpg"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    annotated, descriptions = detect_logos(upload_path)
    annotated_path = "web_output.jpg"
    cv2.imwrite(annotated_path, annotated)

    desc_html = "".join([
        f"<div class='desc-box'><pre>{d}</pre></div>" for d in descriptions
    ])

    return f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>识别结果 - 大学Logo识别系统</title>
        <style>
            body {{
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                background-color: #f4f6f8;
                color: #333;
                text-align: center;
                padding: 30px;
            }}
            h3 {{
                color: #2c3e50;
            }}
            img {{
                margin-top: 20px;
                max-width: 90%;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .desc-box {{
                background-color: #ffffff;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px auto;
                width: 80%;
                max-width: 800px;
                text-align: left;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border-radius: 5px;
                white-space: pre-wrap;
            }}
            a {{
                display: inline-block;
                margin-top: 30px;
                padding: 10px 20px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s ease;
            }}
            a:hover {{
                background-color: #2980b9;
            }}
        </style>
    </head>
    <body>
        <h3>🎓 识别结果如下：</h3>
        <img src='/static/{annotated_path}' alt="识别结果图像">
        {desc_html}
        <br><a href="/">🔙 返回上传页面</a>
    </body>
    </html>
    """

@app.command()
def run_web():
    typer.echo("启动 Web UI...")
    uvicorn.run(web_app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    app()

