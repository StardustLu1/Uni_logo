import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, StringVar
from tkinter import ttk
import threading
import yolov12_Hunyuan
import sys
import io
from ttkthemes import ThemedTk
import webbrowser

class StdoutRedirector(io.TextIOBase):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, s):
        try:
            if isinstance(s, bytes):
                s = s.decode('utf-8')
            elif not isinstance(s, str):
                s = str(s)
        except Exception as e:
            s = f"[日志解码失败]: {e}\n"
        try:
            s = s.replace('\r', '')
            self.text_widget.insert(tk.END, s)
            self.text_widget.see(tk.END)
            self.text_widget.update()
        except Exception as e:
            self.text_widget.insert(tk.END, f"[输出失败]: {e}\n")
            self.text_widget.see(tk.END)

    def flush(self):
        pass

class App:
    def __init__(self, root):
        self.root = root
        self.is_dark_mode = False  # 默认为浅色模式
        self.root.title("🎓 大学 Logo 识别系统")
        self.root.geometry("920x620")
        self.root.minsize(880, 500)

        self.create_widgets()
        self.configure_styles()
        self.redirect_logs()

    def configure_styles(self):
        self.style = ttk.Style()
        self.set_light_mode()

    def set_dark_mode(self):
        self.root.configure(bg="#2e2e2e")
        self.style.theme_use("equilux")
        self.style.configure("TLabel", background="#2e2e2e", foreground="white")
        self.style.configure("TButton", background="#444", foreground="white")
        self.log_box.configure(bg="#1e1e1e", fg="#dcdcdc", insertbackground="white")
        self.is_dark_mode = True

    def set_light_mode(self):
        self.root.configure(bg="#f0f2f5")
        self.style.theme_use("arc")
        self.style.configure("TLabel", background="#f0f2f5", foreground="black")
        self.style.configure("TButton", background="#eee", foreground="black")
        self.log_box.configure(bg="#ffffff", fg="#333333", insertbackground="black")
        self.is_dark_mode = False

    def toggle_theme(self):
        if self.is_dark_mode:
            self.set_light_mode()
        else:
            self.set_dark_mode()

    def create_widgets(self):
        # 切换主题按钮
        ttk.Button(self.root, text="🌓 切换主题", command=self.toggle_theme).grid(row=0, column=2, sticky="e", padx=10, pady=10)

        ttk.Label(self.root, text="🎓 大学 Logo 识别系统", font=("微软雅黑", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=(10, 20), padx=10, sticky="w")

        # 摄像头检测
        ttk.Button(self.root, text="📷 摄像头检测", command=self.run_detect_camera).grid(row=1, column=0, padx=10, pady=5, sticky="w")

        # 图片识别
        ttk.Button(self.root, text="🖼️ 选择图片检测", command=self.select_image).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.image_path_var = StringVar()
        ttk.Entry(self.root, textvariable=self.image_path_var, width=60).grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="we")

        # 视频识别
        ttk.Button(self.root, text="🎬 选择视频检测", command=self.select_video).grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.video_path_var = StringVar()
        ttk.Entry(self.root, textvariable=self.video_path_var, width=60).grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="we")

        # 网页识别
        ttk.Label(self.root, text="🌐 网页URL检测").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.url_var = StringVar()
        ttk.Entry(self.root, textvariable=self.url_var, width=60).grid(row=4, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(self.root, text="🔎 网页图片识别", command=self.run_detect_url).grid(row=4, column=2, padx=10, pady=5, sticky="w")

        # Web 服务
        ttk.Button(self.root, text="🚀 启动 Web 服务", command=self.run_web_service).grid(row=5, column=0, pady=10, padx=10, sticky="w")

        # 日志区域
        ttk.Label(self.root, text="📄 识别日志", font=("微软雅黑", 12)).grid(row=6, column=0, columnspan=3, sticky="w", padx=10)
        self.log_box = scrolledtext.ScrolledText(self.root, width=110, height=20, font=("Consolas", 10))
        self.log_box.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        # 响应式布局
        for i in range(3):
            self.root.columnconfigure(i, weight=1)
        self.root.rowconfigure(7, weight=1)

    def redirect_logs(self):
        sys.stdout = StdoutRedirector(self.log_box)
        sys.stderr = StdoutRedirector(self.log_box)

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.root.update()

    def run_detect_camera(self):
        threading.Thread(target=lambda: self._run_and_log("摄像头检测", yolov12_Hunyuan.detect_camera)).start()

    def select_image(self):
        path = filedialog.askopenfilename(title="选择图片", filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path_var.set(path)
            self.run_detect_image(path)

    def run_detect_image(self, path):
        threading.Thread(target=lambda: self._run_and_log(f"图片识别 - {path}", lambda: yolov12_Hunyuan.detect_image(path))).start()

    def select_video(self):
        path = filedialog.askopenfilename(title="选择视频", filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.video_path_var.set(path)
            self.run_detect_video(path)

    def run_detect_video(self, path):
        threading.Thread(target=lambda: self._run_and_log(f"视频识别 - {path}", lambda: yolov12_Hunyuan.detect_video(path))).start()

    def run_detect_url(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("提示", "请输入有效的网页URL")
            return
        threading.Thread(target=lambda: self._run_and_log(f"网页图片识别 - {url}", lambda: yolov12_Hunyuan.detect_url_images(url))).start()

    def run_web_service(self):
        def task():
            try:
                self.log("[开始] 启动 Web 服务...")
                yolov12_Hunyuan.run_web()  # 这个是阻塞的，需确保它支持非阻塞或已在后台运行
            except Exception as e:
                self.log(f"[错误] Web服务启动失败: {e}")

        threading.Thread(target=task, daemon=True).start()

        try:
            self.log("[提示] 尝试打开浏览器访问 http://127.0.0.1:7860 ...")
            webbrowser.open("http://127.0.0.1:7860")
        except Exception as e:
            self.log(f"[错误] 无法打开浏览器: {e}")

    def _run_and_log(self, name, func):
        try:
            self.log(f"[开始] {name}...")
            func()
            self.log(f"[完成] {name}。")
        except Exception as e:
            self.log(f"[错误] {name}失败: {e}")


if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # 默认用 arc 主题，深色切换为 equilux
    app = App(root)
    root.mainloop()
