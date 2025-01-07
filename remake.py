import tkinter as tk
from tkinter import filedialog, PhotoImage, Toplevel
from PIL import Image, ImageTk
import subprocess
import threading
import shutil
import sys
import os

# 照片的檔案路徑
head_images = ["examples/head/head1.jpg","examples/head/head2.png","examples/head/1-1.png","examples/head/1-2.png","examples/head/1-3.png",
               "examples/head/1-4.png","examples/head/2-2.png","examples/head/2-3.png","examples/head/2-4.png","examples/head/3-1.png",
               "examples/head/3-2.png","examples/head/3-3.png","examples/head/3-4.png"]
body_images = ["examples/body/body1.png","examples/body/1-1.jpg","examples/body/1-2.jpg","examples/body/1-3.jpg","examples/body/1-4.jpg"
               ,"examples/body/2-2.jpg","examples/body/2-3.jpg","examples/body/2-4.jpg","examples/body/3-1.jpg","examples/body/3-4.jpg"]

class EditPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Switcher")
        self.root.geometry("400x550")

        loaded = sys.argv[1]

        if loaded == 'True':
            head_images.insert(0,"examples/head/head.png")
            body_images.insert(0,"input/photo.png")
        else:
            if "examples/head/head.png" in head_images:
                head_images.remove("examples/head/head.png")
            if "examples/body/body.png" in body_images:
                body_images.remove("examples/body/body.png")

        self.head_index = 0
        self.body_index = 0

        self.select_head_button = tk.Button(self.root, text="選擇頭部照片", command=self.select_head)
        self.select_head_button.grid(row=0, column=1)

        self.head_label = tk.Label(self.root)
        self.head_label.grid(row=1, column=1)

        self.prev_head_button = tk.Button(self.root, text="<", command=self.prev_head)
        self.prev_head_button.grid(row=1, column=0)

        self.next_head_button = tk.Button(self.root, text=">", command=self.next_head)
        self.next_head_button.grid(row=1, column=2)

        self.select_body_button = tk.Button(self.root, text="選擇身體照片", command=self.select_body)
        self.select_body_button.grid(row=2, column=1)

        self.body_label = tk.Label(self.root)
        self.body_label.grid(row=3, column=1)

        self.prev_body_button = tk.Button(self.root, text="<", command=self.prev_body)
        self.prev_body_button.grid(row=3, column=0)

        self.next_body_button = tk.Button(self.root, text=">", command=self.next_body)
        self.next_body_button.grid(row=3, column=2)

        self.cancel_button = tk.Button(self.root, text="取消", command=self.cancel)
        self.cancel_button.grid(row=1, column=3, sticky="e", padx=30, pady=50) 

        self.confirm_button = tk.Button(self.root, text="確認", command=self.confirm)
        self.confirm_button.grid(row=3, column=3, sticky="e", padx=30, pady=50)  
        
        self.update_head_image()
        self.update_body_image()

    def resize_image(self, img, max_size):
        """等比例縮小圖像，並保持最長邊為 max_size"""
        original_width, original_height = img.size
        ratio = min(max_size / original_width, max_size / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        return img.resize(new_size, Image.LANCZOS)

    def update_head_image(self):
        img = Image.open(head_images[self.head_index])
        img = self.resize_image(img, 150) 
        self.head_photo = ImageTk.PhotoImage(img)
        self.head_label.config(image=self.head_photo)

    def update_body_image(self):
        img = Image.open(body_images[self.body_index])
        img = self.resize_image(img, 250)  
        self.body_photo = ImageTk.PhotoImage(img)
        self.body_label.config(image=self.body_photo)

    def prev_head(self):
        self.head_index = (self.head_index - 1) % len(head_images)
        self.update_head_image()

    def next_head(self):
        self.head_index = (self.head_index + 1) % len(head_images)
        self.update_head_image()

    def prev_body(self):
        self.body_index = (self.body_index - 1) % len(body_images)
        self.update_body_image()

    def next_body(self):
        self.body_index = (self.body_index + 1) % len(body_images)
        self.update_body_image()

    def select_head(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            head_images.append(file_path)
            self.head_index = len(head_images) - 1
            self.update_head_image()

    def select_body(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            body_images.append(file_path)
            self.body_index = len(body_images) - 1
            self.update_body_image()

    def cancel(self):
        self.root.destroy()
        shutil.copy("check.jpg","results/cancel.jpg")

    def confirm(self):
        shutil.copy(head_images[self.head_index], "results/SUS.png")
        shutil.copy(body_images[self.body_index], "input/photo.jpg")
        self.show_gif()

        self.script_done_event = threading.Event()

        cut_thread = threading.Thread(target=self.run_yolact, args=("input/photo.jpg",))
        cut_thread.start()

        def check_thread():
            if not cut_thread.is_alive():
                self.on_close()
                self.root.destroy()
            else:
                self.gif_window.after(100, check_thread)

        check_thread()

    def show_gif(self):
        self.gif_window = Toplevel()
        self.gif_window.title("Loading...")
        
        # 加載 GIF 動畫
        gif_image = Image.open("loading.gif")
        frames = []
        try:
            while True:
                frames.append(ImageTk.PhotoImage(gif_image.copy()))
                gif_image.seek(len(frames))  # 尋找下一幀
        except EOFError:
            pass  # 到達 GIF 的結尾

        label = tk.Label(self.gif_window)
        label.pack()

        def update_frame(frame_index):
            label.configure(image=frames[frame_index])
            frame_index = (frame_index + 1) % len(frames)
            self.gif_window.after(100, update_frame, frame_index)  # 每100毫秒更新一幀

        update_frame(0)  # 開始更新幀

    def on_close(self):
        self.gif_window.destroy()

    def run_yolact(self,input_path):
        try:
            # 构建命令
            command = [
                "python", "yolact/evalBody.py",
                "--trained_model=weights/yolact_base_2933_17600.pth",
                "--score_threshold=0.15",
                "--top_k=15",
                f"--image={input_path}:input/photo.png"
            ]
                
            # 执行命令并等待完成
            subprocess.run(command, check=True)
            os.remove("input/photo.jpg")

        except subprocess.CalledProcessError as e:
            print(f"执行 eval.py 时出错: {e}")
        except Exception as e:
            print(f"发生错误: {e}")

root = tk.Tk()
app = EditPage(root)
root.mainloop()
