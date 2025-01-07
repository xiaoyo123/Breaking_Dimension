import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Toplevel
import cv2
from PIL import Image, ImageTk
import shutil
import subprocess
import os
from  pathlib import Path
import multiprocessing as mp
import threading
import sys
import time


loaded = False

class MainPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimedia Display")

        # 设置窗口宽扁
        self.root.geometry("1500x600")  # 设置窗口的初始大小为更宽扁的比例

        # Left frame
        self.left_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        self.left_frame.grid(row=1, column=0, sticky="nsew")

        # Configure left frame to expand automatically
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        # Create a new frame for the buttons at the top, 横跨整个页面
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # 按钮横向排列
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        self.button_frame.grid_columnconfigure(3, weight=1)
        self.button_frame.grid_columnconfigure(4, weight=1)
        self.button_frame.grid_columnconfigure(5, weight=1)

        # Take photo button
        self.take_photo_btn = ttk.Button(self.button_frame, text="Take Photo", command=self.take_photo)
        self.take_photo_btn.grid(row=0, column=0, padx=5, pady=5)

        # Import image button
        self.import_image_btn = ttk.Button(self.button_frame, text="Import Image", command=self.import_image)
        self.import_image_btn.grid(row=0, column=1, padx=5, pady=5)

        # Edit image button
        self.edit_image_btn = ttk.Button(self.button_frame, text="Edit Image", command=self.edit)
        self.edit_image_btn.grid(row=0, column=2, padx=5, pady=5)

        # Cartoon button
        self.cartoonify_btn = ttk.Button(self.button_frame, text="Cartoonify", command=self.cartoonify)
        self.cartoonify_btn.grid(row=0, column=3, padx=5, pady=5)

        # Run button
        self.run_btn = ttk.Button(self.button_frame, text="Run", command=self.run)
        self.run_btn.grid(row=0, column=4, padx=5, pady=5)

        # Add a label frame to hold image display (left bottom corner for photo)
        self.left_bottom_frame = ttk.LabelFrame(self.left_frame, text=' Image Display ', padding=(10, 10, 10, 10))
        self.left_bottom_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Placeholder for image display
        self.image_label = tk.Label(self.left_bottom_frame)
        self.image_label.grid(row=0, column=0, pady=5, padx=5)

        # Right frame for gif
        self.right_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        self.right_frame.grid(row=1, column=1, sticky="nsew")

        # Add a label frame to hold video display (right bottom corner for gif)
        self.right_frame_content = ttk.LabelFrame(self.right_frame, text=' Result Display ', padding=(10, 10, 10, 10))
        self.right_frame_content.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # gif canvas for displaying video (right bottom corner)
        self.gif_canvas = tk.Canvas(self.right_frame_content, width=768, height=432)
        self.gif_canvas.grid(row=1, column=0, pady=5)

        self.is_gif_playing = False
        self.loop_gif = True  # Set to True to loop the GIF

        # Initialize webcam-related attributes
        self.webcam_capture = None
        self.is_webcam_active = False

        # Add slider, entry, and button to modify txt file value (placed vertically to the right of gif canvas)
        self.slider_frame = ttk.Frame(self.right_frame_content)
        self.slider_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")

        # Add a scale widget (slider) with range from 2.5 to 4, default value 3, and vertical orientation
        self.value_slider = ttk.Scale(self.slider_frame, from_=4, to=2.5, orient="vertical", command=self.update_entry)
        self.value_slider.grid(row=0, column=0, padx=5, pady=5)
        
         # Add an entry to display the current value of the slider, centered
        self.value_entry = ttk.Entry(self.slider_frame, justify='center', width=8)
        self.value_entry.grid(row=1, column=0, padx=5, pady=5)
        self.value_entry.insert(0, "3.0")  # Initialize with default value 3

        # Add a button to save the value to a txt file
        self.save_btn = ttk.Button(self.slider_frame, text="Save to File", command=self.save_value_to_file)
        self.save_btn.grid(row=2, column=0, padx=5, pady=5)

        self.value_slider.set(3)  # Set default value to 3

        self.value_entry.bind('<Return>', self.update_slider_from_entry)

    def import_image(self):
        self.is_gif_playing = False
        self.update_gif()
        # Ask user to select an image file
        file_path = filedialog.askopenfilename(filetypes=[
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
            ("JPEG files", "*.jpeg"),
        ])
        
        if file_path:
            #confirm = messagebox.askyesno('Confirm', 'Are you sure you want to select this file?')
            
            if file_path:
                # Destination path where you want to copy the selected image
                destination_path = 'input/photo.png'  # Replace with your desired destination folder
                
                try:
                    # Copy the selected image file to the destination folder
                    shutil.copy(file_path, destination_path)
                    #messagebox.showinfo('Success', 'Image copied successfully!')
                except Exception as e:
                    messagebox.showerror('Error', f'Error copying image: {str(e)}')
            else:
                messagebox.showinfo('Cancelled', 'File selection cancelled.')
        self.image_label.configure(image=None)
        self.image_label.image = None
        self.load_image(destination_path)

    def edit(self):
        if loaded:
            self.edit_image()
        else:
            self.edit_no_import()

    def edit_image(self):
        self.show_gif()

        self.script_done_event = threading.Event()

        cut_thread = threading.Thread(target=self.run_yolact, args=("input/photo.png",))
        cut_thread.start()

        def check_thread():
            if not cut_thread.is_alive():
                self.on_close()
                self.run_remake("remake.py")
                if loaded:
                    if not os.path.exists("results/cancel.jpg"):
                        self.is_gif_playing = False
                        self.update_gif()
                        self.load_image("input/photo.png")
                    else:
                        os.remove("results/cancel.jpg")
            else:
                self.gif_window.after(100, check_thread)

        check_thread()

    def edit_no_import(self):
        self.run_remake("remake.py")
        if not os.path.exists("results/cancel.jpg"):
            self.load_image("input/photo.png")
        else:
            os.remove("results/cancel.jpg")

    def run_another_script_win(self, script_path):
        try:
            # 获取 Windows 下虚拟环境的 Python 路径
            venv_python = os.path.join("nenv", "Scripts", "python.exe")
            
            process = subprocess.Popen([venv_python, script_path])
            process.wait()
        except Exception as e:
            print(f"执行脚本时出错: {e}")
        
    def run_webcam(self,script_path):
        try:
            # 獲取當前 Conda 虛擬環境的 Python 路徑
            conda_env = os.environ.get("CONDA_PREFIX")  # 獲取當前 Conda 環境的根目錄
            if conda_env:
                venv_python = os.path.join(conda_env, "bin", "python")  # Linux 下 Conda 的 Python 路徑

                #script_path = "remake.py"  # 你想啟動的 Python 檔案的路徑
                process = subprocess.run([venv_python, script_path])
                # process.wait()
            else:
                print("Conda 虛擬環境未啟動")
        except Exception as e:
            print(f"執行腳本時出錯: {e}")

    def run_remake(self,script_path):
        try:
            # 獲取當前 Conda 虛擬環境的 Python 路徑
            conda_env = os.environ.get("CONDA_PREFIX")  # 獲取當前 Conda 環境的根目錄
            if conda_env:
                venv_python = os.path.join(conda_env, "bin", "python")  # Linux 下 Conda 的 Python 路徑

                loaded_arg = 'True' if loaded else 'False'

                #script_path = "remake.py"  # 你想啟動的 Python 檔案的路徑
                #process = subprocess.run([venv_python, script_path,loaded_arg],capture_output=True, text=True)
                process = subprocess.run([venv_python, script_path,loaded_arg])
                process.wait()
            else:
                print("Conda 虛擬環境未啟動")
        except Exception as e:
            print(f"執行腳本時出錯: {e}")

    def load_image(self, path):
        image = Image.open(path)
        
        # Define the maximum width and height
        max_width, max_height = 520, 380
        
        # Get the original dimensions of the image
        original_width, original_height = image.size
        
        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height
        
        # Determine the new dimensions
        if (original_width > max_width) or (original_height > max_height):
            if (max_width / original_width) < (max_height / original_height):
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
        else:
            new_width, new_height = original_width, original_height

        # Resize the image
        image = image.resize((new_width, new_height), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)
        
        self.image_label.configure(image=self.photo)
        self.image_label.image = self.photo  # Keep reference to avoid garbage collection
        global loaded
        loaded = True
        self.value_entry.insert(0 , "3.0")
        self.value_slider.set(3)
        self.save_value_to_file()
            
    def import_gif(self, gif_path, size=(768, 432)):
        try:
            if gif_path is None or not os.path.isfile(gif_path):
                raise Exception("Error: GIF file does not exist.")

            # Open the GIF file using PIL
            self.gif_image = Image.open(gif_path)

            # Resize the GIF to fit within the specified size while maintaining aspect ratio
            max_width, max_height = size
            self.resized_gif_image = self.resize_image_with_aspect_ratio(self.gif_image, (max_width, max_height))
            self.is_gif_playing = True

            # Display the first frame of the GIF
            self.show_gif_frame(self.resized_gif_image)
            self.update_gif()

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred while importing GIF: {e}")
    
    def resize_image_with_aspect_ratio(self, image, max_size):
        # Calculate the aspect ratio and new size
        width, height = image.size
        max_width, max_height = max_size

        if width > height:
            new_width = min(width, max_width)
            new_height = int((new_width / width) * height)
        else:
            new_height = min(height, max_height)
            new_width = int((new_height / height) * width)

        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def show_gif_frame(self, gif_image):
        # Convert the PIL image to a format that can be displayed with Tkinter
        self.tk_gif_image = ImageTk.PhotoImage(gif_image)

        # Clear the canvas before drawing the new image
        self.gif_canvas.delete("all")
        
        # Draw the image on the canvas
        self.gif_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_gif_image)

    def update_gif(self):
        if self.is_gif_playing:
            try:
                # Get the next frame in the GIF
                self.gif_image.seek(self.gif_image.tell() + 1)
                self.resized_gif_image = self.gif_image.resize((self.gif_canvas.winfo_width(), self.gif_canvas.winfo_height()), Image.LANCZOS)
                self.show_gif_frame(self.resized_gif_image)

                # Schedule the next frame update
                self.root.after(self.gif_image.info['duration'], self.update_gif)
            except EOFError:
                # Loop the GIF or stop the playback if needed
                if self.loop_gif:
                    self.gif_image.seek(0)
                    self.update_gif()
                else:
                    self.is_gif_playing = False
        else:
            self.gif_canvas.delete("all")

    def take_photo(self):
        
        self.run_webcam("detect.py")
        self.is_gif_playing = False
        self.update_gif()
        self.image_label.configure(image=None)
        self.image_label.image = None
        if not os.path.exists("./results/cancel.jpg"):    
            self.load_image("./input/photo.png")
        else:
            os.remove("./results/cancel.jpg")
            
    def run(self):
        if loaded:
            try:
                #self.cartoonify()
                #卡通圖複製
                #self.copyimage()
                self.is_gif_playing = False
                self.update_gif()
                self.import_gif('loading.gif')
                def worker():
                    self.run_merge_and_import_gif()
                    root.quit()
                threading.Thread(target=worker).start()
                root.mainloop()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error running program: {e}")
        else:
            messagebox.showerror("Error", f"Import a photo or run cartoonify first!")

    def run_merge_and_import_gif(self):
        try:
            self.merge()
            self.root.after(3000)
            self.import_gif('results/animated.gif')
        
        except Exception as e:
            # 捕获异常并显示错误信息
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error running background task: {e}"))
            
    def cartoonify(self):
        if loaded:
        #if os.path.exists('input/photo.png'):
            self.is_gif_playing = False
            self.update_gif()
            try:
                self.gif_canvas.delete("all")
                
                # Replace with the path to your Python script
                script_path = "test.py"

                # Run the script using subprocess
                process = subprocess.Popen(["python", script_path])
                process.wait()

            except Exception as e:
                messagebox.showerror("Error", f"Error running program: {e}")
            self.copyimage()
            self.load_image('input/photo.png')
        else:
            messagebox.showerror("Error", f"Import or take a photo first!")
            
    def get_latest_image_file(self,folder):
        # 定義影像檔案的副檔名
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        
        latest_file = None
        latest_mtime = 0
        
        # 確保資料夾存在
        if not folder.exists():
            return None

        # 遍歷資料夾中的所有檔案
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                mtime = file.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = file
                    
        return latest_file

    def copyimage(self):
        # 定義下載資料夾和目的資料夾的路徑
        downloads_folder = Path.home() / '下載'
        destination_folder = Path('input')
        destination_file_name = 'photo.png'
        destination_file_path = destination_folder / destination_file_name

        # 找到最後編輯的影像檔案
        latest_image = self.get_latest_image_file(downloads_folder)

        if latest_image:
            # 複製影像檔案到目的地
            shutil.copy2(latest_image, destination_file_path)
            print(f"檔案 {latest_image.name} 已經被複製到 {destination_folder}。")
        else:
            print("下載資料夾中沒有影像檔案。")

    def merge(self):
        try:
            # Replace with the path to your shell script
            script_path = "run.sh"

            # Run the script using subprocess
            process = subprocess.Popen(["/bin/bash", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            # Check for errors
            if process.returncode != 0:
                raise Exception(f"Script failed with error code {process.returncode}: {stderr.decode()}")

            # Optionally, you can handle the output of the script
            print(stdout.decode())

        except Exception as e:
            messagebox.showerror("Error", f"Error running program: {e}")
    
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
                "python", "yolact/evalHead.py",
                "--trained_model=weights/yolact_base_2933_17600.pth",
                "--score_threshold=0.15",
                "--top_k=15",
                f"--image={input_path}:examples/head/head.png"
            ]
            
            # 执行命令并等待完成
            subprocess.run(command, check=True)

        except subprocess.CalledProcessError as e:
            print(f"执行 evalHead.py 时出错: {e}")
        except Exception as e:
            print(f"发生错误: {e}")

    def run_yolact_merge(self,input_path):
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

        except subprocess.CalledProcessError as e:
            print(f"执行 evalBody.py 时出错: {e}")
        except Exception as e:
            print(f"发生错误: {e}")

    def update_entry(self, value):
        """Update the entry when the slider is moved."""
        self.value_entry.delete(0, tk.END)
        self.value_entry.insert(0, f"{float(value):.1f}")

    def save_value_to_file(self):
        """Save the current slider value to a txt file."""
        value = self.value_entry.get()
        try:
            with open('./blenderScript/variable.txt', 'w') as f:
                f.write(f"{value}")
        except Exception as e:
            print(f"Error saving value to file: {e}")

    def update_slider_from_entry(self, event):
        """Update the slider value when the entry is manually changed."""
        try:
            # Get the value from the entry and update the slider
            value = float(self.value_entry.get())
            if 2.5 <= value <= 4:
                self.value_slider.set(value)
            else:
                print("Value out of range. Please enter a value between 2.5 and 4.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

if __name__ == "__main__":
    # value = 3.0
    # try:
    #     with open('./blenderScript/variable.txt', 'w') as f:
    #         f.write(f"{value}")
    # except Exception as e:
    #     print(f"Error saving value to file: {e}")
    root = tk.Tk()
    app = MainPage(root)
    root.mainloop()
