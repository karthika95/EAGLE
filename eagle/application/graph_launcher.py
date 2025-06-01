# Zoho Labs Kottarakkara

import gradio as gr
from PIL import Image
import time
import threading
import queue
import sys
import webbrowser


IMAGE_REFRESH_DELAY = 1.0 # IN SECONDS
path_queue = queue.Queue()

# Background thread that listens for paths
def get_value():
    while True:
        path, command = input().strip().split(",")

        if command =="refresh":
            try:
                with path_queue.mutex:
                    path_queue.queue.clear()
            except:
                pass
            
        path_queue.put(path)

threading.Thread(target=get_value, daemon=True).start()

    



def image_processor():
    try:
        path = path_queue.get_nowait()
        print(path)
        img = Image.open(path)
        return gr.update(value=img)
    except queue.Empty:
        return gr.update()  # No update to the image
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return gr.update()


with gr.Blocks() as slideshow:
    gr.Markdown("# Draft Tree Visualization")

    img_output = gr.Image(height=750, width=1500)

    slideshow.load(image_processor, inputs=None, outputs=img_output)

    timer = gr.Timer(value=IMAGE_REFRESH_DELAY, render=True)  # ⬅️ This sets the interval
    timer.tick(fn=image_processor, outputs=img_output)


slideshow.launch(share=True)