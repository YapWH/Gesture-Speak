# import gradio as gr

# # def greet(name, intensity):
# #     return "Hello " * intensity + name + "!"

# demo = gr.Interface(lambda x: x, gr.Image(sources="webcam", streaming=True), "image", live=True)
# # fn=greet,
# # inputs=["text", "slider"],
# # outputs=["text"],)


# demo.launch(share=True)

import numpy as np
import gradio as gr


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with gr.Blocks() as demo:
    gr.Markdown("English Sign Language translator - better experience for the needed people")
    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Video()
        image_button = gr.Button("Start to Identify")

    with gr.Accordion("Open for More infoirmation!", open=False):
        gr.Markdown("Look at me...")
        temp_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.1,
            step=0.1,
            interactive=True,
            label="Slide me",
        )
        temp_slider.change(lambda x: x, [temp_slider])

    #image_button.click(flip_image, inputs=image_input, outputs=image_output)

if __name__ == "__main__":
    demo.launch(share=True)
