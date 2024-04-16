import gradio as gr

def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=greet, inputs=gr.Textbox(label="Input review text here.", outputs="text")
iface.launch()