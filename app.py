import gradio as gr
import llama_cpp_python_gradio

gr.load(
    name='llama-3.1-8b-instruct',
    src=llama_cpp_python_gradio.registry,
).launch()