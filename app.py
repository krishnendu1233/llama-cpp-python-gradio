import gradio as gr
import llama_cpp_python_gradio

gr.load(
    name='bartowski/Marco-o1-GGUF',
    src=llama_cpp_python_gradio.registry,
).launch()