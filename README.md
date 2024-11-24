# `llama-cpp-python-gradio`

is a Python package that makes it easy for developers to create machine learning apps powered by llama.cpp models using Gradio.

# Installation

You can install `llama-cpp-python-gradio` directly using pip:

```bash
pip install llama-cpp-python-gradio
```

# Basic Usage

First, you'll need a GGUF model file for llama.cpp. Then in a Python file, write:

```python
import gradio as gr
import llama_cpp_python_gradio

gr.load(
    model_path='path/to/your/model.gguf',
    src=llama_cpp_python_gradio.registry,
).launch()
```

Run the Python file, and you should see a Gradio Interface connected to your local llama.cpp model!

# Customization 

You can customize the interface by passing additional arguments to the Llama constructor:

```python
import gradio as gr
import llama_cpp_python_gradio

gr.load(
    model_path='path/to/your/model.gguf',
    src=llama_cpp_python_gradio.registry,
    n_ctx=2048,  # context window size
    n_gpu_layers=1  # number of layers to offload to GPU
).launch()
```

# Under the Hood

The `llama-cpp-python-gradio` library has two main dependencies: `llama-cpp-python` and `gradio`. It provides a "registry" function that creates a Gradio ChatInterface connected to your local llama.cpp model.

The interface supports both text and image inputs (for multimodal models), with automatic handling of file uploads and base64 encoding.

-------

Note: Make sure you have a compatible GGUF model file before running the interface. You can download models from sources like Hugging Face or convert existing models to GGUF format.