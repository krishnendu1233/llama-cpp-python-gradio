# `llama-cpp-python-gradio`

is a Python package that makes it easy for developers to create machine learning apps powered by llama.cpp models using Gradio.

# Installation

You can install `llama-cpp-python-gradio` directly using pip:

```bash
pip install llama-cpp-python-gradio
```

# Basic Usage

You'll need a GGUF model file for llama.cpp. The easiest way is to use a Hugging Face model repository:

```python
import gradio as gr
import llama_cpp_python_gradio

# Using a Hugging Face model repository
gr.load(
    name='TheBloke/Llama-2-7B-Chat-GGUF',  # Will automatically select best quantized version
    src=llama_cpp_python_gradio.registry,
).launch()
```

While you can technically use a local model path, it's recommended to use Hugging Face repositories as they provide:
- Automatic selection of the best available quantization (Q8 > Q6 > Q4)
- Easy model versioning and updates
- Verified model compatibility

# Customization 

You can customize the interface by passing additional model parameters:

```python
import gradio as gr
import llama_cpp_python_gradio

gr.load(
    model_path='path/to/your/model.gguf',
    src=llama_cpp_python_gradio.registry,
    n_ctx=8192,      # context window size (default)
    n_batch=512,     # batch size (default)
    # ... other llama.cpp parameters
).launch()
```

The interface includes several adjustable parameters:
- System prompt (default: "You are a helpful AI assistant.")
- Temperature (0-1, default: 0.7)
- Max new tokens (128-4096, default: 1024)
- Top K sampling (1-80, default: 40)
- Repetition penalty (0-2, default: 1.1)
- Top P sampling (0-1, default: 0.95)

# Under the Hood

The `llama-cpp-python-gradio` library combines `llama-cpp-python` and `gradio` to create a chat interface. Key features include:

- Automatic model downloading from Hugging Face (with smart quantization selection)
- ChatML-formatted conversation handling
- Streaming responses
- Support for both text and image inputs (for multimodal models)
- Configurable generation parameters through the UI

-------

Note: Make sure you have a compatible GGUF model file before running the interface. You can download models from sources like Hugging Face or convert existing models to GGUF format.