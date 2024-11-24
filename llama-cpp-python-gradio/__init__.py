import os
from llama_cpp import Llama
import gradio as gr
from typing import Callable
import base64

__version__ = "0.0.3"


def get_fn(model_path: str, preprocess: Callable, postprocess: Callable, **model_kwargs):
    def fn(message, history):
        inputs = preprocess(message, history)
        llm = Llama(model_path=model_path, **model_kwargs)
        
        # Create chat completion with streaming
        completion = llm.create_chat_completion(
            messages=inputs["messages"],
            stream=True
        )
        
        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                response_text += delta
                yield postprocess(response_text)

    return fn


def get_image_base64(url: str, ext: str):
    with open(url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return "data:image/" + ext + ";base64," + encoded_string


def handle_user_msg(message: str):
    if type(message) is str:
        return message
    elif type(message) is dict:
        if message["files"] is not None and len(message["files"]) > 0:
            ext = os.path.splitext(message["files"][-1])[1].strip(".")
            if ext.lower() in ["png", "jpg", "jpeg", "gif", "pdf"]:
                encoded_str = get_image_base64(message["files"][-1], ext)
            else:
                raise NotImplementedError(f"Not supported file type {ext}")
            content = [
                    {"type": "text", "text": message["text"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_str,
                        }
                    },
                ]
        else:
            content = message["text"]
        return content
    else:
        raise NotImplementedError


def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):
            messages = []
            files = None
            for user_msg, assistant_msg in history:
                if assistant_msg is not None:
                    messages.append({"role": "user", "content": handle_user_msg(user_msg)})
                    messages.append({"role": "assistant", "content": assistant_msg})
                else:
                    files = user_msg
            if type(message) is str and files is not None:
                message = {"text":message, "files":files}
            elif type(message) is dict and files is not None:
                if message["files"] is None or len(message["files"]) == 0:
                    message["files"] = files
            messages.append({"role": "user", "content": handle_user_msg(message)})
            return {"messages": messages}

        postprocess = lambda x: x
    else:
        # Add other pipeline types when they will be needed
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    # Determine the pipeline type based on the model name
    # For simplicity, assuming all models are chat models at the moment
    return "chat"


def registry(name: str = None, model_path: str = None, **kwargs):
    """
    Create a Gradio Interface for a llama.cpp model.

    Parameters:
        - name (str, optional): Name of the model to load
        - model_path (str, optional): Path to the GGUF model file
        - kwargs: Additional arguments passed to Llama constructor
    """
    if name and not model_path:
        # Add logic here to map model names to paths
        # This is just an example - you should implement your own mapping
        model_mapping = {
            "llama-3.1-8b-instruct": "path/to/llama-3.1-8b-instruct.gguf"
        }
        if name not in model_mapping:
            raise ValueError(f"Unknown model name: {name}")
        model_path = model_mapping[name]
    
    if not model_path:
        raise ValueError("Either name or model_path must be provided")

    pipeline = "chat"  # Currently only supporting chat models
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(model_path, preprocess, postprocess, **kwargs)

    interface = gr.ChatInterface(fn=fn, multimodal=True)
    return interface
