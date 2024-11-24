import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import gradio as gr
from typing import Callable
import base64

__version__ = "0.0.1"


def get_fn(model_path: str, preprocess: Callable, postprocess: Callable, chat_format: str = None, **model_kwargs):
    """
    Available chat formats:
    - "llama-2": Llama 2 chat format
    - "chatml": ChatML format (used by mistral/mixtral)
    - "openchat": OpenChat format
    - "vicuna": Vicuna chat format
    - "openbuddy": OpenBuddy chat format
    - "neural-chat": Intel neural chat format
    - "zephyr": Zephyr chat format
    - None: No chat format (use for instruction models)
    """
    # Initialize model once, outside the fn function
    llm = Llama(
        model_path=model_path,
        chat_format=chat_format,
        **model_kwargs
    )
    
    def fn(message, history):
        inputs = preprocess(message, history)
        
        # Create chat completion with streaming
        completion = llm.create_chat_completion(
            messages=inputs["messages"],
            stream=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1
        )
        
        response_text = ""
        for chunk in completion:
            if isinstance(chunk, dict) and 'choices' in chunk:
                delta = chunk['choices'][0].get('delta', {}).get('content', '')
                if delta:
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


def get_model_path(name: str = None, model_path: str = None) -> str:
    """Get the local path to the model, downloading it from HF if necessary."""
    if model_path:
        return model_path
    
    if name:
        if "/" in name:
            repo_id = name
            try:
                # List all files in the repo
                from huggingface_hub import list_repo_files
                files = [f for f in list_repo_files(repo_id) if f.endswith('.gguf')]
                
                # Find best available quantization (Q8 > Q6 > Q4)
                for prefix in ['Q8', 'Q6', 'Q4']:
                    best_match = next((f for f in files if prefix in f), None)
                    if best_match:
                        print(f"Found {prefix} model: {best_match}")
                        return hf_hub_download(
                            repo_id=repo_id,
                            filename=best_match
                        )
                
                # Fallback to first available .gguf file if no quantized version found
                if files:
                    print(f"No quantized version found, using: {files[0]}")
                    return hf_hub_download(
                        repo_id=repo_id,
                        filename=files[0]
                    )
                    
                raise ValueError(f"Could not find any GGUF model file in repository {repo_id}")
            except Exception as e:
                raise ValueError(f"Error accessing repository {repo_id}: {str(e)}")
        else:
            # Fallback to legacy model mapping for backward compatibility
            model_mapping = {
                "llama-3.1-8b-instruct": {
                    "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
                    "filename": "llama-2-7b-chat.q4_K_M.gguf"
                }
            }
            if name not in model_mapping:
                raise ValueError(f"Unknown model name: {name}")
            config = model_mapping[name]
            repo_id = config["repo_id"]
            filename = config["filename"]
            
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename
            )
    
    raise ValueError("Either name or model_path must be provided")


def registry(name: str = None, model_path: str = None, **kwargs):
    """
    Create a Gradio Interface for a llama.cpp model.

    Parameters:
        - name (str, optional): Name of the model to load
        - model_path (str, optional): Path to the GGUF model file
        - kwargs: Additional arguments passed to Llama constructor
    """
    model_path = get_model_path(name, model_path)
    
    pipeline = "chat"  # Currently only supporting chat models
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(model_path, preprocess, postprocess, **kwargs)

    interface = gr.ChatInterface(fn=fn, multimodal=True)
    return interface
