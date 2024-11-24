import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import gradio as gr
from typing import Callable
import base64

__version__ = "0.0.1"


def get_fn(model_path: str, **model_kwargs):
    """Create a chat function with the specified model."""
    
    # Initialize model once
    llm = Llama(
        model_path=model_path,
        n_ctx=8192,  # Large context window
        n_batch=512,
        **model_kwargs
    )
    
    def predict(
        message: str,
        history,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_k: int,
        repetition_penalty: float,
        top_p: float
    ):
        try:
            # Format conversation with ChatML format
            messages = []
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            # Add conversation history
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})
            
            # Add current message
            messages.append({"role": "user", "content": str(message)})

            # Generate response with streaming
            response_text = ""
            for chunk in llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
            ):
                if chunk and "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        response_text += delta
                        yield response_text.strip()
            
            # Ensure we yield something if no response
            if not response_text.strip():
                yield "I apologize, but I was unable to generate a response. Please try again."
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            yield f"An error occurred: {str(e)}"

    return predict


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
    """Create a Gradio Interface with similar styling and parameters."""
    
    model_path = get_model_path(name, model_path)
    fn = get_fn(model_path, **kwargs)

    interface = gr.ChatInterface(
        fn=fn,
        title="🤖 LLaMA Chat",
        description="Chat with a LLaMA model. Ask me anything!",
        additional_inputs_accordion=gr.Accordion("⚙️ Parameters", open=False),
        additional_inputs=[
            gr.Textbox(
                "You are a helpful AI assistant.",
                label="System prompt"
            ),
            gr.Slider(0, 1, 0.7, label="Temperature"),
            gr.Slider(128, 4096, 1024, label="Max new tokens"),
            gr.Slider(1, 80, 40, label="Top K sampling"),
            gr.Slider(0, 2, 1.1, label="Repetition penalty"),
            gr.Slider(0, 1, 0.95, label="Top P sampling"),
        ],
        theme=gr.themes.Soft(primary_hue="blue"),
    )
    
    return interface
