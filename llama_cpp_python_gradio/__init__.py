import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, list_repo_files
import gradio as gr
from typing import Callable, Any, List
import base64
from deep_coding_module import DeepCoder, ModelManager # Import the new module

__version__ = "0.0.1"


# Fixes for the local path and quantization issues are still here
def get_model_path(name: str = None, model_path: str = None) -> str:
    """
    Get the local path to the model, downloading it from HF if necessary.
    This version correctly handles local paths and quantization preferences.
    """
    if model_path:
        # Check if the provided path exists and is a file
        if os.path.exists(model_path) and os.path.isfile(model_path):
            print(f"Using local model file from: {model_path}")
            return model_path
        else:
            raise FileNotFoundError(f"Local model file not found at: {model_path}")

    if not name:
        raise ValueError("Either 'name' or 'model_path' must be provided")

    if "/" in name:
        # This is a Hugging Face repository
        repo_id = name
        try:
            files = [f for f in list_repo_files(repo_id) if f.endswith('.gguf')]
            
            for prefix in ['Q8', 'Q6', 'Q5', 'Q4']:
                best_match = next((f for f in files if prefix in f.upper()), None)
                if best_match:
                    print(f"Found {prefix} quantized model: {best_match}")
                    return hf_hub_download(
                        repo_id=repo_id,
                        filename=best_match
                    )
            
            if files:
                print(f"No preferred quantized version found, using: {files[0]}")
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=files[0]
                )
            
            raise ValueError(f"Could not find any GGUF model file in repository {repo_id}")
        except Exception as e:
            raise ValueError(f"Error accessing repository {repo_id}: {str(e)}")
    else:
        model_mapping = {
            "llama-3.1-8b-instruct": {
                "repo_id": "mradermacher/ReasonFlux-Coder-14B-GGUF",
                "filename": "ReasonFlux-Coder-14B.Q4_K_M.gguf"
            }
        }
        if name not in model_mapping:
            raise ValueError(f"Unknown model name: {name}")
        
        config = model_mapping[name]
        return hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"]
        )

    raise ValueError("Could not determine model path. Check input parameters.")


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
        history: List,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_k: int,
        repetition_penalty: float,
        top_p: float
    ):
        try:
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})
            
            messages.append({"role": "user", "content": str(message)})

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
                try:
                    if chunk and isinstance(chunk, dict) and "choices" in chunk:
                        delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            response_text += delta
                            yield response_text.strip()
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing chunk: {str(e)}")
                    continue

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


def registry(name: str = None, model_path: str = None, **kwargs):
    """Create a Gradio Interface with similar styling and parameters."""
    
    model_path = get_model_path(name, model_path)
    fn = get_fn(model_path, **kwargs)
    
    # Initialize the Llama model here once
    llm_instance = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_batch=512,
        **kwargs
    )

    # Wrap it in our new ModelManager class
    model_manager_state = ModelManager(llm_instance)
    deep_coder_instance = DeepCoder(model_manager_state)

    with gr.Blocks(title="Llama-CPP Gradio Interface") as demo:
        gr.Markdown("# 🚀 Gradio Chat Interface with Llama-CPP")
        gr.Markdown("---")
        
        planning_complete = gr.State(False)

        # Chatbot UI
        with gr.Row():
            chatbot = gr.Chatbot(label="Chatbot", height=400)
            user_input = gr.Textbox(placeholder="Enter your request here...", label="User Request")

        # Chat buttons
        with gr.Row():
            chat_btn = gr.Button("Send to Chat", variant="primary")
            clear_btn = gr.Button("Clear Chat")
        
        # Deep Coding Accordion
        with gr.Accordion("🛠️ Deep Coding Controls", open=False):
            
            with gr.Row():
                plan_btn = gr.Button("Generate Plan", variant="secondary")
                finalize_plan_btn = gr.Button("Finalize Plan", interactive=False, variant="secondary")
                execute_auto_btn = gr.Button("Execute Deep Coding", interactive=False, variant="primary")

            current_plan_display = gr.Textbox(label="Generated Plan", lines=5, interactive=False, visible=True)
            execution_status_display = gr.Textbox(label="Execution Status", lines=5, interactive=False, visible=True)
        
        # Generation Parameters for both Chat and Deep Coding
        with gr.Accordion("⚙️ Parameters", open=False):
            system_prompt_textbox = gr.Textbox(
                "You are a helpful AI assistant.",
                label="System prompt"
            )
            with gr.Row():
                temperature_slider = gr.Slider(0, 1, 0.7, label="Temperature")
                max_new_tokens_slider = gr.Slider(128, 4096, 1024, label="Max new tokens")
            with gr.Row():
                top_k_slider = gr.Slider(1, 80, 40, label="Top K sampling")
                repetition_penalty_slider = gr.Slider(0, 2, 1.1, label="Repetition penalty")
                top_p_slider = gr.Slider(0, 1, 0.95, label="Top P sampling")
        
        # --- Event Handlers ---
        def update_chat(message: str, history: List, system_prompt: str, temperature: float, max_new_tokens: int, top_k: int, repetition_penalty: float, top_p: float):
            # This uses the original predict function, which works with the chat interface
            yield from fn(message, history, system_prompt, temperature, max_new_tokens, top_k, repetition_penalty, top_p)

        def finalize_plan(chatbot_history, current_plan):
            if not current_plan:
                chatbot_history.append(("System", "❌ Please generate a plan first."))
                return chatbot_history, False, gr.Button(interactive=False), gr.Button(interactive=False)
            
            chatbot_history.append(("System", "✅ Plan finalized. You can now execute the deep coding loop."))
            return chatbot_history, True, gr.Button(interactive=True), gr.Button(interactive=True)

        user_input.submit(
            fn=lambda message, history: (history + [[message, None]], ""),
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input],
            queue=False,
        ).then(
            fn=update_chat,
            inputs=[user_input, chatbot, system_prompt_textbox, temperature_slider, max_new_tokens_slider, top_k_slider, repetition_penalty_slider, top_p_slider],
            outputs=[chatbot],
        )

        chat_btn.click(
            fn=lambda message, history: (history + [[message, None]], ""),
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input],
            queue=False,
        ).then(
            fn=update_chat,
            inputs=[user_input, chatbot, system_prompt_textbox, temperature_slider, max_new_tokens_slider, top_k_slider, repetition_penalty_slider, top_p_slider],
            outputs=[chatbot],
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=[],
            outputs=[chatbot, user_input]
        )

        plan_btn.click(
            fn=deep_coder_instance.generate_plan,
            inputs=[user_input, max_new_tokens_slider, temperature_slider],
            outputs=[current_plan_display],
        ).then(
            fn=lambda: (gr.Button(interactive=True), gr.Button(interactive=False)),
            inputs=[],
            outputs=[finalize_plan_btn, execute_auto_btn],
            queue=False
        )

        finalize_plan_btn.click(
            finalize_plan,
            [chatbot, current_plan_display],
            [chatbot, planning_complete, finalize_plan_btn, execute_auto_btn]
        )
        
        execute_auto_btn.click(
            fn=deep_coder_instance.run_deep_coding_loop,
            inputs=[
                user_input, 
                current_plan_display, 
                gr.Slider(1, 10, 3, label="Max Debug Attempts"), # Hard-coding a new slider for attempts
                max_new_tokens_slider, 
                temperature_slider, 
                chatbot
            ],
            outputs=[execution_status_display, chatbot],
        )

    return demo

