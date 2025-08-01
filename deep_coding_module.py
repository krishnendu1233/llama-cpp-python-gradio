import os
import re
import json
from llama_cpp import Llama
from typing import List, Dict, Tuple, Any

# --- Model Management Class ---
# This class wraps the llama_cpp model to provide a consistent interface
# for the deep coding module.
class ModelManager:
    def __init__(self, model: Llama):
        self.model = model

    def generate(self, prompt: str, system_prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generates text from the loaded llama_cpp model."""
        if self.model is None:
            return "❌ Error: No model is loaded."

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ Generation error: {str(e)}"

# --- Helper Functions for Deep Coding ---
def parse_code_block(text: str) -> str:
    """Extracts a Python code block from a text string."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def execute_python_code(code: str) -> str:
    """Executes Python code and captures the output."""
    temp_script_path = "temp_deep_coding_script.py"
    with open(temp_script_path, "w") as f:
        f.write(code)
    
    try:
        # We redirect stderr to stdout to capture both normal output and errors
        process = os.popen(f"python {temp_script_path} 2>&1")
        output = process.read()
        return output
    except Exception as e:
        return f"Execution failed: {e}"
    finally:
        # Clean up the temporary script file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

# --- Deep Coding Logic Class ---
class DeepCoder:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        # Define default system prompts for the deep coding loop
        self.default_prompts = {
            "plan": "You are an expert software architect. Given the following user request, outline a detailed, step-by-step technical plan to achieve it using Python. Break down the task into logical, actionable steps including necessary libraries, data structures, and algorithms. If you need more information, ask clarifying questions.",
            "code_gen": "You are a senior Python developer. Your task is to write a single Python script that perfectly solves the user's request based on the provided plan. Use clear, well-commented code. Do not include explanations outside the code. Start with '```python' and end with '```'.",
            "code_refine": "You are a debugging expert. Your task is to identify and fix the bugs in the provided Python code based on the execution error. Return the complete, corrected code. Use clear, well-commented code. Do not include explanations outside the code. Start with '```python' and end with '```'.",
        }
    
    def generate_plan(self, user_request: str, max_new_tokens: int, temperature: float) -> str:
        """Generates a detailed plan for a given user request."""
        if not user_request:
            return "Please provide a request in the chat first."

        plan_text = self.model_manager.generate(
            prompt=user_request,
            system_prompt=self.default_prompts["plan"],
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        return plan_text

    def run_deep_coding_loop(
        self,
        user_request: str,
        current_plan: str,
        max_debug_attempts: int,
        max_new_tokens: int,
        temperature: float,
        chatbot: List[List[str]]
    ) -> Tuple[str, List[List[str]]]:
        """
        Main loop for the deep coding process: generate -> execute -> debug.
        """
        if not current_plan:
            return "❌ Error: Please finalize a plan before running.", chatbot

        loop_log = []
        execution_status_display = ""
        
        # 1. Initial code generation
        generation_prompt = f"User Request: {user_request}\n\nPlan: {current_plan}"
        code_output = self.model_manager.generate(
            prompt=generation_prompt,
            system_prompt=self.default_prompts["code_gen"],
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        initial_code = parse_code_block(code_output)
        execution_status_display = "➡️ Generated initial code. Executing..."
        chatbot.append(("Code Generation", f"```python\n{initial_code}\n```"))
        loop_log.append({"step": "code_gen", "prompt": generation_prompt, "code": initial_code})

        # 2. Execution and debugging loop
        for attempt in range(max_debug_attempts):
            execution_output = execute_python_code(initial_code)
            loop_log.append({"step": "execution", "attempt": attempt + 1, "output": execution_output})
            
            # Check for success
            if "Traceback" not in execution_output and "Error" not in execution_output and "Exception" not in execution_output:
                execution_status_display = f"✅ Execution successful after {attempt + 1} attempt(s)."
                chatbot.append(("Execution Output", f"```\n{execution_output}\n```"))
                loop_log.append({"step": "success", "message": "Execution successful."})
                break
            
            # If failed, attempt to debug
            execution_status_display = f"❌ Execution failed (Attempt {attempt + 1}/{max_debug_attempts}). Debugging..."
            chatbot.append(("Execution Output", f"```\n{execution_output}\n```"))
            
            debug_prompt = f"Existing Code:\n```python\n{initial_code}\n```\n\nExecution Error:\n{execution_output}\n\nUser Plan: {current_plan}"
            refined_code_output = self.model_manager.generate(
                prompt=debug_prompt,
                system_prompt=self.default_prompts["code_refine"],
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            initial_code = parse_code_block(refined_code_output)
            chatbot.append(("Debugging", f"```python\n{initial_code}\n```"))
            loop_log.append({"step": "refine_gen", "prompt": debug_prompt, "code": initial_code})
        else:
            # If loop completes without breaking, it means all attempts failed
            execution_status_display = f"❌ Failed to debug after {max_debug_attempts} attempts. Check the code and plan."
            loop_log.append({"step": "failure", "message": "Max debug attempts reached."})

        # Log the entire process to a JSON file
        with open("deep_coding_log.json", "w") as f:
            json.dump(loop_log, f, indent=4)
        
        return execution_status_display, chatbot