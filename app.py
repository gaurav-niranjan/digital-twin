import sys
from pathlib import Path

# Make src/ importable without `pip install -e .` (Hugging Face Spaces installs
# requirements before the app code is copied in, so `-e .` can't be used there).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from openai import OpenAI
from digital_twin.context import TWIN_SYSTEM_PROMPT
from digital_twin.tools import tools, handle_tool_calls
from digital_twin.styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr
import os

load_dotenv(override=True)

MODEL_NAME = "deepseek-flash"

deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
openai = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]


def chat(message, history):
    messages = system + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
