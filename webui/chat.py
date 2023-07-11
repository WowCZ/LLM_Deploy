# -*- coding:utf-8 -*-
import os
import logging
import sys
import gradio as gr
import torch
from analysis import visit_llm
from webui.modules.utils import *
from webui.modules.presets import *
from webui.modules.overwrites import *

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

# load_8bit = sys.argv[3].lower().startswith("8") if len(sys.argv) > 3 else False
# base_model = sys.argv[1]
# adapter_model = None if sys.argv[2].lower() == "none" else sys.argv[2]
tokenizer = load_tokenizer('/mnt/lustre/chenzhi/workspace/LLM/models/BaiChuan-ShortChat-7B')


def predict(
    text,
    chatbot,
    history,
    llm_url,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
):
    header = {'Content-Type': 'application/json'}
    if text == "":
        yield chatbot, history, "Empty context."
        return

    inputs = generate_prompt_with_history(
        text, history, tokenizer, max_length=max_context_length_tokens
    )
    if inputs is None:
        yield chatbot, history, "Input too long."
        return
    else:
        prompt, inputs = inputs
    
    print('#'*20)
    print(llm_url)
    print(prompt)
    response = visit_llm([llm_url], header, [{'prompt':prompt}])['outputs'][0]
    
    a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
        [text, convert_to_markdown(response)]
    ], history + [[text, response]]
    yield a, b, "Generating..."
        
    if shared_state.interrupted:
        shared_state.recover()
        try:
            yield a, b, "Stop: Success"
            return
        except:
            pass

    print(prompt)
    print(response)
    print("=" * 80)
    try:
        yield a, b, "Generate: Success"
    except:
        pass


def retry(
    text,
    chatbot,
    history,
    llm_url,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, "Empty context."
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict(
        inputs,
        chatbot,
        history,
        llm_url,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
    ):
        yield x

def chat_with_api(url: str, port: int):
    gr.Chatbot.postprocess = postprocess

    with open("assets/webui/custom.css", "r", encoding="utf-8") as f:
        customCSS = f.read()

    with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
        history = gr.State([])
        user_question = gr.State("")
        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")
        gr.Markdown(description_top)
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=5):
                with gr.Row():
                    chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height="100%")
                with gr.Row():
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(
                            show_label=False, placeholder="Enter text"
                        ).style(container=False)
                    with gr.Column(min_width=70, scale=1):
                        submitBtn = gr.Button("Send")
                    with gr.Column(min_width=70, scale=1):
                        cancelBtn = gr.Button("Stop")

                with gr.Row():
                    emptyBtn = gr.Button(
                        "🧹 New Conversation",
                    )
                    retryBtn = gr.Button("🔄 Regenerate")
                    delLastBtn = gr.Button("🗑️ Remove Last Turn")
            with gr.Column():
                with gr.Column(min_width=50, scale=1):
                    with gr.Tab(label="URL Setting"):
                        llm_url = gr.Textbox(
                            label='LLM URL 🔗',
                            value=url
                            )
                    with gr.Tab(label="Parameter Setting"):
                        gr.Markdown("# Parameters")
                        top_p = gr.Slider(
                            minimum=-0,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            interactive=True,
                            label="Top-p",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1,
                            step=0.1,
                            interactive=True,
                            label="Temperature",
                        )
                        max_length_tokens = gr.Slider(
                            minimum=0,
                            maximum=512,
                            value=512,
                            step=8,
                            interactive=True,
                            label="Max Generation Tokens",
                        )
                        max_context_length_tokens = gr.Slider(
                            minimum=0,
                            maximum=4096,
                            value=2048,
                            step=128,
                            interactive=True,
                            label="Max History Tokens",
                        )
        gr.Markdown(description)

        predict_args = dict(
            fn=predict,
            inputs=[
                user_question,
                chatbot,
                history,
                llm_url,
                top_p,
                temperature,
                max_length_tokens,
                max_context_length_tokens,
            ],
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )
        retry_args = dict(
            fn=retry,
            inputs=[
                user_input,
                chatbot,
                history,
                llm_url,
                top_p,
                temperature,
                max_length_tokens,
                max_context_length_tokens,
            ],
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )

        reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])

        # Chatbot
        cancelBtn.click(cancel_outputing, [], [status_display])
        transfer_input_args = dict(
            fn=transfer_input,
            inputs=[user_input],
            outputs=[user_question, user_input, submitBtn, cancelBtn],
            show_progress=True,
        )

        user_input.submit(**transfer_input_args).then(**predict_args)

        submitBtn.click(**transfer_input_args).then(**predict_args)

        # user_input.submit(**predict_args)
        # submitBtn.click(**predict_args)

        emptyBtn.click(
            reset_state,
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )
        emptyBtn.click(**reset_args)

        retryBtn.click(**retry_args)

        delLastBtn.click(
            delete_last_conversation,
            [chatbot, history],
            [chatbot, history, status_display],
            show_progress=True,
        )

    demo.title = "ChatX"

    reload_javascript()
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
        share=True, favicon_path="./assets/favicon.ico"
    )
