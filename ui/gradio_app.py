import gradio as gr
import json

from agent import agent, build_lc_messages
from langchain_core.messages import AIMessage, ToolMessage


def run_agent(user_message: str, history: list):
    history.append({"role": "user", "content": user_message})
    yield history

    lc_messages = build_lc_messages(
        [m for m in history[:-1] if m.get("metadata") is None],
        user_message
    )

    for step in agent.stream({"messages": lc_messages}):
        _, node_state = next(iter(step.items()))

        for msg in node_state.get("messages", []):

            # 🛠️ Tool call + result مع بعض
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    history.append({
                        "role": "assistant",
                        "content": f"```json\n{tc['args']}\n```",
                        "metadata": {
                            "title": f"🛠️ `{tc['name']}`",
                            "status": "pending",
                        }
                    })
                yield history

            elif isinstance(msg, ToolMessage):
                history.append({
                    "role": "assistant",
                    "content": f"```\n{msg.content}\n```",
                    "metadata": {
                        "title": f"✅ `{msg.name}` result",
                        "status": "done",
                    }
                })
                yield history

            # 💬 Final answer
            elif isinstance(msg, AIMessage) and msg.content:
                history.append({"role": "assistant", "content": msg.content})
                yield history

with gr.Blocks(title="Conversational Agent") as demo:   # ✅ Fix 2: js removed from here

    gr.Markdown("# 🧠 Conversational Agent")
    gr.Markdown(
        "A verified-answer agent that reasons step by step — "
        "tool calls appear inline as collapsible cards."
    )

    chatbot = gr.Chatbot(
        height=600,
        show_label=False,
        elem_classes=["chatbot"],
        avatar_images=(
            None,
            "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg",
        ),
    )

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your question here...",
            show_label=False,
            scale=6,
            lines=1,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    
    clear_btn = gr.Button("🗑️ Clear conversation", variant="secondary", size="sm")

    gr.on(
        triggers=[send_btn.click, user_input.submit],
        fn=run_agent,
        inputs=[user_input, chatbot],
        outputs=[chatbot],
    ).then(fn=lambda: "", outputs=user_input)

    clear_btn.click(fn=lambda: [], outputs=chatbot)


# if __name__ == "__main__":
#     demo.launch(
#         server_port=7860,
#         share=False   
#     )