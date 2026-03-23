import gradio as gr
import json

from agent import agent, build_lc_messages
from langchain_core.messages import AIMessage, ToolMessage


def run_agent(user_message: str, history: list):
    history.append({"role": "user", "content": user_message})
    yield history

    # ✅ Fix 1: history items are plain dicts — use m["role"], not m.role
    lc_messages = build_lc_messages(
        [{"role": m["role"], "content": m["content"]}
         for m in history[:-1]
         if isinstance(m, dict) and m.get("metadata") is None],   # skip tool-call accordion entries
        user_message
    )

    pending_indices: dict[str, int] = {}

    for step in agent.stream({"messages": lc_messages}):
        _, node_state = next(iter(step.items()))

        for msg in node_state.get("messages", []):

            # ── Agent calling a tool → open accordion with spinner ──
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc["args"], indent=2)
                    idx = len(history)
                    history.append({
                        "role": "assistant",
                        "content": f"```json\n{args_str}\n```",
                        "metadata": {
                            "title": f"🛠️ Calling `{tc['name']}`",
                            "status": "pending",
                        }
                    })
                    pending_indices[tc["name"]] = idx
                yield history

            # ── Tool result → close the accordion ──
            elif isinstance(msg, ToolMessage):
                preview = str(msg.content)[:500]
                ellipsis = "..." if len(str(msg.content)) > 500 else ""
                idx = pending_indices.get(msg.name)
                if idx is not None:
                    existing_content = history[idx]["content"]   # ✅ dict access
                    history[idx] = {
                        "role": "assistant",
                        "content": existing_content + f"\n\n**Result:**\n```\n{preview}{ellipsis}\n```",
                        "metadata": {
                            "title": f"✅ `{msg.name}` — done",
                            "status": "done",
                        }
                    }
                yield history

            # ── Final answer ──
            elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                history.append({"role": "assistant", "content": msg.content})
                yield history


AUTOSCROLL_JS = """
function() {
    const observer = new MutationObserver(() => {
        document.querySelectorAll('.chatbot').forEach(el => {
            el.scrollTop = el.scrollHeight;
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
}
"""

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
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
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


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        js=AUTOSCROLL_JS,           # ✅ Fix 2: js belongs here in Gradio 6
    )