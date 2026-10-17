"""
app.py — Gradio Web Interface for HydroRAG.
"""
import gradio as gr
from config import CROPS, CHROMA_DIR, BM25_INDEX_PATH
from rag_engine import RAGEngine

# ZeroGPU Compatibility: Satisfies HF ZeroGPU startup check without GPU queuing on chat
try:
    import spaces

    @spaces.GPU
    def _zerogpu_probe():
        """Dummy probe to satisfy ZeroGPU startup requirements."""
        pass
except ImportError:
    pass

# Cloud Auto-Init: If database doesn't exist (like on Hugging Face), build it automatically!
if not BM25_INDEX_PATH.exists() or not CHROMA_DIR.exists():
    print("⚡ Database not found. Running initial ingestion...")
    from ingest import main as run_ingestion
    run_ingestion()


# Initialize the RAG engine
engine = RAGEngine()

CROP_LIST = ["All Crops"] + [c.capitalize() for c in CROPS]


def chat(message, history, crop, mode):
    """Processes user question and formats answer with source citations."""
    cleaned = message.strip()
    if not cleaned:
        return ""

    # 1. Quick greeting handling (ignore trailing punctuation)
    greetings = {"hi", "hello", "hey", "hii", "hii there", "hello there", "hey there", "thanks", "thank you"}
    greeting_text = cleaned.lower().strip(" !.?,")
    if greeting_text in greetings:
        return "Hello! I am HydroRAG. Ask me any question about hydroponic crops, nutrient solutions (EC/pH), or environmental controls!"

    # 2. Map radio choice to backend search mode
    mode_map = {
        "Hybrid Search": "hybrid",
        "Vector Only": "dense",
        "Keyword (BM25)": "sparse"
    }
    selected_mode = mode_map.get(mode, "hybrid")

    # 3. Query the HydroRAG engine
    result = engine.query(cleaned, crop_filter=crop.lower(), mode=selected_mode)

    answer = result["answer"]
    citations = result["citations"]

    # Append simple text citations at the end
    if citations:
        source_text = "\n\n**Sources:**\n"
        for c in citations:
            source_text += f"- [{c['index']}] `{c['source']}` (Page {c['page']})\n"
        answer += source_text

    return answer


# Build clean Gradio UI
with gr.Blocks(title="HydroRAG") as demo:
    gr.Markdown("# 🌱 HydroRAG — Precision Hydroponics Assistant")
    gr.Markdown("Ask questions about hydroponic farming grounded in peer-reviewed research papers.")

    with gr.Row():
        crop_dropdown = gr.Dropdown(
            choices=CROP_LIST,
            value="All Crops",
            label="Filter by Crop"
        )
        mode_radio = gr.Radio(
            choices=["Hybrid Search", "Vector Only", "Keyword (BM25)"],
            value="Hybrid Search",
            label="Search Mode"
        )

    # Large spacious chat window
    custom_chatbot = gr.Chatbot(height=750)

    gr.ChatInterface(
        fn=chat,
        chatbot=custom_chatbot,
        additional_inputs=[crop_dropdown, mode_radio]
    )

if __name__ == "__main__":
    demo.launch()
