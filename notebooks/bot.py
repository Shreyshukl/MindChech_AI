# =========================
# MindCheck Telegram Bot
# =========================

import faiss, pickle
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import nest_asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

nest_asyncio.apply()

# -------------------------
# Load FAISS data
# -------------------------
import os
BASE_DIR = os.path.dirname(__file__)  # folder where bot.py is located

index = faiss.read_index(os.path.join(BASE_DIR, "faiss_index.bin"))
with open(os.path.join(BASE_DIR, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)
with open(os.path.join(BASE_DIR, "texts.pkl"), "rb") as f:
    texts = pickle.load(f)

# -------------------------
# SentenceTransformer
# -------------------------
model = SentenceTransformer("intfloat/multilingual-e5-base")

def search(query, top_k=5):
    query_with_prefix = f"query: {query}"
    query_vec = model.encode([query_with_prefix], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append({
            "distance": float(distances[0][i]),
            "text": texts[idx],
            "meta": metadata[idx]
        })
    return results if results else [{"distance": None, "text": "No similar cases found.", "meta": {}}]

# -------------------------
# Gemini setup
# -------------------------
genai.configure(api_key="AIzaSyAaynt5eBqIvaCygtu9l8UFhQHM-6-GFrQ")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def generate_response(user_input, retrieved_results, history, stage):
    if stage == 0:
        return "I hear you. Can you tell me a bit more about how this affects you?"

    context_text = "\n".join(
        [f"- {r['meta'].get('disorder', 'Unknown')}: {r['text']}" for r in retrieved_results]
    )

    system_instruction = """You are MindCheck, a supportive mental health chatbot.
-  frame the answer strictly in bullet points
-  First, analyse the retrieved context and tell the most possible condition in **bold letters** (important)
-  Then, explain the reason for this suggestion based on the user’s context
-  Always mention that information is based on the NIMH website: https://www.nimh.nih.gov
-  Keep responses short and conversational
-  Do NOT give medical advice unless the user types 'what should I do'
-  If the user asks 'what should I do', suggest safe self-care strategies and write **NOTE: encourage seeking professional help**
"""

    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    prompt = f"""{system_instruction}

Conversation so far:
{history_text}

User: {user_input}

Context from similar cases:
{context_text}

Assistant:"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# -------------------------
# Conversation state per user
# -------------------------
user_states = {}  # key = chat_id, value = {"history": [], "stage": 0}

# -------------------------
# Telegram handlers
# -------------------------
TOKEN = "8453597341:AAFHdF43XY42KhytmPwkgw4n1Kgu1qQgsJ4"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_states[chat_id] = {"history": [], "stage": 0}
    await update.message.reply_text("Hello! I’m MindCheck AI. Send me how you’re feeling today.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in user_states:
        user_states[chat_id] = {"history": [], "stage": 0}

    state = user_states[chat_id]
    user_text = update.message.text
    state["history"].append({"role": "user", "content": user_text})

    # Determine stage
    stage = state["stage"]
    if stage == 0:
        reply = generate_response(user_text, [], state["history"], stage=0)
        state["stage"] = 1
    else:
        results = search(user_text, top_k=3)
        reply = generate_response(user_text, results, state["history"], stage=stage)

    state["history"].append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)

# -------------------------
# Run bot
# -------------------------
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
