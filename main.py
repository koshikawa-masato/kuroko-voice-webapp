"""
Kuroko Interview - Web App
English Interview Practice for iPhone/Web
"""

import os
import json
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Kuroko Interview")

# Response settings
RESPONSE_MAX_TOKENS = 80
RESPONSE_RULES = """STRICT RULES:
- 2 sentences max
- No markdown, no asterisks, no bold, no lists
- Plain text only"""

# Session storage (in-memory for simplicity)
sessions = {}

# History storage
HISTORY_DIR = Path("/tmp/kuroko_history")
HISTORY_DIR.mkdir(exist_ok=True)

def save_session_log(session_id: str, session_data: dict, score: str = None):
    """Save session conversation to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = HISTORY_DIR / f"{timestamp}_{session_id[:8]}.json"

    log_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "messages": session_data.get("messages", []),
        "score": score
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    return filename

def get_history_list() -> List[dict]:
    """Get list of past sessions"""
    sessions_list = []
    for file in sorted(HISTORY_DIR.glob("*.json"), reverse=True)[:20]:  # Last 20 sessions
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions_list.append({
                    "filename": file.name,
                    "timestamp": data.get("timestamp"),
                    "message_count": len(data.get("messages", [])),
                    "has_score": data.get("score") is not None
                })
        except:
            pass
    return sessions_list

def get_session_detail(filename: str) -> dict:
    """Get detailed session data"""
    filepath = HISTORY_DIR / filename
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# RAG Engine
rag_engine = None

def get_rag_engine():
    """Lazy-load RAG engine"""
    global rag_engine
    if rag_engine is None:
        from rag import RAGEngine
        rag_engine = RAGEngine(index_name="interview")
        if rag_engine.load_index():
            print(f"RAG loaded: {len(rag_engine.documents)} chunks")
        else:
            print("RAG index not found")
    return rag_engine


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    text: str


class StartRequest(BaseModel):
    session_id: str


class ScoreRequest(BaseModel):
    session_id: str


def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def get_openai_client():
    from openai import OpenAI
    return OpenAI()


def get_elevenlabs_client():
    from elevenlabs import ElevenLabs
    return ElevenLabs()


def get_interviewer_prompt(level: str = "intermediate", context: str = "") -> str:
    context_section = ""
    if context:
        context_section = f"""
CANDIDATE'S BACKGROUND (from their projects):
{context}

Use this background to ask specific questions about their actual projects and experience.
"""

    return f"""You are Kuroko, a technical interviewer at a global tech company.
User level: {level.upper()}
{context_section}
FORMAT RULES:
- 2 sentences max
- No markdown, no asterisks, no bold
- Plain text only

INTERVIEW RULES:
- Listen carefully to their answer
- Ask follow-up questions that dig deeper into WHAT THEY JUST SAID
- Reference their actual projects when relevant
- Examples of good follow-ups:
  - They mention Python -> Ask what libraries or frameworks
  - They mention a project -> Ask about challenges they faced
  - They mention a decision -> Ask why they chose that approach
- Do NOT repeat similar questions
- Do NOT ask generic questions if they gave specific details
- Build on their previous answers

If answer is too short, say: Please tell me more.
"""


@app.post("/api/start")
async def start_interview(req: StartRequest):
    """Start a new interview session"""
    try:
        client = get_anthropic_client()

        # Get RAG context about the candidate
        rag = get_rag_engine()
        context = ""
        if rag and rag.index is not None:
            context = rag.get_context("technical projects AI LLM Python development experience", max_tokens=1500)

        system_prompt = get_interviewer_prompt(context=context)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=RESPONSE_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": "Ask one short question about the candidate's background or projects."}]
        )

        first_question = response.content[0].text

        # Store session
        sessions[req.session_id] = {
            "messages": [{"role": "assistant", "content": first_question}],
            "system_prompt": system_prompt
        }

        return {"question": first_question}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Process user answer and get next question"""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[req.session_id]

    # Check for stop command
    if any(word in req.text.lower() for word in ['stop', 'quit', 'exit', 'end', 'bye']):
        return {"action": "stop"}

    try:
        client = get_anthropic_client()

        # Add user message
        session["messages"].append({"role": "user", "content": req.text})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=RESPONSE_MAX_TOKENS,
            system=session["system_prompt"],
            messages=session["messages"]
        )

        answer = response.content[0].text
        session["messages"].append({"role": "assistant", "content": answer})

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/score")
async def score_interview(req: ScoreRequest):
    """Score the interview session"""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[req.session_id]
    messages = session["messages"]

    if not messages:
        return {"score": "No conversation to score."}

    # Build transcript
    transcript = ""
    for m in messages:
        role = "Interviewer" if m["role"] == "assistant" else "Candidate"
        transcript += f"{role}: {m['content']}\n"

    try:
        client = get_anthropic_client()

        scoring_prompt = """You are an interview coach evaluating a mock technical interview.

Score the candidate on these 5 criteria (1-5 scale):

1. Clarity - How clearly did they explain their ideas?
2. Technical Depth - Did they show technical understanding?
3. Communication - Did they answer questions directly?
4. Fluency - How smoothly did they speak in English?
5. Confidence - Did they seem confident in their answers?

FORMAT (plain text, no markdown):
Clarity: X/5 - [one short comment]
Technical Depth: X/5 - [one short comment]
Communication: X/5 - [one short comment]
Fluency: X/5 - [one short comment]
Confidence: X/5 - [one short comment]

Total: XX/25

One advice: [single actionable tip for improvement]
"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=scoring_prompt,
            messages=[{"role": "user", "content": f"Interview transcript:\n{transcript}"}]
        )

        score_text = response.content[0].text

        # Save session log before cleanup
        save_session_log(req.session_id, session, score_text)

        # Clean up session
        del sessions[req.session_id]

        return {"score": score_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts")
async def text_to_speech(text: str):
    """Convert text to speech using ElevenLabs"""
    try:
        client = get_elevenlabs_client()
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

        from elevenlabs import VoiceSettings
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                speed=0.7
            )
        )

        audio_bytes = b"".join(audio_generator)

        return Response(content=audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text using Whisper"""
    try:
        client = get_openai_client()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            content = await audio.read()
            f.write(content)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            return {"text": result.text}
        finally:
            os.unlink(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def list_history():
    """Get list of past interview sessions"""
    return {"sessions": get_history_list()}


@app.get("/api/history/{filename}")
async def get_history(filename: str):
    """Get details of a specific session"""
    data = get_session_detail(filename)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


@app.delete("/api/history/{filename}")
async def delete_history(filename: str):
    """Delete a specific session"""
    filepath = HISTORY_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    filepath.unlink()
    return {"status": "deleted"}


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
