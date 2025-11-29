"""
Kuroko Interview - Web App
English Interview Practice for iPhone/Web
"""

import os
import json
import base64
import tempfile
import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Cookie, Response as FastAPIResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Import encryption module
from encryption import (
    encrypt_user_data, decrypt_user_data,
    encrypt_history_data, decrypt_history_data
)

app = FastAPI(title="Kuroko Interview")

# Response settings
RESPONSE_MAX_TOKENS = 80
RESPONSE_RULES = """STRICT RULES:
- 2 sentences max
- No markdown, no asterisks, no bold, no lists
- Plain text only"""

# Session storage (in-memory for simplicity)
sessions = {}

# Data storage
DATA_DIR = Path("/tmp/kuroko_data")
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
USER_TOKENS = {}  # token -> username mapping

# History storage (per user)
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

# RAG storage (per user)
RAG_DIR = DATA_DIR / "rag"
RAG_DIR.mkdir(exist_ok=True)


def load_users() -> dict:
    """Load users from file (with decryption)"""
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            encrypted_users = json.load(f)
        # Decrypt each user's data
        return {username: decrypt_user_data(data) for username, data in encrypted_users.items()}
    return {}


def save_users(users: dict):
    """Save users to file (with encryption)"""
    # Encrypt each user's data before saving
    encrypted_users = {username: encrypt_user_data(data) for username, data in users.items()}
    with open(USERS_FILE, "w") as f:
        json.dump(encrypted_users, f, indent=2)


def hash_password(password: str) -> str:
    """Hash password with salt"""
    return hashlib.sha256(password.encode()).hexdigest()


def get_user_from_token(token: str) -> Optional[str]:
    """Get username from token"""
    return USER_TOKENS.get(token)

def save_session_log(session_id: str, session_data: dict, score: str = None, username: str = None):
    """Save session conversation to file (with encryption)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # User-specific directory
    if username:
        user_history_dir = HISTORY_DIR / username
        user_history_dir.mkdir(exist_ok=True)
        filename = user_history_dir / f"{timestamp}_{session_id[:8]}.json"
    else:
        # Guest - don't save
        return None

    log_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "messages": session_data.get("messages", []),
        "score": score
    }

    # Encrypt sensitive data before saving
    encrypted_data = encrypt_history_data(log_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(encrypted_data, f, ensure_ascii=False, indent=2)

    return filename

def get_history_list(username: str = None) -> List[dict]:
    """Get list of past sessions for a user"""
    if not username:
        return []

    user_history_dir = HISTORY_DIR / username
    if not user_history_dir.exists():
        return []

    sessions_list = []
    for file in sorted(user_history_dir.glob("*.json"), reverse=True)[:20]:
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

def get_session_detail(filename: str, username: str = None) -> dict:
    """Get detailed session data for a user (with decryption)"""
    if not username:
        return None

    filepath = HISTORY_DIR / username / filename
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        encrypted_data = json.load(f)
    # Decrypt sensitive data
    return decrypt_history_data(encrypted_data)

# RAG Engines (per user)
rag_engines = {}

def get_rag_engine(username: str = None):
    """Get RAG engine for user"""
    if not username:
        return None

    if username not in rag_engines:
        from rag import RAGEngine
        index_path = RAG_DIR / username
        rag_engines[username] = RAGEngine(index_name=f"user_{username}")
        # Override index path
        rag_engines[username].index_path = index_path
        if rag_engines[username].load_index():
            print(f"RAG loaded for {username}: {len(rag_engines[username].documents)} chunks")
        else:
            print(f"No RAG index for {username}")

    return rag_engines[username]


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    text: str


class StartRequest(BaseModel):
    session_id: str
    level: str = "L4"  # L4, L5, L6


class ScoreRequest(BaseModel):
    session_id: str


class AuthRequest(BaseModel):
    username: str
    password: str


class SettingsUpdate(BaseModel):
    github_username: Optional[str] = None


def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def get_openai_client():
    from openai import OpenAI
    return OpenAI()


def get_elevenlabs_client():
    from elevenlabs import ElevenLabs
    return ElevenLabs()


def get_level_expectations(level: str) -> dict:
    """Get interview expectations based on FAANG engineering level."""
    levels = {
        "L4": {
            "title": "Mid-Level Engineer (L4/SDE II)",
            "years": "2-5 years",
            "focus": [
                "Feature ownership and autonomous work",
                "Basic system design (polling vs websockets, caching basics)",
                "Problem-solving and debugging skills",
                "Code quality and testing practices"
            ],
            "question_style": "Focus on specific technical implementations, debugging experiences, and feature delivery.",
            "behavioral_focus": "Teamwork, handling technical disagreements, learning from mistakes"
        },
        "L5": {
            "title": "Senior Engineer (L5/SDE III)",
            "years": "5-8 years",
            "focus": [
                "Technical leadership and mentoring",
                "System design for scalability and reliability",
                "Cross-functional collaboration",
                "Driving technical decisions"
            ],
            "question_style": "Emphasize leadership, system design trade-offs, and influencing without authority.",
            "behavioral_focus": "Leading projects, mentoring others, driving technical direction, handling ambiguity"
        },
        "L6": {
            "title": "Staff Engineer (L6/Principal)",
            "years": "8+ years",
            "focus": [
                "Architecture decisions with org-wide impact",
                "Cross-team technical strategy",
                "Complex distributed systems",
                "Building consensus across stakeholders"
            ],
            "question_style": "Deep architecture discussions, organizational influence, long-term technical vision.",
            "behavioral_focus": "Org-wide impact, strategic planning, building technical culture, executive communication"
        }
    }
    return levels.get(level, levels["L4"])


def get_interviewer_prompt(level: str = "L4", context: str = "") -> str:
    level_info = get_level_expectations(level)

    context_section = ""
    if context:
        context_section = f"""
CANDIDATE'S BACKGROUND (from their projects):
{context}

Use this background to ask specific questions about their actual projects and experience.
"""

    focus_points = "\n".join(f"  - {f}" for f in level_info["focus"])

    return f"""You are Kuroko, a technical interviewer at a FAANG company (Google, Amazon, Meta level).
You are interviewing for: {level_info["title"]} ({level_info["years"]} experience)
{context_section}
LEVEL-SPECIFIC EXPECTATIONS:
{focus_points}

QUESTION STYLE FOR THIS LEVEL:
{level_info["question_style"]}

BEHAVIORAL FOCUS:
{level_info["behavioral_focus"]}

FORMAT RULES:
- 2 sentences max
- No markdown, no asterisks, no bold
- Plain text only

INTERVIEW RULES:
- Listen carefully to their answer
- Ask follow-up questions that dig deeper into WHAT THEY JUST SAID
- Match your expectations to the {level} level - don't ask L6 questions to L4 candidates
- For {level}, focus on: {level_info["question_style"]}
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
async def start_interview(req: StartRequest, token: Optional[str] = Cookie(None)):
    """Start a new interview session"""
    try:
        client = get_anthropic_client()
        username = get_user_from_token(token) if token else None

        # Validate level
        level = req.level if req.level in ["L4", "L5", "L6"] else "L4"
        level_info = get_level_expectations(level)

        # Get RAG context about the candidate (user-specific or none for guests)
        context = ""
        if username:
            rag = get_rag_engine(username)
            if rag and rag.index is not None:
                context = rag.get_context("technical projects AI LLM Python development experience", max_tokens=1500)

        system_prompt = get_interviewer_prompt(level=level, context=context)

        # Different first message based on level and whether we have RAG context
        if context:
            first_message = f"Ask one short question about the candidate's background or projects, appropriate for a {level_info['title']} interview."
        else:
            first_message = f"Ask one short technical interview question appropriate for a {level_info['title']} candidate."

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=RESPONSE_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": first_message}]
        )

        first_question = response.content[0].text

        # Store session with username and level
        sessions[req.session_id] = {
            "messages": [{"role": "assistant", "content": first_question}],
            "system_prompt": system_prompt,
            "username": username,
            "level": level
        }

        return {"question": first_question, "level": level}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


GUEST_TURN_LIMIT = 5  # Guest users limited to 5 turns

# RAG generation limits
RAG_MAX_REPOS = 10  # Maximum number of repositories to clone
RAG_MAX_DOCUMENTS = 5000  # Maximum total document chunks
RAG_MAX_PER_REPO = 2000  # Maximum chunks per repository


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Process user answer and get next question"""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[req.session_id]

    # Check for stop command
    if any(word in req.text.lower() for word in ['stop', 'quit', 'exit', 'end', 'bye']):
        return {"action": "stop"}

    # Check guest turn limit
    if not session.get("username"):
        # Count user messages (turns)
        user_turns = len([m for m in session["messages"] if m["role"] == "user"])
        if user_turns >= GUEST_TURN_LIMIT:
            return {"action": "guest_limit", "message": "Guest mode is limited to 5 turns. Please login to continue."}

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
    level = session.get("level", "L4")
    level_info = get_level_expectations(level)

    if not messages:
        return {"score": "No conversation to score."}

    # Build transcript
    transcript = ""
    for m in messages:
        role = "Interviewer" if m["role"] == "assistant" else "Candidate"
        transcript += f"{role}: {m['content']}\n"

    try:
        client = get_anthropic_client()

        scoring_prompt = f"""You are an interview coach evaluating a mock technical interview.
The candidate was interviewing for: {level_info['title']} ({level_info['years']} experience)

Score the candidate on these 5 criteria (1-5 scale), evaluating against {level} level expectations:

1. Clarity - How clearly did they explain their ideas?
2. Technical Depth - Did they demonstrate {level}-appropriate technical understanding?
3. Communication - Did they answer questions directly and professionally?
4. Fluency - How smoothly did they speak in English?
5. {"Leadership & Influence" if level in ["L5", "L6"] else "Problem Solving"} - {"Did they show leadership qualities and ability to influence?" if level in ["L5", "L6"] else "Did they show good problem-solving approach?"}

FORMAT (plain text, no markdown):
Interview Level: {level} ({level_info['title']})

Clarity: X/5 - [one short comment]
Technical Depth: X/5 - [one short comment]
Communication: X/5 - [one short comment]
Fluency: X/5 - [one short comment]
{"Leadership: X/5" if level in ["L5", "L6"] else "Problem Solving: X/5"} - [one short comment]

Total: XX/25

{"Ready for " + level + "? [Yes/Almost/Need more practice]" if level != "L4" else ""}
One advice: [single actionable tip for improvement targeting {level} level interviews]
"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=scoring_prompt,
            messages=[{"role": "user", "content": f"Interview transcript:\n{transcript}"}]
        )

        score_text = response.content[0].text

        # Save session log before cleanup (only for logged-in users)
        username = session.get("username")
        save_session_log(req.session_id, session, score_text, username)

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
async def list_history(token: Optional[str] = Cookie(None)):
    """Get list of past interview sessions"""
    username = get_user_from_token(token) if token else None
    return {"sessions": get_history_list(username)}


@app.get("/api/history/{filename}")
async def get_history(filename: str, token: Optional[str] = Cookie(None)):
    """Get details of a specific session"""
    username = get_user_from_token(token) if token else None
    data = get_session_detail(filename, username)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


@app.delete("/api/history/{filename}")
async def delete_history(filename: str, token: Optional[str] = Cookie(None)):
    """Delete a specific session"""
    username = get_user_from_token(token) if token else None
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")

    filepath = HISTORY_DIR / username / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    filepath.unlink()
    return {"status": "deleted"}


# ==================== Authentication API ====================

@app.post("/api/auth/register")
async def register(req: AuthRequest, response: FastAPIResponse):
    """Register a new user"""
    users = load_users()

    if req.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")

    users[req.username] = {
        "password": hash_password(req.password),
        "github_username": None,
        "created_at": datetime.now().isoformat()
    }
    save_users(users)

    # Auto login
    token = secrets.token_urlsafe(32)
    USER_TOKENS[token] = req.username
    response.set_cookie(key="token", value=token, httponly=True, max_age=86400*30)

    return {"status": "registered", "username": req.username}


@app.post("/api/auth/login")
async def login(req: AuthRequest, response: FastAPIResponse):
    """Login user"""
    users = load_users()

    if req.username not in users:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if users[req.username]["password"] != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_urlsafe(32)
    USER_TOKENS[token] = req.username
    response.set_cookie(key="token", value=token, httponly=True, max_age=86400*30)

    return {"status": "logged_in", "username": req.username}


@app.post("/api/auth/logout")
async def logout(response: FastAPIResponse, token: Optional[str] = Cookie(None)):
    """Logout user"""
    if token and token in USER_TOKENS:
        del USER_TOKENS[token]
    response.delete_cookie(key="token")
    return {"status": "logged_out"}


@app.get("/api/auth/me")
async def get_current_user(token: Optional[str] = Cookie(None)):
    """Get current user info"""
    if not token:
        return {"logged_in": False}

    username = get_user_from_token(token)
    if not username:
        return {"logged_in": False}

    users = load_users()
    user_data = users.get(username, {})

    return {
        "logged_in": True,
        "username": username,
        "github_username": user_data.get("github_username"),
        "has_rag": (RAG_DIR / username / "index.faiss").exists()
    }


# ==================== Settings API ====================

@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate, token: Optional[str] = Cookie(None)):
    """Update user settings"""
    username = get_user_from_token(token) if token else None
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")

    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")

    if settings.github_username is not None:
        users[username]["github_username"] = settings.github_username

    save_users(users)
    return {"status": "updated"}


@app.get("/api/settings/generate-rag")
async def generate_rag_stream(token: Optional[str] = Cookie(None)):
    """Generate RAG index from user's GitHub repos with progress streaming"""
    import subprocess
    import shutil
    import urllib.request
    from fastapi.responses import StreamingResponse

    username = get_user_from_token(token) if token else None
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")

    users = load_users()
    github_username = users.get(username, {}).get("github_username")
    if not github_username:
        raise HTTPException(status_code=400, detail="GitHub username not set")

    async def generate():
        try:
            yield f"data: {json.dumps({'step': 'Fetching repo list...', 'progress': 5})}\n\n"

            # Create temp dir for cloning
            temp_dir = Path(tempfile.mkdtemp())

            # Get public repos via GitHub API (fetch more to sort by size)
            api_url = f"https://api.github.com/users/{github_username}/repos?type=public&per_page=100"
            req = urllib.request.Request(api_url, headers={"User-Agent": "Kuroko-Interview"})
            with urllib.request.urlopen(req, timeout=30) as response:
                all_repos = json.loads(response.read().decode())

            if not all_repos:
                yield f"data: {json.dumps({'error': 'No public repos found'})}\n\n"
                return

            # Sort by size (smaller first) and limit
            all_repos.sort(key=lambda r: r.get('size', 0))
            repos = all_repos[:RAG_MAX_REPOS]

            if len(all_repos) > RAG_MAX_REPOS:
                yield f"data: {json.dumps({'step': f'Found {len(all_repos)} repos, limiting to {RAG_MAX_REPOS}', 'progress': 10})}\n\n"
            else:
                yield f"data: {json.dumps({'step': f'Found {len(repos)} repos', 'progress': 10})}\n\n"

            # Clone repos using git
            total_repos = len(repos)
            for i, repo in enumerate(repos):
                repo_name = repo["name"]
                clone_url = repo["clone_url"]
                progress = 10 + int((i / total_repos) * 50)
                yield f"data: {json.dumps({'step': f'Cloning {repo_name}...', 'progress': progress})}\n\n"

                subprocess.run(
                    ["/usr/bin/git", "clone", "--depth", "1", clone_url],
                    cwd=temp_dir, capture_output=True
                )

            yield f"data: {json.dumps({'step': 'Building RAG index...', 'progress': 65})}\n\n"

            # Generate RAG index with limits
            from rag import RAGEngine
            user_rag_dir = RAG_DIR / username
            user_rag_dir.mkdir(exist_ok=True)

            rag = RAGEngine(index_name=f"user_{username}")
            rag.index_path = user_rag_dir

            yield f"data: {json.dumps({'step': 'Indexing .md files...', 'progress': 75})}\n\n"

            # Index with document limits
            rag.index_directory(
                str(temp_dir),
                extensions=[".md"],
                max_documents=RAG_MAX_DOCUMENTS,
                max_per_source=RAG_MAX_PER_REPO
            )

            yield f"data: {json.dumps({'step': 'Saving index...', 'progress': 90})}\n\n"

            # Cleanup temp dir
            shutil.rmtree(temp_dir)

            # Update cache
            rag_engines[username] = rag

            doc_count = len(rag.documents)
            msg = f'Done! {doc_count} chunks indexed'
            if doc_count >= RAG_MAX_DOCUMENTS:
                msg += f' (limit: {RAG_MAX_DOCUMENTS})'
            yield f"data: {json.dumps({'step': msg, 'progress': 100, 'chunks': doc_count})}\n\n"

        except Exception as e:
            import traceback
            print(f"RAG generation error: {traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
