import os
import json
import random
import pdfplumber
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import httpx
from gtts import gTTS
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
import uuid
import tempfile
import uvicorn
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from weasyprint import HTML, CSS
from io import BytesIO
import re
import os
import smtplib
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
# ATS functionality imports
import PyPDF2 as pdf
import google.generativeai as genai

os.environ["GLOG_minloglevel"] = "2"  # Hide MediaPipe startup dump

# Import the proctoring service
from proctoring_service import ProctoringService

# FIXED: Track which sessions have been emailed to prevent duplicates
email_sent_for_session = set()

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
load_dotenv()

# Check email configuration on startup
def check_email_configuration():
    """Check and log email configuration status on startup"""
    if EMAIL_USERNAME and EMAIL_PASSWORD:
        logger.info(f"✅ EMAIL CONFIGURED: {EMAIL_USERNAME} via {EMAIL_SMTP_SERVER}:{EMAIL_SMTP_PORT}")
        logger.info("✅ Automatic interview report emails will be sent to candidates")
    else:
        logger.warning("⚠️ EMAIL NOT CONFIGURED: Automatic report sending is DISABLED")
        logger.warning("💡 To enable email functionality:")
        logger.warning("   1. Edit the .env file")
        logger.warning("   2. Configure EMAIL_USERNAME and EMAIL_PASSWORD")
        logger.warning("   3. See EMAIL_README.md for detailed instructions")
        logger.warning("   4. Visit /email_setup_guide for step-by-step guidance")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set default JSON response charset
@app.middleware("http")
async def add_charset_middleware(request: Request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith(("application/json", "text/html")):
        response.headers["content-type"] += "; charset=utf-8"
    return response

# Custom WebSocket class to ensure proper encoding
class EncodedWebSocket(WebSocket):
    async def send_text(self, data: str):
        # Ensure proper UTF-8 encoding and handle special characters
        if isinstance(data, str):
            data = data.encode('utf-8').decode('utf-8')
        await super().send_text(data)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

AUDIO_FOLDER = "audio_files"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"

# Email Configuration
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "AI Interview System")
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"

# ATS Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("✅ Google Gemini configured for ATS functionality")
else:
    logger.warning("⚠️ GOOGLE_API_KEY not set - ATS functionality will be disabled")

# Check email configuration on startup
check_email_configuration()

# Create a thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize proctoring service
proctoring_service = ProctoringService()

# --- PDF Parsing (pdfplumber) ---
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        logger.info(f"pdfplumber extraction complete. Total text length: {len(text)}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error during pdfplumber text extraction: {e}")
        return ""

async def query_gemini_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Query Gemini with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            return await query_gemini(prompt)
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limited. Retrying in {wait_time:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error("Max retries exceeded for API call")
                    # Return fallback instead of crashing
                    return "Unable to generate response due to API rate limits. Please try again later."
            else:
                raise
    return ""

# --- Resume Summarization Function ---
async def summarize_resume(resume_text: str) -> str:
    """Generate a comprehensive summary of the resume once to reuse in all prompts."""
    prompt = f"""
You are a smart AI assistant. The following is a candidate's resume.

Your job is to extract and summarize all key elements: education, projects, internships, certifications, skills, achievements, tools, and any other relevant details.

Present them in a clean and complete paragraph format. This will be reused throughout the interview.

Resume:
--------
{resume_text}
"""
    try:
        summary = (await query_gemini_with_retry(prompt)).strip()
        logger.info("Resume summarized successfully.")
        return summary
    except Exception as e:
        logger.error(f"Failed to summarize resume: {e}")
        return resume_text  # fallback to raw

# --- State management ---
# ADDED: User details model for face capture
class UserDetails(BaseModel):
    name: str = ""
    phone: str = ""
    email: str = ""

class InterviewState(BaseModel):
    user_details: UserDetails = UserDetails()  # ADDED
    resume_text: str = ""
    resume_summary: str = ""  # NEW: Store resume summary instead of full text
    current_dialogue: List[Dict] = []
    is_interview_active: bool = False
    current_question_count: int = 0
    max_questions: int = 7
    min_questions: int = 5
    last_question: str = ""
    consent_received: bool = False
    answer_evaluations: List[Dict] = []  # Store AI evaluations silently
    total_score: float = 0.0
    average_score: float = 0.0
    proctoring_session_id: str = ""  # Track proctoring session
    proctoring_violations: List[Dict] = []  # Track violations
    reference_face_captured: bool = False  # ADDED

class InterviewKPIs(BaseModel):
    communication_score: float = 0.0
    technical_score: float = 0.0
    problem_solving_score: float = 0.0
    confidence_score: float = 0.0
    clarity_score: float = 0.0
    response_time_avg: float = 0.0
    questions_answered: int = 0
    completion_rate: float = 0.0
    engagement_level: str = "Medium"
    strengths_count: int = 0
    improvement_areas_count: int = 0

class InterviewReport(BaseModel):
    candidate_name: str = "Candidate"
    candidate_phone: str = ""  # ADDED
    candidate_email: str = ""  # ADDED
    interview_date: str = ""
    overall_score: float = 0.0
    technical_score: float = 0.0
    communication_score: float = 0.0
    experience_score: float = 0.0
    problem_solving_score: float = 0.0
    detailed_feedback: str = ""
    resume_quality: str = ""
    technical_skills: str = ""
    communication_skills: str = ""
    strengths: List[str] = []
    areas_for_improvement: List[str] = []
    recommendations: str = ""
    interview_transcript: List[Dict] = []
    kpis: InterviewKPIs = InterviewKPIs()
    proctoring_violations: List[Dict] = []  # Add proctoring violations

# ATS Models
class ATSRequest(BaseModel):
    job_description: str
    resume_file: Optional[str] = None

class ATSResponse(BaseModel):
    jd_match: str
    missing_keywords: List[str]
    profile_summary: str
    success: bool = True
    error_message: Optional[str] = None

# --- ATS Functions ---
def extract_text_from_pdf_ats(uploaded_file) -> str:
    """Extract text from PDF using PyPDF2 for ATS functionality"""
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page_text = reader.pages[page].extract_text()
        if page_text:
            text += page_text
    return text

async def get_gemini_ats_response(input_text: str) -> str:
    """Call Gemini for ATS analysis using Google AI SDK"""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=400, detail="Google API key not configured")
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Configure generation settings for better JSON output
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Low temperature for consistent output
            max_output_tokens=1000,  # Reasonable limit
            top_p=0.8,
            top_k=10
        )
        
        response = model.generate_content(
            input_text,
            generation_config=generation_config
        )
        
        if not response.text:
            logger.error("Gemini returned empty response")
            raise Exception("Empty response from Gemini")
            
        return response.text
        
    except Exception as e:
        logger.error(f"Error calling Gemini for ATS: {e}")
        # Return a fallback JSON response instead of raising exception
        return '''{
  "JD Match": "Unable to analyze",
  "MissingKeywords": ["Analysis failed"],
  "Profile Summary": "Unable to complete analysis due to API error"
}'''

async def analyze_resume_ats(resume_text: str, job_description: str) -> ATSResponse:
    """Analyze resume against job description using ATS"""
    input_prompt = f"""You are an expert ATS (Application Tracking System) with deep knowledge of hiring practices.

Task: Compare this resume against the job description and provide analysis.

RESUME:
{resume_text[:3000]}  # Limit resume length

JOB DESCRIPTION:
{job_description[:2000]}  # Limit JD length

Instructions for analysis:
1. CAREFULLY read both the resume and job description
2. Look for skills mentioned in DIFFERENT WAYS (e.g., "JavaScript" = "JS", "Machine Learning" = "ML")
3. Consider RELATED SKILLS as matches (e.g., "React" covers "Frontend Development")
4. Only mark skills as "missing" if they are TRULY not present or implied
5. Be GENEROUS in matching - if a skill is present in any form, don't mark it as missing
6. Focus on MAJOR missing skills, not minor variations

Provide your analysis in this EXACT JSON format (no additional text):
{{
  "JD Match": "XX%",
  "MissingKeywords": ["only major missing skills here"],
  "Profile Summary": "Brief 2-3 sentence summary highlighting candidate's strengths and fit for this role"
}}

IMPORTANT: 
- Return ONLY the JSON object, no other text
- Be conservative with "MissingKeywords" - only include truly absent skills
- Give higher match percentages when skills are present but phrased differently"""
    
    try:
        # Get response from Gemini
        response = await get_gemini_ats_response(input_prompt)
        
        # Clean up the response
        response = response.strip()
        logger.info(f"Raw Gemini response: {response[:200]}...")  # Log first 200 chars
        
        # Try to extract JSON from response if it has extra text
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = response
        
        # Parse JSON response
        parsed_response = json.loads(json_str)
        
        return ATSResponse(
            jd_match=parsed_response.get("JD Match", "N/A"),
            missing_keywords=parsed_response.get("MissingKeywords", []),
            profile_summary=parsed_response.get("Profile Summary", "No summary available"),
            success=True
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ATS response as JSON: {e}")
        logger.error(f"Raw response was: {response if 'response' in locals() else 'No response received'}")
        
        # Fallback analysis if JSON parsing fails
        return ATSResponse(
            jd_match="75%",  # Default reasonable score
            missing_keywords=["Unable to parse keywords"],
            profile_summary=f"Analysis completed but response format was invalid. Raw response: {response[:100] if 'response' in locals() else 'No response'}...",
            success=False,
            error_message="Invalid JSON response from AI model"
        )
    except Exception as e:
        logger.error(f"ATS analysis failed: {e}")
        return ATSResponse(
            jd_match="N/A",
            missing_keywords=[],
            profile_summary="Analysis failed due to system error",
            success=False,
            error_message=str(e)
        )

interview_state = InterviewState()
logger.info("InterviewState initialized.")

# Global variable to store generated reports
generated_reports = {}

# ADDED: Simple storage for user details (for production, use a proper database)
user_sessions = {}

# --- WebSocket manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.proctoring_connections: List[WebSocket] = []
        logger.info("ConnectionManager initialized.")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket.client}")

    async def connect_proctoring(self, websocket: WebSocket):
        await websocket.accept()
        self.proctoring_connections.append(websocket)
        logger.info(f"Proctoring WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {websocket.client}")

    def disconnect_proctoring(self, websocket: WebSocket):
        if websocket in self.proctoring_connections:
            self.proctoring_connections.remove(websocket)
        logger.info(f"Proctoring WebSocket disconnected: {websocket.client}")

manager = ConnectionManager()

# --- Async Audio Generation ---
async def generate_audio_async(text: str, lang: str = 'en') -> str:
    """Generate audio asynchronously using thread pool"""
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)

    def _generate_audio():
        for attempt in range(3):
            try:
                tts = gTTS(text=text, lang=lang)
                tts.save(filepath)
                logger.info(f"Audio generated successfully: {filename}")
                return filename
            except Exception as e:
                logger.warning(f"[gTTS] Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    logger.error("gTTS failed after 3 retries.")
                    raise Exception("gTTS failed after 3 retries")
                time.sleep(2)
        # This should never be reached due to the exception raised above
        raise Exception("Audio generation failed")

    # Run gTTS in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    filename = await loop.run_in_executor(executor, _generate_audio)
    return filename

# Legacy function for backward compatibility
def generate_audio(text: str, lang: str = 'en') -> str:
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    for attempt in range(3):
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(filepath)
            logger.info(f"Audio generated successfully: {filename}")
            return filename
        except Exception as e:
            logger.warning(f"[gTTS] Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    logger.error("gTTS failed after 3 retries.")
    raise Exception("gTTS failed after 3 retries")

async def query_gemini(prompt: str) -> str:
    logger.info(f"Querying Gemini (via OpenRouter) with prompt: {prompt[:500]}...")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional AI interviewer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    timeout = httpx.Timeout(20.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            logger.info("Gemini query successful.")
            return content
    except httpx.RequestError as e:
        logger.error(f"Error querying Gemini API (OpenRouter): {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query_gemini: {e}")
        raise

async def generate_introductory_question(resume_summary: str) -> str:
    """Always start with this fixed question after consent."""
    return "Tell me about yourself."

# Global state for tracking
current_topic = None
topic_question_count = 0
covered_topics = set()

async def generate_dynamic_question(resume_summary: str, last_response: str, dialogue: List[Dict]) -> str:
    global current_topic, topic_question_count, covered_topics

    # Get recent conversation context (last 4 exchanges)
    recent_context = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in dialogue[-4:]
    )

    prompt = f"""Candidate Summary (from resume):
--------
{resume_summary}

Recent conversation:
{recent_context}

Last answer: {last_response}

Topics already covered: {', '.join(covered_topics) if covered_topics else 'None'}

Generate ONE interview question following these rules:

1. Focus on: projects, internships, certifications, technical skills, tools used
2. If last answer was detailed and complete → move to NEW topic from resume
3. If last answer was brief/incomplete → ask ONE follow-up, then move on
4. Don't repeat covered topics: {covered_topics}
5. Prioritize: projects > internships > certifications > skills
6. Ask specific technical questions about implementation, challenges, results

Return ONLY the question. No explanations."""

    try:
        question = (await query_gemini_with_retry(prompt)).strip()

        # Simple topic management
        followup_keywords = ["how", "what", "can you elaborate", "tell me more", "explain"]
        is_followup = any(keyword in question.lower()[:20] for keyword in followup_keywords)

        if is_followup and current_topic:
            topic_question_count += 1
            if topic_question_count >= 2:  # Max 2 questions per topic
                covered_topics.add(current_topic)
                current_topic = None
                topic_question_count = 0
        else:
            # New topic detected
            if current_topic:
                covered_topics.add(current_topic)
            current_topic = extract_topic_from_question(question)
            topic_question_count = 1

        return question

    except Exception as e:
        logger.error(f"Error generating question: {e}")
        return get_fallback_question()

def extract_topic_from_question(question: str) -> str:
    """Extract main topic from question for tracking"""
    question_lower = question.lower()

    # Topic keywords mapping
    topics = {
        'project': ['project', 'application', 'system', 'built', 'developed'],
        'internship': ['internship', 'intern', 'company', 'workplace'],
        'certification': ['certification', 'certified', 'course', 'training'],
        'skills': ['technology', 'language', 'framework', 'tool', 'skill']
    }

    for topic, keywords in topics.items():
        if any(keyword in question_lower for keyword in keywords):
            return topic
    return 'general'

def get_fallback_question() -> str:
    """Simple fallback questions covering main resume areas"""
    fallbacks = [
        "What's the most challenging project you've worked on?",
        "Tell me about your internship experience and key learnings.",
        "Which certification or course has been most valuable to you?",
        "What technologies are you most comfortable working with?"
    ]

    return fallbacks[len(covered_topics) % len(fallbacks)]

# Reset function for new interviews
def reset_interview_state():
    global current_topic, topic_question_count, covered_topics
    current_topic = None
    topic_question_count = 0
    covered_topics = set()

async def evaluate_user_answer(question: str, user_answer: str, resume_summary: str) -> Dict:
    """Evaluate user's answer in real-time and provide detailed feedback."""
    prompt = f"""
You are an expert interview evaluator. Analyze this candidate's response to an interview question.

QUESTION ASKED:
{question}

CANDIDATE'S ANSWER:
{user_answer}

CANDIDATE'S RESUME CONTEXT:
{resume_summary}

Provide evaluation in this EXACT format:

SCORE: [0-10]
TECHNICAL_ACCURACY: [Rate technical correctness 0-10]
COMMUNICATION_CLARITY: [Rate clarity and articulation 0-10]
RELEVANCE: [Rate how well answer addresses question 0-10]
DEPTH: [Rate depth of explanation 0-10]
CONFIDENCE: [Rate confidence level 0-10]
PROBLEM_SOLVING: [Rate problem-solving approach 0-10]
STRENGTHS: [List 2-3 key strengths in the answer]
WEAKNESSES: [List 2-3 areas for improvement]
FEEDBACK: [Constructive feedback in 2-3 sentences]
FOLLOW_UP_SUGGESTION: [Suggest what interviewer should ask next based on this answer]

Evaluation Criteria:
- Technical Accuracy (20%): Correctness of technical information
- Communication Clarity (20%): Clear articulation and professional presentation
- Relevance (20%): How well the answer addresses the specific question
- Depth (20%): Level of detail and insight provided
- Confidence (10%): Confidence in delivery and knowledge
- Problem-Solving (10%): Approach to solving problems

Provide specific, actionable feedback that helps the candidate improve.
"""

    try:
        evaluation = await query_gemini_with_retry(prompt)

        # Extract evaluation components using regex
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', evaluation, re.IGNORECASE)
        tech_match = re.search(r'TECHNICAL_ACCURACY:\s*(\d+(?:\.\d+)?)', evaluation, re.IGNORECASE)
        comm_match = re.search(r'COMMUNICATION_CLARITY:\s*(\d+(?:\.\d+)?)', evaluation, re.IGNORECASE)
        rel_match = re.search(r'RELEVANCE:\s*(\d+(?:\.\d+)?)', evaluation, re.IGNORECASE)
        depth_match = re.search(r'DEPTH:\s*(\d+(?:\.\d+)?)', evaluation, re.IGNORECASE)
        conf_match = re.search(r'CONFIDENCE:\s*(\d+(?:\.\d+)?)', evaluation, re.IGNORECASE)
        prob_match = re.search(r'PROBLEM_SOLVING:\s*(\d+(?:\.\d+)?)', evaluation, re.IGNORECASE)

        # Extract scores with validation
        overall_score = float(score_match.group(1)) if score_match else 5.0
        technical_score = float(tech_match.group(1)) if tech_match else 5.0
        communication_score = float(comm_match.group(1)) if comm_match else 5.0
        relevance_score = float(rel_match.group(1)) if rel_match else 5.0
        depth_score = float(depth_match.group(1)) if depth_match else 5.0
        confidence_score = float(conf_match.group(1)) if conf_match else 5.0
        problem_solving_score = float(prob_match.group(1)) if prob_match else 5.0

        # Extract text sections
        strengths = extract_section(evaluation, 'STRENGTHS', 'WEAKNESSES')
        weaknesses = extract_section(evaluation, 'WEAKNESSES', 'FEEDBACK')
        feedback = extract_section(evaluation, 'FEEDBACK', 'FOLLOW_UP_SUGGESTION')
        follow_up_suggestion = extract_section(evaluation, 'FOLLOW_UP_SUGGESTION', None)

        return {
            "overall_score": min(10.0, max(0.0, overall_score)),
            "technical_accuracy": min(10.0, max(0.0, technical_score)),
            "communication_clarity": min(10.0, max(0.0, communication_score)),
            "relevance": min(10.0, max(0.0, relevance_score)),
            "depth": min(10.0, max(0.0, depth_score)),
            "confidence": min(10.0, max(0.0, confidence_score)),
            "problem_solving": min(10.0, max(0.0, problem_solving_score)),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "feedback": feedback,
            "follow_up_suggestion": follow_up_suggestion,
            "raw_evaluation": evaluation
        }

    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        return {
            "overall_score": 5.0,
            "technical_accuracy": 5.0,
            "communication_clarity": 5.0,
            "relevance": 5.0,
            "depth": 5.0,
            "confidence": 5.0,
            "problem_solving": 5.0,
            "strengths": "Response provided",
            "weaknesses": "Evaluation temporarily unavailable",
            "feedback": "Please continue with the interview",
            "follow_up_suggestion": "Ask a follow-up question based on the response",
            "raw_evaluation": "Evaluation failed"
        }

def extract_section(text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
    """Extract text section between markers."""
    start_pattern = rf'{re.escape(start_marker)}:\s*'
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    
    if not start_match:
        return "Not available"
    
    start_pos = start_match.end()
    
    if end_marker:
        end_pattern = rf'{re.escape(end_marker)}:'
        end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE)
        if end_match:
            return text[start_pos:start_pos + end_match.start()].strip()
    
    return text[start_pos:].strip()

def calculate_kpis(evaluations: List[Dict]) -> InterviewKPIs:
    """Calculate KPIs from evaluation data."""
    if not evaluations:
        return InterviewKPIs()

    # Calculate average scores
    communication_scores = [eval_data["evaluation"]["communication_clarity"] for eval_data in evaluations]
    technical_scores = [eval_data["evaluation"]["technical_accuracy"] for eval_data in evaluations]
    problem_solving_scores = [eval_data["evaluation"].get("problem_solving", 5.0) for eval_data in evaluations]
    confidence_scores = [eval_data["evaluation"].get("confidence", 5.0) for eval_data in evaluations]

    # Calculate averages
    communication_avg = sum(communication_scores) / len(communication_scores)
    technical_avg = sum(technical_scores) / len(technical_scores)
    problem_solving_avg = sum(problem_solving_scores) / len(problem_solving_scores)
    confidence_avg = sum(confidence_scores) / len(confidence_scores)

    # Calculate completion rate
    expected_questions = interview_state.max_questions
    actual_questions = len(evaluations)
    completion_rate = (actual_questions / expected_questions) * 100

    # Determine engagement level
    overall_avg = (communication_avg + technical_avg + problem_solving_avg + confidence_avg) / 4
    if overall_avg >= 8:
        engagement_level = "High"
    elif overall_avg >= 6:
        engagement_level = "Medium"
    else:
        engagement_level = "Low"

    # Count strengths and improvement areas
    strengths_count = sum(1 for eval_data in evaluations if eval_data["evaluation"]["strengths"] != "Not available")
    improvement_areas_count = sum(1 for eval_data in evaluations if eval_data["evaluation"]["weaknesses"] != "Not available")

    return InterviewKPIs(
        communication_score=communication_avg,
        technical_score=technical_avg,
        problem_solving_score=problem_solving_avg,
        confidence_score=confidence_avg,
        clarity_score=communication_avg,  # Using communication as clarity proxy
        response_time_avg=0.0,  # Would need timing data
        questions_answered=actual_questions,
        completion_rate=completion_rate,
        engagement_level=engagement_level,
        strengths_count=strengths_count,
        improvement_areas_count=improvement_areas_count
    )

async def generate_interview_report(dialogue: List[Dict], resume_summary: str, evaluations: List[Dict]) -> InterviewReport:
    """Generate comprehensive report including AI evaluations and KPIs."""
    # Filter valid dialogue entries
    valid_dialogue = [
        msg for msg in dialogue
        if msg.get("content") and msg["content"] != "[No response]" and len(msg["content"].strip()) > 3
    ]

    # Calculate KPIs
    kpis = calculate_kpis(evaluations)

    # Calculate evaluation statistics
    if evaluations:
        overall_scores = [eval_data["evaluation"]["overall_score"] for eval_data in evaluations]
        technical_scores = [eval_data["evaluation"]["technical_accuracy"] for eval_data in evaluations]
        communication_scores = [eval_data["evaluation"]["communication_clarity"] for eval_data in evaluations]

        avg_overall = sum(overall_scores) / len(overall_scores)
        avg_technical = sum(technical_scores) / len(technical_scores)
        avg_communication = sum(communication_scores) / len(communication_scores)

        # Create detailed evaluation summary
        detailed_evaluations = "\n\n".join([
            f"Q{i+1}: {eval_data['question']}\n"
            f"Answer: {eval_data['answer'][:200]}{'...' if len(eval_data['answer']) > 200 else ''}\n"
            f"Score: {eval_data['evaluation']['overall_score']}/10\n"
            f"Communication: {eval_data['evaluation']['communication_clarity']}/10\n"
            f"Technical: {eval_data['evaluation']['technical_accuracy']}/10\n"
            f"Confidence: {eval_data['evaluation'].get('confidence', 5.0)}/10\n"
            f"Feedback: {eval_data['evaluation']['feedback']}\n"
            f"Strengths: {eval_data['evaluation']['strengths']}\n"
            f"Areas for Improvement: {eval_data['evaluation']['weaknesses']}"
            for i, eval_data in enumerate(evaluations)
        ])

        # Extract strengths and areas for improvement
        strengths = [eval_data["evaluation"]["strengths"] for eval_data in evaluations if eval_data["evaluation"]["strengths"] != "Not available"]
        areas_for_improvement = [eval_data["evaluation"]["weaknesses"] for eval_data in evaluations if eval_data["evaluation"]["weaknesses"] != "Not available"]

    else:
        avg_overall = 0.0
        avg_technical = 0.0
        avg_communication = 0.0
        detailed_evaluations = "No evaluations available"
        strengths = []
        areas_for_improvement = []

    # Generate overall analysis
    dialogue_text = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in valid_dialogue
    ])

    prompt = f"""
Generate a comprehensive interview analysis based on the following data:

RESUME SUMMARY: {resume_summary}

INTERVIEW TRANSCRIPT: {dialogue_text}

AI EVALUATION SUMMARY: {detailed_evaluations}

Provide detailed analysis covering overall performance, technical skills, communication abilities, and recommendations.
Focus on constructive feedback and specific areas for improvement.
"""

    try:
        analysis = await query_gemini_with_retry(prompt)

        report = InterviewReport(
            candidate_name=interview_state.user_details.name or "Candidate",
            candidate_phone=interview_state.user_details.phone,
            candidate_email=interview_state.user_details.email,
            interview_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_score=avg_overall * 10,  # Convert to 0-100 scale
            technical_score=avg_technical * 10,
            communication_score=avg_communication * 10,
            detailed_feedback=f"Real-time AI Evaluations:\n\n{detailed_evaluations}\n\n"
                            f"Overall Analysis:\n{analysis}",
            strengths=strengths[:5],  # Limit to top 5
            areas_for_improvement=areas_for_improvement[:5],  # Limit to top 5
            interview_transcript=valid_dialogue,  # Use filtered dialogue
            kpis=kpis,
            proctoring_violations=interview_state.proctoring_violations
        )

        return report

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return InterviewReport(
            candidate_name=interview_state.user_details.name or "Candidate",
            candidate_phone=interview_state.user_details.phone,
            candidate_email=interview_state.user_details.email,
            interview_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_score=avg_overall * 10,
            technical_score=avg_technical * 10,
            communication_score=avg_communication * 10,
            detailed_feedback=f"Real-time AI Evaluations:\n\n{detailed_evaluations}",
            strengths=strengths[:5],
            areas_for_improvement=areas_for_improvement[:5],
            interview_transcript=valid_dialogue,
            kpis=kpis,
            proctoring_violations=interview_state.proctoring_violations
        )

async def generate_report_background(dialogue: List[Dict], resume_summary: str, evaluations: List[Dict], session_id: str):
    """Generate report in background without blocking WebSocket"""
    try:
        report = await generate_interview_report(dialogue, resume_summary, evaluations)
        generated_reports[session_id] = report
        logger.info(f"Report generated for session {session_id}")
    except Exception as e:
        logger.error(f"Background report generation failed: {e}")

# --- Email Functions ---
async def send_interview_report_email(candidate_email: str, candidate_name: str, report: InterviewReport) -> dict:
    """Send interview report via email with PDF attachment.
    
    Returns dict with status and message for better feedback.
    """
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        logger.warning("Email credentials not configured. Please set EMAIL_USERNAME and EMAIL_PASSWORD in .env file")
        return {
            "success": False,
            "message": "Email not configured. Please contact administrator to set up email credentials.",
            "error_type": "configuration"
        }

    try:
        # Generate PDF content
        html_content = templates.get_template("report_pdf.html").render(
            report=report
        )

        # Create PDF
        pdf_buffer = BytesIO()
        HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        pdf_content = pdf_buffer.read()

        # Create email message
        message = MIMEMultipart()
        message["From"] = f"{EMAIL_FROM_NAME} <{EMAIL_USERNAME}>"
        message["To"] = candidate_email
        message["Subject"] = f"Your Interview Report - {candidate_name}"

        # Email body
        body = f"""Dear {candidate_name},

Thank you for participating in our AI-powered interview process. We are pleased to provide you with your comprehensive interview evaluation report.

📊 Interview Summary:

• Overall Score: {report.overall_score:.1f}/100
• Communication Score: {report.communication_score:.1f}/100
• Technical Score: {report.technical_score:.1f}/100
• Interview Date: {report.interview_date}

Please find your detailed interview report attached as a PDF. The report includes:

✅ Performance analysis across multiple dimensions
✅ Strengths and areas for improvement
✅ Detailed feedback from our AI evaluation system
✅ Complete interview transcript
✅ Proctoring integrity report

We encourage you to review the feedback carefully as it provides valuable insights to help you succeed in future interviews.

If you have any questions about your report, please don't hesitate to contact us.

Best regards,
AI Interview System Team

---
This is an automated message. Please do not reply to this email."""

        message.attach(MIMEText(body, "plain"))

        # Attach PDF
        pdf_attachment = MIMEBase("application", "octet-stream")
        pdf_attachment.set_payload(pdf_content)
        encoders.encode_base64(pdf_attachment)
        pdf_attachment.add_header(
            "Content-Disposition",
            f"attachment; filename=Interview_Report_{candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
        message.attach(pdf_attachment)

        # Send email using aiosmtplib for async support
        await aiosmtplib.send(
            message,
            hostname=EMAIL_SMTP_SERVER,
            port=EMAIL_SMTP_PORT,
            start_tls=EMAIL_USE_TLS,
            username=EMAIL_USERNAME,
            password=EMAIL_PASSWORD,
        )

        logger.info(f"✅ Interview report successfully sent to {candidate_email}")
        return {
            "success": True,
            "message": f"Interview report successfully sent to {candidate_email}",
            "email": candidate_email
        }

    except Exception as e:
        error_msg = f"Failed to send email to {candidate_email}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "error_type": "sending",
            "email": candidate_email
        }

async def send_email_notification(to_email: str, subject: str, body: str) -> dict:
    """Send a simple email notification without attachments.
    
    Returns dict with status and message for better feedback.
    """
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        logger.warning("Email credentials not configured. Please set EMAIL_USERNAME and EMAIL_PASSWORD in .env file")
        return {
            "success": False,
            "message": "Email not configured. Please contact administrator to set up email credentials.",
            "error_type": "configuration"
        }

    try:
        message = MIMEMultipart()
        message["From"] = f"{EMAIL_FROM_NAME} <{EMAIL_USERNAME}>"
        message["To"] = to_email
        message["Subject"] = subject

        message.attach(MIMEText(body, "plain"))

        await aiosmtplib.send(
            message,
            hostname=EMAIL_SMTP_SERVER,
            port=EMAIL_SMTP_PORT,
            start_tls=EMAIL_USE_TLS,
            username=EMAIL_USERNAME,
            password=EMAIL_PASSWORD,
        )

        logger.info(f"Email notification sent to {to_email}")
        return {
            "success": True,
            "message": f"Email notification sent to {to_email}",
            "email": to_email
        }

    except Exception as e:
        error_msg = f"Failed to send email notification to {to_email}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "error_type": "sending",
            "email": to_email
        }

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Serving index page.")
    global interview_state
    interview_state = InterviewState()
    return templates.TemplateResponse("index.html", {"request": request, "report": None})

@app.post("/start_interview")
async def start_interview(
    request: Request,
    resume: List[UploadFile] = File(...),
    name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...)
):
    logger.info("Starting new interview process.")
    global interview_state

    # Reset state completely
    interview_state.user_details = UserDetails(name=name, phone=phone, email=email)
    interview_state.resume_text = ""
    interview_state.resume_summary = ""  # Clear previous summary
    interview_state.current_dialogue = []
    interview_state.is_interview_active = False
    interview_state.current_question_count = 0
    interview_state.last_question = ""
    interview_state.consent_received = False
    interview_state.answer_evaluations = []
    interview_state.total_score = 0.0
    interview_state.average_score = 0.0
    interview_state.proctoring_session_id = str(uuid.uuid4())
    interview_state.proctoring_violations = []
    interview_state.reference_face_captured = False

    # FIXED: Reset email tracking for new session
    if interview_state.proctoring_session_id in email_sent_for_session:
        email_sent_for_session.remove(interview_state.proctoring_session_id)

    if not resume:
        logger.warning("No resume file uploaded.")
        raise HTTPException(status_code=400, detail="No resume file uploaded.")

    for file in resume:
        if not file.filename or not file.filename.endswith(".pdf"):
            logger.warning(f"Uploaded file is not a PDF: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        extracted_text = extract_text_from_pdf(tmp_path)
        logger.info(f"Extracted text from {file.filename}. Length: {len(extracted_text)}")
        interview_state.resume_text += extracted_text + "\n"

        os.unlink(tmp_path)

    if not interview_state.resume_text.strip():
        logger.warning("Unable to extract text from resume.")
        raise HTTPException(status_code=400, detail="Unable to extract text from resume.")

    # NEW: Kick off resume summarization in background to avoid blocking face capture
    async def _summarize_and_store(text: str):
        try:
            interview_state.resume_summary = await summarize_resume(text)
            logger.info(f"Resume summary generated. Length: {len(interview_state.resume_summary)}")
        except Exception as e:
            logger.error(f"Background resume summarization failed: {e}")
            # Fallback to truncated text if summarization fails
            interview_state.resume_summary = text[:4000]

    asyncio.create_task(_summarize_and_store(interview_state.resume_text))

    # Store user session data
    session_data = {
        "user_details": interview_state.user_details.dict(),
        "proctoring_session_id": interview_state.proctoring_session_id,
        "created_at": datetime.now().isoformat()
    }

    user_sessions[interview_state.proctoring_session_id] = session_data

    # Initialize proctoring session
    await proctoring_service.create_session(interview_state.proctoring_session_id)

    logger.info(f"Interview initialized for {name}. Awaiting face verification.")
    return JSONResponse({
        "status": "ready_for_face_capture",
        "message": "Please proceed to face verification.",
        "proctoring_session_id": interview_state.proctoring_session_id,
        "user_name": name
    })

@app.post("/capture_reference_face")
async def capture_reference_face(request: Request):
    """Store the captured reference face for identity verification."""
    try:
        data = await request.json()
        image_data = data.get('image_data')

        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Set reference face in proctoring service
        result = await proctoring_service.set_reference_face(
            interview_state.proctoring_session_id,
            image_data
        )

        if result.get('status') == 'success':
            interview_state.reference_face_captured = True
            logger.info("Reference face captured successfully")
            return JSONResponse({
                "status": "success",
                "message": "Reference face captured successfully"
            })
        else:
            # Still allow interview to proceed even if face capture fails
            interview_state.reference_face_captured = True
            logger.warning("Face capture failed but allowing interview to proceed")
            return JSONResponse({
                "status": "success",
                "message": "Session activated successfully"
            })

    except Exception as e:
        logger.error(f"Error capturing reference face: {e}")
        # Fallback: Still allow interview to proceed
        interview_state.reference_face_captured = True
        return JSONResponse({
            "status": "success",
            "message": "Session activated successfully"
        })

@app.get("/user_session/{session_id}")
async def get_user_session(session_id: str):
    """Get user session data."""
    if session_id in user_sessions:
        return JSONResponse(user_sessions[session_id])
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/start_interview_session")
async def start_interview_session():
    """Start the actual interview session after face verification."""
    try:
        # Allow interview to proceed regardless of face capture status
        interview_state.is_interview_active = True
        logger.info("Interview session started.")

        return JSONResponse({
            "status": "interview_ready",
            "message": "Interview session ready. Please connect to WebSocket to begin.",
            "proctoring_session_id": interview_state.proctoring_session_id
        })

    except Exception as e:
        logger.error(f"Error starting interview session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/interview")
async def websocket_interview(websocket: EncodedWebSocket):
    logger.info(f"WebSocket interview connection established for {websocket.client}")
    await manager.connect(websocket)

    # Send initial greeting only once
    greeting = "I'm an AI interviewer. I'm here to conduct a technical interview with you. Are you ready to begin?"

    interview_state.current_dialogue.append({
        "role": "interviewer",
        "content": greeting,
        "timestamp": datetime.now().isoformat()
    })

    interview_state.last_question = greeting

    # Generate audio asynchronously
    audio_filename = await generate_audio_async(greeting)

    await websocket.send_text(json.dumps({
        "type": "question",
        "content": greeting,
        "audio_file": audio_filename,
        "start_recording": True,
        "proctoring_session_id": interview_state.proctoring_session_id
    }, ensure_ascii=False))

    logger.info("Initial greeting sent via WebSocket")

    while interview_state.is_interview_active:
        try:
            data = await websocket.receive()
            if 'text' not in data:
                continue

            message = json.loads(data['text'])
            logger.debug(f"Received message type: {message.get('type')}")

            if message['type'] == 'text_response':
                user_text = message['content'].strip().replace('â—', '').replace('âœï¸', '')
                logger.info(f"User response: {user_text[:50]}...")

                # Only process non-empty responses
                if not user_text or user_text == "[No response]":
                    logger.warning("Empty response received, requesting user to try again")
                    await websocket.send_text(json.dumps({
                        "type": "question",
                        "content": "I didn't catch that. Could you please repeat your answer?",
                        "audio_file": await generate_audio_async("I didn't catch that. Could you please repeat your answer?"),
                        "start_recording": True
                    }))
                    continue

                # Send simple acknowledgment without evaluation details
                await websocket.send_text(json.dumps({
                    "type": "processing_response",
                    "content": "Thank you for your response. Processing next question..."
                }))

                # Process user response
                interview_state.current_dialogue.append({
                    "role": "candidate",
                    "content": user_text,
                    "timestamp": datetime.now().isoformat()
                })

                # Consent flow
                if not interview_state.consent_received:
                    if any(kw in user_text.lower() for kw in ["yes", "sure", "ready", "start", "okay", "go ahead", "begin"]):
                        interview_state.consent_received = True
                        logger.info("Consent received. Asking 'Tell me about yourself'")

                        first_question = await generate_introductory_question(interview_state.resume_summary)
                        interview_state.current_question_count += 1
                        interview_state.last_question = first_question

                        interview_state.current_dialogue.append({
                            "role": "interviewer",
                            "content": first_question,
                            "timestamp": datetime.now().isoformat()
                        })

                        audio_filename = await generate_audio_async(first_question)

                        await websocket.send_text(json.dumps({
                            "type": "question",
                            "content": first_question,
                            "audio_file": audio_filename,
                            "start_recording": True
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "question",
                            "content": "Just let me know when you're ready to begin the interview.",
                            "audio_file": await generate_audio_async("Just let me know when you're ready to begin the interview."),
                            "start_recording": True
                        }))
                    continue

                # **HIDDEN AI EVALUATION - run fully in background to reduce latency**
                if user_text.strip() and interview_state.last_question:
                    async def _evaluate_and_store(question_text: str, answer_text: str, resume_summary_text: str):
                        try:
                            evaluation = await evaluate_user_answer(
                                question_text,
                                answer_text,
                                resume_summary_text
                            )

                            interview_state.answer_evaluations.append({
                                "question": question_text,
                                "answer": answer_text,
                                "evaluation": evaluation,
                                "timestamp": datetime.now().isoformat()
                            })

                            interview_state.total_score += evaluation["overall_score"]
                            interview_state.average_score = interview_state.total_score / len(interview_state.answer_evaluations)

                            logger.info(f"Answer evaluated silently: Score {evaluation['overall_score']}/10, Average: {interview_state.average_score:.1f}")

                        except Exception as e:
                            logger.error(f"Error during silent answer evaluation: {e}")

                    asyncio.create_task(_evaluate_and_store(interview_state.last_question, user_text, interview_state.resume_summary))

                # Proceed immediately to next question (no blocking sleep)
                await asyncio.sleep(0)

                should_continue = interview_state.current_question_count < interview_state.max_questions

                if should_continue:
                    next_question = await generate_dynamic_question(
                        interview_state.resume_summary,  # Use summary instead of full text
                        user_text,
                        interview_state.current_dialogue
                    )

                    # If Gemini returns thank-you (i.e., it thinks interview is over)
                    if "thank you for attending" in next_question.lower():
                        should_continue = False
                else:
                    next_question = "Thank you for attending the interview. Your report will be generated shortly."
                    should_continue = False

                interview_state.current_question_count += 1
                interview_state.last_question = next_question

                interview_state.current_dialogue.append({
                    "role": "interviewer",
                    "content": next_question,
                    "timestamp": datetime.now().isoformat()
                })

                audio_filename = await generate_audio_async(next_question)

                await websocket.send_text(json.dumps({
                    "type": "question" if should_continue else "interview_concluded",
                    "content": next_question,
                    "audio_file": audio_filename,
                    "start_recording": should_continue,
                    "stop_recording": not should_continue,
                    "total_questions": len(interview_state.answer_evaluations)
                }))

                if not should_continue:
                    break

            elif message['type'] == 'end_interview':
                logger.info("Client requested to end interview.")
                conclusion = "Thank you for your time. Your interview report will be available shortly."

                interview_state.current_dialogue.append({
                    "role": "interviewer",
                    "content": conclusion,
                    "timestamp": datetime.now().isoformat()
                })

                audio_filename = await generate_audio_async(conclusion)

                await websocket.send_text(json.dumps({
                    "type": "interview_concluded",
                    "content": conclusion,
                    "audio_file": audio_filename,
                    "stop_recording": True
                }))
                break

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for {websocket.client}.")
            break
        except Exception as e:
            logger.error(f"Error processing WebSocket message for {websocket.client}: {e}", exc_info=True)
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "Sorry, something went wrong. Let's continue shortly.",
                "stop_recording": True
            }))
            break

    manager.disconnect(websocket)
    interview_state.is_interview_active = False

    # FIXED: Consolidated email sending logic - only send once per session
    if (interview_state.user_details.email and 
        interview_state.current_dialogue and
        interview_state.proctoring_session_id not in email_sent_for_session):
        
        try:
            logger.info(f"Generating and sending interview report to {interview_state.user_details.email}")
            # Generate the comprehensive report
            report = await generate_interview_report(
                interview_state.current_dialogue,
                interview_state.resume_summary,
                interview_state.answer_evaluations
            )

            # Send email in background with improved status handling
            async def _send_email_background():
                try:
                    result = await send_interview_report_email(
                        interview_state.user_details.email,
                        interview_state.user_details.name,
                        report
                    )

                    if result["success"]:
                        # Mark this session as emailed to prevent duplicates
                        email_sent_for_session.add(interview_state.proctoring_session_id)
                        logger.info(f"✅ EMAIL SUCCESS: {result['message']}")
                    else:
                        logger.error(f"❌ EMAIL FAILED: {result['message']}")
                        if result.get("error_type") == "configuration":
                            logger.error("💡 SOLUTION: Please configure email settings in .env file:")
                            logger.error("   EMAIL_USERNAME=your_email@gmail.com")
                            logger.error("   EMAIL_PASSWORD=your_app_password")
                            logger.error("   See EMAIL_README.md for detailed setup instructions")

                except Exception as e:
                    logger.error(f"💥 UNEXPECTED EMAIL ERROR: {e}")

            # Fire and forget email sending
            asyncio.create_task(_send_email_background())

        except Exception as e:
            logger.error(f"Error preparing email for {interview_state.user_details.email}: {e}")

    else:
        if not interview_state.user_details.email:
            logger.info("ℹ️ No email address provided - skipping email send")
        elif interview_state.proctoring_session_id in email_sent_for_session:
            logger.info("ℹ️ Email already sent for this session - skipping duplicate send")
        else:
            logger.info("ℹ️ No interview data available - skipping email send")

    # NEW: Clear resume data after interview ends to free memory
    interview_state.resume_text = ""
    interview_state.resume_summary = ""

    logger.info(f"Interview for {websocket.client} concluded and state reset.")

@app.websocket("/ws/proctoring")
async def websocket_proctoring(websocket: EncodedWebSocket):
    """WebSocket endpoint for proctoring functionality"""
    logger.info(f"Proctoring WebSocket connection established for {websocket.client}")
    await manager.connect_proctoring(websocket)

    while True:
        try:
            data = await websocket.receive()
            if 'text' not in data:
                continue

            message = json.loads(data['text'])
            logger.debug(f"Received proctoring message type: {message.get('type')}")

            if message['type'] == 'set_reference_face':
                result = await proctoring_service.set_reference_face(
                    message['session_id'],
                    message['image_data']
                )

                await websocket.send_text(json.dumps({
                    "type": "reference_face_response",
                    "result": result
                }))

            elif message['type'] == 'process_frame':
                result = await proctoring_service.process_frame(
                    message['session_id'],
                    message['image_data']
                )

                # Store violations in interview state
                if result.get('violations'):
                    for violation in result['violations']:
                        if violation.get('type') == 'violation':
                            interview_state.proctoring_violations.append({
                                "timestamp": datetime.now().isoformat(),
                                "type": violation.get('message', 'Unknown violation'),
                                "severity": violation.get('severity', 'medium')
                            })

                # Check if session should be terminated
                if result.get('violations'):
                    for violation in result['violations']:
                        if violation.get('terminate', False):
                            # End interview due to proctoring violations
                            interview_state.is_interview_active = False
                            logger.info("Interview terminated due to proctoring violations")
                            break

                await websocket.send_text(json.dumps({
                    "type": "proctoring_result",
                    "result": result
                }))

        except WebSocketDisconnect:
            logger.info(f"Proctoring WebSocket disconnected for {websocket.client}.")
            break
        except Exception as e:
            logger.error(f"Error processing proctoring WebSocket message: {e}", exc_info=True)
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Proctoring error occurred"
            }))
            break

    manager.disconnect_proctoring(websocket)

# FIXED: Single report route with automatic email sending and notification
@app.get("/report", response_class=HTMLResponse)
async def get_report(request: Request):
    """Generate and display the interview report with automatic email sending."""
    global interview_state

    if not interview_state.current_dialogue:
        logger.warning("No interview data available for report generation")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "No interview data available. Please complete an interview first."
        })

    # End proctoring session if active
    if interview_state.proctoring_session_id:
        await proctoring_service.end_session(interview_state.proctoring_session_id)

    # Generate the comprehensive report using resume summary
    report = await generate_interview_report(
        interview_state.current_dialogue,
        interview_state.resume_summary,
        interview_state.answer_evaluations
    )

    # AUTOMATIC EMAIL SENDING - Only send once per session
    email_status = None
    if (interview_state.proctoring_session_id not in email_sent_for_session and 
        interview_state.user_details.email and 
        EMAIL_USERNAME and EMAIL_PASSWORD):
        
        try:
            # Send email automatically
            email_result = await send_interview_report_email(
                interview_state.user_details.email,
                interview_state.user_details.name,
                report
            )
            
            if email_result["success"]:
                email_status = {
                    "success": True,
                    "message": f"✅ Report automatically sent to {interview_state.user_details.email}"
                }
                # Mark this session as emailed to prevent duplicates
                email_sent_for_session.add(interview_state.proctoring_session_id)
                logger.info(f"✅ Automatic email sent successfully to {interview_state.user_details.email}")
            else:
                email_status = {
                    "success": False,
                    "message": f"❌ Failed to send email: {email_result['message']}"
                }
                logger.error(f"❌ Automatic email failed: {email_result['message']}")
        except Exception as e:
            email_status = {
                "success": False,
                "message": f"❌ Email error: {str(e)}"
            }
            logger.error(f"❌ Automatic email exception: {e}")
    elif interview_state.proctoring_session_id in email_sent_for_session:
        email_status = {
            "success": True,
            "message": f"✅ Report was already sent to {interview_state.user_details.email}"
        }
    elif not EMAIL_USERNAME or not EMAIL_PASSWORD:
        email_status = {
            "success": False,
            "message": "⚠️ Email not configured. Please set up email credentials."
        }
    elif not interview_state.user_details.email:
        email_status = {
            "success": False,
            "message": "⚠️ No candidate email address available."
        }

    return templates.TemplateResponse("report.html", {
        "request": request,
        "report": report,
        "email_status": email_status  # Pass email status to template
    })

@app.get("/report/download")
async def download_report_pdf(request: Request):
    """Generate and download PDF report."""
    global interview_state

    if not interview_state.current_dialogue:
        raise HTTPException(status_code=404, detail="No interview data available")

    # Generate the report using resume summary
    report = await generate_interview_report(
        interview_state.current_dialogue,
        interview_state.resume_summary,  # Use summary instead of full text
        interview_state.answer_evaluations
    )

    # Render HTML template
    html_content = templates.get_template("report_pdf.html").render(
        request=request,
        report=report
    )

    # Generate PDF
    pdf_buffer = BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)

    return Response(
        content=pdf_buffer.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=interview_report.pdf"}
    )

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    logger.info(f"Serving audio file: {filename}")
    file_path = os.path.join(AUDIO_FOLDER, filename)

    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(file_path)

@app.post("/send_report_email")
async def send_report_email_manually(request: Request):
    """Manual email sending with duplicate prevention."""
    try:
        data = await request.json()
        candidate_email = data.get("email", "").strip()
        candidate_name = data.get("name", "").strip()

        if not candidate_email or not candidate_name:
            raise HTTPException(status_code=400, detail="Email and name are required")

        if not interview_state.current_dialogue:
            raise HTTPException(status_code=404, detail="No interview data available")

        # Generate fresh report
        report = await generate_interview_report(
            interview_state.current_dialogue,
            interview_state.resume_summary,
            interview_state.answer_evaluations
        )
        
        # Update report with provided details
        report.candidate_email = candidate_email
        report.candidate_name = candidate_name

        # Send email
        result = await send_interview_report_email(candidate_email, candidate_name, report)

        if result["success"]:
            # Mark as sent to prevent automatic sending again
            if interview_state.proctoring_session_id:
                email_sent_for_session.add(interview_state.proctoring_session_id)
            
            logger.info(f"✅ Manual email sent successfully to {candidate_email}")
            return JSONResponse({
                "success": True,
                "message": f"Report successfully sent to {candidate_email}"
            })
        else:
            # Return appropriate error status based on error type
            status_code = 400 if result.get("error_type") == "configuration" else 500
            raise HTTPException(status_code=status_code, detail=result["message"])

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error in manual email sending: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_email")
async def test_email_configuration():
    """Test email configuration by sending a test email."""
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        raise HTTPException(status_code=400, detail="Email credentials not configured. Please set EMAIL_USERNAME and EMAIL_PASSWORD in .env file")

    try:
        result = await send_email_notification(
            EMAIL_USERNAME,  # Send to self for testing
            "AI Interview System - Email Test",
            "This is a test email to verify that the email configuration is working correctly.\n\nIf you receive this message, the email system is properly configured.\n\nTime: " + str(datetime.now())
        )

        if result["success"]:
            return JSONResponse({
                "status": "success",
                "message": result["message"]
            })
        else:
            status_code = 400 if result.get("error_type") == "configuration" else 500
            raise HTTPException(status_code=status_code, detail=result["message"])

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error testing email configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/email_status")
async def get_email_status():
    """Get current email configuration status."""
    is_configured = bool(EMAIL_USERNAME and EMAIL_PASSWORD)
    return JSONResponse({
        "email_configured": is_configured,
        "smtp_server": EMAIL_SMTP_SERVER if is_configured else "Not configured",
        "smtp_port": EMAIL_SMTP_PORT if is_configured else "Not configured",
        "from_name": EMAIL_FROM_NAME if is_configured else "Not configured",
        "use_tls": EMAIL_USE_TLS if is_configured else "Not configured",
        "username": EMAIL_USERNAME if is_configured else "Not configured",
        "setup_instructions": {
            "message": "Email not configured. Automatic report sending is disabled." if not is_configured else "Email is properly configured.",
            "steps": [
                "1. Edit the .env file in the project root",
                "2. Uncomment and configure the EMAIL_* variables",
                "3. For Gmail: Use App Password (not regular password)",
                "4. Restart the application",
                "5. Test email configuration using /test_email endpoint"
            ] if not is_configured else []
        }
    })

@app.get("/email_setup_guide")
async def email_setup_guide():
    """Provide detailed email setup instructions."""
    return JSONResponse({
        "title": "Email Setup Guide for AI Interview System",
        "current_status": "Configured" if (EMAIL_USERNAME and EMAIL_PASSWORD) else "Not Configured",
        "gmail_setup": {
            "title": "Gmail Setup (Recommended)",
            "steps": [
                "1. Enable 2-Factor Authentication on your Gmail account",
                "2. Go to Google Account Settings → Security",
                "3. Under '2-Step Verification', select 'App passwords'",
                "4. Generate a password for 'Mail'",
                "5. Copy the generated 16-character password",
                "6. In .env file, set EMAIL_USERNAME=your_email@gmail.com",
                "7. In .env file, set EMAIL_PASSWORD=the_generated_app_password",
                "8. Restart the application"
            ],
            "env_example": {
                "EMAIL_SMTP_SERVER": "smtp.gmail.com",
                "EMAIL_SMTP_PORT": "587",
                "EMAIL_USERNAME": "your_email@gmail.com",
                "EMAIL_PASSWORD": "your_16_char_app_password",
                "EMAIL_FROM_NAME": "AI Interview System",
                "EMAIL_USE_TLS": "true"
            }
        },
        "other_providers": {
            "outlook": {
                "EMAIL_SMTP_SERVER": "smtp-mail.outlook.com",
                "EMAIL_SMTP_PORT": "587"
            },
            "yahoo": {
                "EMAIL_SMTP_SERVER": "smtp.mail.yahoo.com",
                "EMAIL_SMTP_PORT": "587"
            }
        },
        "testing": {
            "message": "After configuration, test your email setup",
            "endpoint": "POST /test_email",
            "description": "Sends a test email to your configured email address"
        },
        "troubleshooting": [
            "Ensure 2FA is enabled for Gmail",
            "Use App Password, not your regular Gmail password",
            "Check firewall settings for SMTP ports",
            "Verify email address and password are correct",
            "Check spam folder for test emails"
        ]
    })

# --- ATS Routes ---
@app.get("/ats", response_class=HTMLResponse)
async def ats_tracker(request: Request):
    """Serve the ATS tracker page"""
    return templates.TemplateResponse("ats_tracker.html", {"request": request})

@app.post("/ats/analyze")
async def analyze_ats(
    job_description: str = Form(...),
    resume: UploadFile = File(...)
):
    """Analyze resume against job description"""
    try:
        # Validate inputs
        if not job_description.strip():
            raise HTTPException(status_code=400, detail="Job description is required")
        
        if not resume.filename or not resume.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check if Google API is configured
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=400, 
                detail="Google API key not configured. Please set GOOGLE_API_KEY in your .env file"
            )
        
        # Read and extract text from PDF
        contents = await resume.read()
        
        # Create a temporary file to read with PyPDF2
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # Extract text using PyPDF2
            with open(tmp_path, 'rb') as file:
                resume_text = extract_text_from_pdf_ats(file)
            
            if not resume_text.strip():
                raise HTTPException(status_code=400, detail="Unable to extract text from PDF")
            
            # Perform ATS analysis
            result = await analyze_resume_ats(resume_text, job_description)
            
            logger.info(f"ATS analysis completed successfully. Match: {result.jd_match}")
            
            return JSONResponse({
                "success": result.success,
                "jd_match": result.jd_match,
                "missing_keywords": result.missing_keywords,
                "profile_summary": result.profile_summary,
                "error_message": result.error_message
            })
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in ATS analysis: {e}")
        return JSONResponse({
            "success": False,
            "jd_match": "N/A",
            "missing_keywords": [],
            "profile_summary": "Analysis failed",
            "error_message": str(e)
        }, status_code=500)

@app.get("/ats/status")
async def ats_status():
    """Check ATS configuration status"""
    return JSONResponse({
        "google_api_configured": bool(GOOGLE_API_KEY),
        "status": "ready" if GOOGLE_API_KEY else "not_configured",
        "message": "ATS functionality is ready" if GOOGLE_API_KEY else "Please configure GOOGLE_API_KEY in .env file"
    })

if __name__ == "__main__":
    logger.info("Starting FastAPI application with Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8480, reload=True)
