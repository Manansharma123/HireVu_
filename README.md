# AI-Powered Interview Platform

An AI-driven interview automation system that conducts, evaluates, and proctors candidate interviews in real-time.

## 🚀 Core Features

- **AI Interview Conductor**: Dynamically generates questions based on resume content and conversation flow
- **Real-Time Answer Evaluation**: Scores responses across 6 metrics (technical accuracy, communication, relevance, depth, confidence, problem-solving)
- **ATS Resume Screening**: Analyzes resumes against job descriptions with match scoring and keyword extraction
- **Video Proctoring**: Real-time facial recognition, gaze tracking, and identity verification using MediaPipe
- **Automated Reports**: Generates comprehensive interview reports with KPIs, feedback, and violation logs
- **Email Notifications**: Sends interview reports as PDF attachments

## 🛠️ Technology Stack

- **Backend**: FastAPI with WebSocket support
- **AI Models**: Google Gemini, OpenRouter, Azure OpenAI
- **Computer Vision**: MediaPipe, OpenCV
- **PDF Processing**: PyPDF2, pdfplumber
- **Text-to-Speech**: gTTS
- **Email**: aiosmtplib

## 📁 Project Structure

├── app.py                    # Main FastAPI application
├── proctoring_service.py     # Video proctoring logic
├── ats_tracker.py            # ATS analysis module
├── interview.js              # Frontend JavaScript
├── report.html               # Report template
└── .env                      # Configuration

## 🎯 Workflow

1. **Resume Screening**: Upload resume → ATS analyzes match score and keywords
2. **Interview Setup**: Enter details → Capture face for proctoring verification
3. **Live Interview**: AI asks questions → Records answers → Evaluates in real-time
4. **Proctoring**: Monitors face, gaze, and detects violations (looking away, multiple faces, identity change)
5. **Report Generation**: Creates detailed report → Emails to candidate

## 🔒 Proctoring Rules

- No face detected for 5+ seconds → Violation
- Multiple faces detected → Violation
- Looking away for 5+ seconds → Violation
- Identity mismatch → Violation
- Max 3 violations → Session terminated

## 📊 Evaluation Metrics (0-10 scale)

- Technical Accuracy
- Communication Clarity
- Relevance
- Depth of Knowledge
- Confidence
- Problem-Solving Ability

## 📡 Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/interview` | GET | Interview interface |
| `/ats/analyze` | POST | Analyze resume |
| `/ws/interview` | WebSocket | Real-time interview |
| `/ws/proctoring/{id}` | WebSocket | Proctoring stream |
| `/report` | GET | View report |

---

**Built with Python, FastAPI, and AI**
