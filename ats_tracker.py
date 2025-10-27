import google.generativeai as genai
import PyPDF2 as pdf
import json
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class ATSTracker:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
    
    def get_gemini_response(self, input_text: str) -> str:
        """Get response from Gemini model"""
        try:
            response = self.model.generate_content(input_text)
            return response.text
        except Exception as e:
            logger.error(f"Error getting Gemini response: {e}")
            raise
    
    def analyze_resume_ats(self, resume_text: str, job_description: str) -> Dict:
        """Analyze resume against job description for ATS score"""
        
        prompt = f"""
You are an expert ATS (Application Tracking System) with a strong understanding of 
software engineering, data science, data analytics, and big data engineering roles.

Evaluate the resume against the job description and provide:
1. A percentage match score (0–100).
2. A list of missing important keywords.
3. A short profile summary (2–3 lines).
4. Specific recommendations to improve ATS score.
5. Technical skills alignment score.

Resume: {resume_text}
Job Description: {job_description}

Respond ONLY in the following valid JSON format:
{{
  "JD Match": "85%",
  "MissingKeywords": ["keyword1", "keyword2"],
  "Profile Summary": "short summary here",
  "Technical Skills Alignment": "78%",
  "Recommendations": ["suggestion1", "suggestion2"],
  "Overall ATS Score": "82%"
}}
"""
        
        try:
            response = self.get_gemini_response(prompt)
            parsed_response = json.loads(response)
            return {
                "success": True,
                "data": parsed_response
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {
                "success": False,
                "error": "Failed to parse ATS analysis",
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"ATS analysis error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise
