# speech_processor_enhanced.py - Enhanced Speech Processing with Advanced Features
from dotenv import load_dotenv
load_dotenv()

import os
import json
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import numpy as np

# Core speech processing imports
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper imported:", whisper.__version__)
except Exception as e:
    WHISPER_AVAILABLE = False
    print("❌ Whisper import failed:", e)
    
# Audio processing imports
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("⚠️ Warning: Audio processing not available. Install: pip install librosa soundfile")

# TTS imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ Warning: TTS not available. Install: pip install pyttsx3")

# Advanced NLP imports
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Warning: Transformers not available. Install: pip install transformers torch")

# Diarization imports
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("⚠️ Warning: Speaker diarization not available. Install: pip install pyannote.audio")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def ensure_dir(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

def save_text(path: Path, text: str):
    """Save text to file"""
    path.write_text(text, encoding="utf-8")

def save_json(path: Path, obj: Any):
    """Save object as JSON"""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

class SpeechProcessorEnhanced:
    """Enhanced speech processor with advanced features"""
    
    def __init__(self, 
                 whisper_model: str = "base",
                 enable_diarization: bool = True,
                 enable_sentiment: bool = True,
                 enable_summarization: bool = True):
        
        self.whisper_model = whisper_model
        self.enable_diarization = enable_diarization and DIARIZATION_AVAILABLE
        self.enable_sentiment = enable_sentiment and TRANSFORMERS_AVAILABLE
        self.enable_summarization = enable_summarization and TRANSFORMERS_AVAILABLE
        
        # Initialize models
        self._init_whisper()
        self._init_diarization()
        self._init_nlp_models()
        self._init_tts()
        
        print(f"✅ Enhanced Speech Processor initialized")
        print(f"   - Whisper: {'✅' if WHISPER_AVAILABLE else '❌'}")
        print(f"   - Diarization: {'✅' if self.enable_diarization else '❌'}")
        print(f"   - Sentiment Analysis: {'✅' if self.enable_sentiment else '❌'}")
        print(f"   - Summarization: {'✅' if self.enable_summarization else '❌'}")
        print(f"   - TTS: {'✅' if TTS_AVAILABLE else '❌'}")

    def _init_whisper(self):
        """Initialize Whisper model"""
        if WHISPER_AVAILABLE:
            try:
                self.whisper = whisper.load_model(self.whisper_model)
                print(f"✅ Whisper model '{self.whisper_model}' loaded")
            except Exception as e:
                print(f"❌ Error loading Whisper: {e}")
                self.whisper = None
        else:
            self.whisper = None

    def _init_diarization(self):
        """Initialize speaker diarization"""
        if self.enable_diarization:
            try:
                # Note: This requires a Hugging Face token for the pretrained model
                # You can get one at https://huggingface.co/settings/tokens
                hf_token = os.environ.get("HUGGINGFACE_TOKEN")
                if hf_token:
                    self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                        "pyannote/speaker-diarization@2022.07",
                        use_auth_token=hf_token
                    )
                    print("✅ Speaker diarization pipeline loaded")
                else:
                    print("⚠️ No Hugging Face token found. Speaker diarization disabled.")
                    self.diarization_pipeline = None
                    self.enable_diarization = False
            except Exception as e:
                print(f"❌ Error loading diarization pipeline: {e}")
                self.diarization_pipeline = None
                self.enable_diarization = False
        else:
            self.diarization_pipeline = None

    def _init_nlp_models(self):
        """Initialize NLP models for analysis"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Sentiment analysis
                if self.enable_sentiment:
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        return_all_scores=True
                    )
                
                # Summarization
                if self.enable_summarization:
                    self.summarizer = pipeline(
                        "summarization",
                        model="facebook/bart-large-cnn",
                        max_length=150,
                        min_length=50,
                        do_sample=False
                    )
                
                # Question answering for key moments
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad"
                )
                
                print("✅ NLP models loaded successfully")
            except Exception as e:
                print(f"❌ Error loading NLP models: {e}")
                self.sentiment_analyzer = None
                self.summarizer = None
                self.qa_pipeline = None
        else:
            self.sentiment_analyzer = None
            self.summarizer = None
            self.qa_pipeline = None

    def _init_tts(self):
        """Initialize Text-to-Speech"""
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                # Configure TTS settings
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                self.tts_engine.setProperty('rate', 180)  # Speaking rate
                self.tts_engine.setProperty('volume', 0.9)  # Volume level
                print("✅ TTS engine initialized")
            except Exception as e:
                print(f"❌ Error initializing TTS: {e}")
                self.tts_engine = None
        else:
            self.tts_engine = None

    def preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio file for better quality"""
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio_path
        
        try:
            # Load audio
            audio, sr_rate = librosa.load(audio_path, sr=16000)
            
            # Apply noise reduction and normalization
            audio = librosa.effects.preemphasis(audio)
            audio = librosa.util.normalize(audio)
            
            # Save preprocessed audio
            temp_path = audio_path.replace(".wav", "_processed.wav")
            sf.write(temp_path, audio, sr_rate)
            
            return temp_path
        except Exception as e:
            print(f"⚠️ Audio preprocessing failed: {e}")
            return audio_path

    def transcribe_with_whisper(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe audio using Whisper with enhanced features"""
        if not self.whisper:
            return {"error": "Whisper model not available"}
        
        try:
            # Preprocess audio
            processed_path = self.preprocess_audio(audio_path)
            
            # Transcribe with Whisper
            options = {}
            if language and language != "auto-detect":
                options["language"] = language
            
            result = self.whisper.transcribe(processed_path, **options)
            
            # Clean up processed file if different from original
            if processed_path != audio_path:
                try:
                    os.remove(processed_path)
                except:
                    pass
            
            # Extract detailed information
            transcript_data = {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": []
            }
            
            # Process segments with more details
            for segment in result.get("segments", []):
                segment_data = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": segment.get("avg_logprob", 0.0),
                    "no_speech_prob": segment.get("no_speech_prob", 0.0)
                }
                transcript_data["segments"].append(segment_data)
            
            return transcript_data
        except Exception as e:
            return {"error": f"Transcription failed: {e}"}

    def perform_speaker_diarization(self, audio_path: str) -> Dict[str, Any]:
        """Perform speaker diarization to identify different speakers"""
        if not self.enable_diarization or not self.diarization_pipeline:
            return {"speakers": {}, "segments": []}
        
        try:
            # Run diarization
            diarization = self.diarization_pipeline(audio_path)
            
            speakers = {}
            segments = []
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = speaker
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        "total_time": 0.0,
                        "segments": []
                    }
                
                segment_duration = turn.end - turn.start
                speakers[speaker_id]["total_time"] += segment_duration
                speakers[speaker_id]["segments"].append({
                    "start": turn.start,
                    "end": turn.end,
                    "duration": segment_duration
                })
                
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker_id,
                    "duration": segment_duration
                })
            
            return {
                "speakers": speakers,
                "segments": segments,
                "num_speakers": len(speakers)
            }
        except Exception as e:
            print(f"❌ Speaker diarization failed: {e}")
            return {"speakers": {}, "segments": []}

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        if not self.enable_sentiment or not self.sentiment_analyzer:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        try:
            # Split text into chunks if too long
            max_length = 512
            if len(text) > max_length:
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            else:
                chunks = [text]
            
            sentiments = []
            for chunk in chunks:
                result = self.sentiment_analyzer(chunk)
                sentiments.extend(result)
            
            # Aggregate sentiments
            sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
            for sentiment_list in sentiments:
                for sentiment in sentiment_list:
                    label = sentiment["label"].lower()
                    if "positive" in label:
                        sentiment_scores["positive"] += sentiment["score"]
                    elif "negative" in label:
                        sentiment_scores["negative"] += sentiment["score"]
                    else:
                        sentiment_scores["neutral"] += sentiment["score"]
            
            # Determine overall sentiment
            max_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            max_score = sentiment_scores[max_sentiment] / len(sentiments) if sentiments else 0
            
            return {
                "sentiment": max_sentiment,
                "confidence": max_score,
                "scores": sentiment_scores
            }
        except Exception as e:
            print(f"❌ Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.0}

    def extract_keywords_and_phrases(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords and key phrases from text"""
        try:
            # Simple TF-IDF based keyword extraction
            from collections import Counter
            import math
            
            # Tokenize and clean text
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Remove common stop words
            stop_words = {
                'the', 'and', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'did',
                'this', 'that', 'these', 'those', 'with', 'from', 'they', 'them',
                'she', 'her', 'his', 'him', 'you', 'your', 'our', 'out', 'about',
                'into', 'than', 'only', 'other', 'some', 'what', 'when', 'where',
                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                'such', 'not', 'too', 'very', 'just', 'now', 'also', 'then'
            }
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Calculate word frequencies
            word_counts = Counter(filtered_words)
            total_words = len(filtered_words)
            
            # Calculate TF-IDF-like scores
            keywords = []
            for word, count in word_counts.most_common(50):
                tf = count / total_words
                # Simple IDF approximation
                idf = math.log(total_words / count)
                score = tf * idf
                keywords.append((word, score))
            
            # Sort by score and return top keywords
            keywords.sort(key=lambda x: x[1], reverse=True)
            return keywords[:20]
            
        except Exception as e:
            print(f"❌ Keyword extraction failed: {e}")
            return []

    def extract_action_items(self, text: str) -> List[str]:
        """Extract action items and tasks from text"""
        try:
            action_items = []
            
            # Patterns for action items
            patterns = [
                r'(?:TODO|Action|Task|Follow up|Next step)[:\s]+([^\n.!?]*)',
                r'(?:Need to|Should|Must|Have to|Will)\s+([^\n.!?]*)',
                r'(?:^|\n)\s*[-•]\s*([^\n]*(?:complete|finish|implement|create|develop|schedule|prepare|organize|plan|review|update|contact|call|email|send|meet)[^\n]*)',
                r'(?:Assigned to|Responsible for|Owner)[:\s]+([^\n]*)',
                r'(?:By|Due|Deadline)[:\s]+([^\n]*)',
                r'(?:Action required|Please)[:\s]+([^\n]*)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    clean_match = match.strip()
                    if len(clean_match) > 10 and clean_match not in action_items:
                        action_items.append(clean_match)
            
            # Also use QA pipeline if available
            if self.qa_pipeline:
                try:
                    questions = [
                        "What needs to be done?",
                        "What are the action items?",
                        "What tasks were mentioned?",
                        "What follow-up is needed?"
                    ]
                    
                    for question in questions:
                        result = self.qa_pipeline(question=question, context=text)
                        if result["score"] > 0.1:  # Confidence threshold
                            answer = result["answer"].strip()
                            if len(answer) > 10 and answer not in action_items:
                                action_items.append(answer)
                except Exception:
                    pass
            
            return action_items[:15]  # Return top 15 action items
            
        except Exception as e:
            print(f"❌ Action item extraction failed: {e}")
            return []

    def identify_key_moments(self, segments: List[Dict], text: str) -> List[Dict]:
        """Identify key moments in the audio based on various criteria"""
        try:
            key_moments = []
            
            if not segments:
                return key_moments
            
            # Calculate importance scores for each segment
            for segment in segments:
                segment_text = segment.get("text", "")
                if not segment_text or len(segment_text.strip()) < 10:
                    continue
                
                importance_score = 0.0
                
                # Check for important keywords
                important_keywords = [
                    "important", "crucial", "critical", "key", "significant",
                    "decision", "conclusion", "summary", "action", "next steps",
                    "deadline", "urgent", "priority", "problem", "solution",
                    "recommend", "suggest", "propose", "agree", "disagree"
                ]
                
                text_lower = segment_text.lower()
                for keyword in important_keywords:
                    if keyword in text_lower:
                        importance_score += 1.0
                
                # Check for questions
                if "?" in segment_text:
                    importance_score += 0.5
                
                # Check for emphasis (repeated words, capitals)
                if re.search(r'[A-Z]{2,}', segment_text):
                    importance_score += 0.3
                
                # Check for numbers and dates
                if re.search(r'\d+', segment_text):
                    importance_score += 0.2
                
                # Normalize by segment length
                importance_score = importance_score / max(len(segment_text.split()), 1)
                
                if importance_score > 0.1:  # Threshold for key moments
                    key_moments.append({
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "text": segment_text,
                        "score": importance_score
                    })
            
            # Sort by importance score
            key_moments.sort(key=lambda x: x["score"], reverse=True)
            return key_moments[:10]  # Return top 10 key moments
            
        except Exception as e:
            print(f"❌ Key moment identification failed: {e}")
            return []

    def detect_contradictions(self, text: str) -> List[str]:
        """Detect potential contradictions or uncertainties in the text"""
        try:
            contradictions = []
            
            # Patterns for contradictory statements
            contradiction_patterns = [
                r'(?:but|however|although|despite|nevertheless|on the other hand)[^\n.!?]*',
                r'(?:not sure|uncertain|unclear|maybe|perhaps|might be|could be)[^\n.!?]*',
                r'(?:contradicts|conflicts with|differs from|opposite)[^\n.!?]*',
                r'(?:changed my mind|reconsider|rethink)[^\n.!?]*'
            ]
            
            for pattern in contradiction_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    clean_match = match.strip()
                    if len(clean_match) > 15 and clean_match not in contradictions:
                        contradictions.append(clean_match)
            
            return contradictions[:10]
            
        except Exception as e:
            print(f"❌ Contradiction detection failed: {e}")
            return []

    def generate_mindmap(self, text: str, keywords: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Generate a mind map structure from the content"""
        try:
            # Create a simple mind map structure
            central_topic = "Main Topic"
            
            # Group keywords into categories
            categories = {
                "Actions": [],
                "People": [],
                "Concepts": [],
                "Time": [],
                "Other": []
            }
            
            action_words = {"do", "make", "create", "implement", "develop", "build", "plan", "organize"}
            people_words = {"team", "person", "people", "client", "customer", "user", "manager"}
            time_words = {"time", "date", "week", "month", "year", "deadline", "schedule"}
            
            for word, score in keywords[:15]:
                if any(action in word for action in action_words):
                    categories["Actions"].append(word)
                elif any(people in word for people in people_words):
                    categories["People"].append(word)
                elif any(time in word for time in time_words):
                    categories["Time"].append(word)
                elif len(word) > 5:  # Likely concepts
                    categories["Concepts"].append(word)
                else:
                    categories["Other"].append(word)
            
            # Generate Mermaid diagram syntax
            mermaid_lines = ["graph TD"]
            mermaid_lines.append(f"    A[{central_topic}]")
            
            node_id = ord('B')
            for category, items in categories.items():
                if items:
                    cat_node = chr(node_id)
                    mermaid_lines.append(f"    A --> {cat_node}[{category}]")
                    
                    for i, item in enumerate(items[:5]):  # Limit items per category
                        item_node = chr(node_id + i + 1)
                        clean_item = item.replace('"', '').replace("'", "")[:20]
                        mermaid_lines.append(f"    {cat_node} --> {item_node}[{clean_item}]")
                    
                    node_id += len(items[:5]) + 1
            
            return {
                "categories": categories,
                "mermaid": "\n".join(mermaid_lines)
            }
            
        except Exception as e:
            print(f"❌ Mind map generation failed: {e}")
            return {"categories": {}, "mermaid": ""}

    def generate_timeline(self, action_items: List[str], text: str) -> List[Dict]:
        """Generate a suggested timeline for action items"""
        try:
            timeline = []
            
            # Priority keywords for urgency
            urgent_keywords = ["urgent", "asap", "immediately", "critical", "priority"]
            medium_keywords = ["soon", "this week", "next week", "important"]
            
            # Time frame keywords
            time_patterns = {
                "today": 0,
                "tomorrow": 1,
                "this week": 7,
                "next week": 14,
                "this month": 30,
                "next month": 60,
                "quarter": 90
            }
            
            for i, action in enumerate(action_items):
                action_lower = action.lower()
                
                # Determine timeframe based on content
                timeframe = "TBD"
                priority = "Medium"
                days_estimate = 30  # Default
                
                # Check for urgent indicators
                if any(urgent in action_lower for urgent in urgent_keywords):
                    priority = "High"
                    timeframe = "This week"
                    days_estimate = 7
                elif any(medium in action_lower for medium in medium_keywords):
                    priority = "Medium"
                    timeframe = "Next 2 weeks"
                    days_estimate = 14
                else:
                    priority = "Low"
                    timeframe = "This month"
                    days_estimate = 30
                
                # Check for specific time mentions
                for time_phrase, days in time_patterns.items():
                    if time_phrase in action_lower:
                        timeframe = time_phrase.title()
                        days_estimate = days
                        break
                
                timeline.append({
                    "task": action,
                    "timeframe": timeframe,
                    "priority": priority,
                    "estimated_days": days_estimate,
                    "order": i
                })
            
            # Sort by priority and estimated days
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            timeline.sort(key=lambda x: (priority_order.get(x["priority"], 2), x["estimated_days"]))
            
            return timeline
            
        except Exception as e:
            print(f"❌ Timeline generation failed: {e}")
            return []

    def generate_summary(self, text: str) -> str:
        """Generate a summary of the text"""
        if not self.enable_summarization or not self.summarizer:
            # Fallback: simple extractive summary
            sentences = text.split('. ')
            if len(sentences) <= 3:
                return text
            
            # Return first few sentences as summary
            return '. '.join(sentences[:3]) + '.'
        
        try:
            # Handle long text by chunking
            max_chunk_length = 1024
            if len(text) <= max_chunk_length:
                result = self.summarizer(text, max_length=150, min_length=50, do_sample=False)
                return result[0]['summary_text']
            else:
                # Split into chunks and summarize each
                chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length//2)]
                summaries = []
                
                for chunk in chunks[:3]:  # Limit to 3 chunks to avoid too long processing
                    if len(chunk.strip()) > 100:
                        result = self.summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                        summaries.append(result[0]['summary_text'])
                
                return ' '.join(summaries)
                
        except Exception as e:
            print(f"❌ Summarization failed: {e}")
            # Fallback summary
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text

    def process_audio(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Main method to process audio with all enhanced features"""
        print(f"🎵 Processing audio: {Path(audio_path).name}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "audio_file": str(Path(audio_path).name)
        }
        
        try:
            # Step 1: Transcribe audio
            print("📝 Transcribing audio...")
            transcript_result = self.transcribe_with_whisper(audio_path, language)
            
            if "error" in transcript_result:
                return {"error": transcript_result["error"]}
            
            results["transcript"] = transcript_result
            full_text = transcript_result["text"]
            
            if not full_text.strip():
                return {"error": "No speech detected in audio"}
            
            # Step 2: Speaker diarization
            print("👥 Performing speaker diarization...")
            diarization_result = self.perform_speaker_diarization(audio_path)
            results["speakers"] = diarization_result.get("speakers", {})
            results["speaker_segments"] = diarization_result.get("segments", [])
            
            # Step 3: Extract keywords and phrases
            print("🔑 Extracting keywords...")
            keywords = self.extract_keywords_and_phrases(full_text)
            results["keywords"] = keywords
            
            # Step 4: Extract action items
            print("✅ Extracting action items...")
            action_items = self.extract_action_items(full_text)
            results["action_items"] = action_items
            
            # Step 5: Identify key moments
            print("⭐ Identifying key moments...")
            key_moments = self.identify_key_moments(transcript_result.get("segments", []), full_text)
            results["key_moments"] = key_moments
            
            # Step 6: Analyze sentiment
            print("😊 Analyzing sentiment...")
            sentiment = self.analyze_sentiment(full_text)
            results["sentiment"] = sentiment
            
            # Step 7: Detect contradictions
            print("⚠️ Detecting contradictions...")
            contradictions = self.detect_contradictions(full_text)
            results["contradictions"] = contradictions
            
            # Step 8: Generate mind map
            print("🧠 Generating mind map...")
            mindmap = self.generate_mindmap(full_text, keywords)
            results["mindmap"] = mindmap
            
            # Step 9: Generate timeline
            print("📅 Generating timeline...")
            timeline = self.generate_timeline(action_items, full_text)
            results["timeline"] = timeline
            
            # Step 10: Generate summary
            print("📝 Generating summary...")
            summary = self.generate_summary(full_text)
            results["summary"] = summary
            
            # Additional metadata
            results["statistics"] = {
                "duration_seconds": transcript_result.get("segments", [])[-1].get("end", 0) if transcript_result.get("segments") else 0,
                "word_count": len(full_text.split()),
                "speaker_count": len(results["speakers"]),
                "action_item_count": len(action_items),
                "key_moment_count": len(key_moments),
                "keyword_count": len(keywords)
            }
            
            print("✅ Audio processing completed successfully!")
            return results
            
        except Exception as e:
            error_msg = f"❌ Error in audio processing: {e}"
            print(error_msg)
            return {"error": error_msg}

    def generate_podcast(self, content: str, output_path: str) -> bool:
        """Generate podcast audio from text content"""
        if not self.tts_engine:
            print("❌ TTS engine not available")
            return False
        
        try:
            print(f"🎧 Generating podcast audio...")
            
            # Prepare content for speech
            # Add pauses and improve readability
            content = content.replace('\n\n', '. ')
            content = content.replace('\n', ' ')
            content = re.sub(r'\s+', ' ', content)
            
            # Limit content length for reasonable podcast duration
            max_length = 2000
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            # Generate speech
            self.tts_engine.save_to_file(content, output_path)
            self.tts_engine.runAndWait()
            
            print(f"✅ Podcast generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Podcast generation failed: {e}")
            return False

    def save_results(self, results: Dict[str, Any], output_dir: str = "./outputs") -> Dict[str, str]:
        """Save processing results to files"""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        try:
            # Save full results as JSON
            json_path = output_path / f"audio_analysis_{timestamp}.json"
            save_json(json_path, results)
            saved_files["json"] = str(json_path)
            
            # Save transcript
            if results.get("transcript", {}).get("text"):
                transcript_path = output_path / f"transcript_{timestamp}.txt"
                save_text(transcript_path, results["transcript"]["text"])
                saved_files["transcript"] = str(transcript_path)
            
            # Save summary
            if results.get("summary"):
                summary_path = output_path / f"summary_{timestamp}.txt"
                save_text(summary_path, results["summary"])
                saved_files["summary"] = str(summary_path)
            
            # Save action items
            if results.get("action_items"):
                action_items_text = "\n".join([f"- {item}" for item in results["action_items"]])
                action_path = output_path / f"action_items_{timestamp}.txt"
                save_text(action_path, action_items_text)
                saved_files["action_items"] = str(action_path)
            
            # Save mind map
            if results.get("mindmap", {}).get("mermaid"):
                mindmap_path = output_path / f"mindmap_{timestamp}.mmd"
                save_text(mindmap_path, results["mindmap"]["mermaid"])
                saved_files["mindmap"] = str(mindmap_path)
            
            print(f"✅ Results saved to {output_path}")
            return saved_files
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    processor = SpeechProcessorEnhanced()
    
    # Test with a sample audio file (you would provide this)
    # results = processor.process_audio("sample_audio.wav")
    # print(json.dumps(results, indent=2))