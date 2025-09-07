# enhanced_rag_app_v2.py - Enhanced RAG with Speech Processing and Advanced Features

from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/Users/skmda/OneDrive/Desktop/RAG_Project/.env")

import os
import fitz
import re
import requests
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
import streamlit as st
import tempfile
import json
import warnings
import time
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Google Gemini imports
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Audio recorder import
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
    print("✅ Audio recorder component loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Audio recorder not available: {e}")
    AUDIO_RECORDER_AVAILABLE = False
    def audio_recorder(*args, **kwargs):
        return None

# Import enhanced speech processor
try:
    from speech_processor_enhanced import SpeechProcessorEnhanced, ensure_dir, save_text, save_json
    SPEECH_PROCESSING_AVAILABLE = True
    print("✅ Enhanced speech processing library loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Speech processing not available: {e}")
    SPEECH_PROCESSING_AVAILABLE = False
    # Create dummy classes
    class SpeechProcessorEnhanced:
        def __init__(self, *args, **kwargs):
            pass
        def process_audio(self, *args, **kwargs):
            return {"error": "Speech processing not available"}
        def generate_podcast(self, *args, **kwargs):
            return False
    
    def ensure_dir(p): p.mkdir(parents=True, exist_ok=True)
    def save_text(p, t): p.write_text(t, encoding="utf-8")
    def save_json(p, o): p.write_text(json.dumps(o, indent=2), encoding="utf-8")

# Hybrid search imports
try:
    from hybrid_search import HybridSearchEngine
    HYBRID_SEARCH_AVAILABLE = True
    print("✅ Hybrid search engine loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Hybrid search not available: {e}")
    HYBRID_SEARCH_AVAILABLE = False
    class HybridSearchEngine:
        def __init__(self, *args, **kwargs):
            pass
        def add_documents(self, *args, **kwargs):
            pass
        def search(self, *args, **kwargs):
            return []

# Optional Supabase client
try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except Exception:
    create_client = None
    SUPABASE_AVAILABLE = False

# Environment variables
SUPABASE_DB_PASSWORD = os.environ.get("SUPABASE_DB_PASSWORD")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://zwtvfgptlsrmzimecpkp.supabase.co")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google Gemini API configured")
else:
    print("⚠️ Warning: No Google API key found")

llm = GenerativeModel("gemini-2.0-flash-exp") if GOOGLE_API_KEY else None
embedding_model = GenerativeModel("embedding-001") if GOOGLE_API_KEY else None

# --------- Core Embedding ---------
class GoogleEmbedder:
    def __init__(self):
        self.model = embedding_model
        self.cache = {}

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        if not GOOGLE_API_KEY or not self.model:
            print("❌ Google embedding model not available")
            return []
        
        embeddings = []
        for text in texts:
            # Simple caching to avoid re-computing embeddings
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
                continue
                
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                # Flexible extraction
                emb = []
                if isinstance(result, dict):
                    if "embedding" in result:
                        maybe = result["embedding"]
                        if isinstance(maybe, dict) and "values" in maybe:
                            emb = maybe["values"]
                        elif isinstance(maybe, list):
                            emb = maybe
                    elif "values" in result:
                        emb = result["values"]
                elif hasattr(result, "embedding"):
                    maybe = result.embedding
                    if isinstance(maybe, dict) and "values" in maybe:
                        emb = maybe["values"]
                    elif isinstance(maybe, list):
                        emb = maybe
                if not emb:
                    try:
                        emb = result["embedding"]["values"]
                    except Exception:
                        emb = []
                emb = [float(x) for x in emb] if emb else []
                self.cache[text_hash] = emb
                embeddings.append(emb)
            except Exception as e:
                print(f"Embedding error: {e}")
                default_emb = [0.0] * 768
                self.cache[text_hash] = default_emb
                embeddings.append(default_emb)
        return embeddings

embeddings = GoogleEmbedder()

# --------- Enhanced Chunking / Document Processing ---------
class SmartChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document_text: str, metadata: Dict = None) -> List[Dict]:
        content = document_text or ""
        chunks = []
        
        # Try to split by paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunk_metadata = {"chunk_index": chunk_idx}
                    if metadata:
                        chunk_metadata.update(metadata)
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    chunk_idx += 1
                current_chunk = para + "\n\n"
        
        # Add remaining content
        if current_chunk.strip():
            chunk_metadata = {"chunk_index": chunk_idx}
            if metadata:
                chunk_metadata.update(metadata)
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        # Fallback to simple chunking if no paragraphs found
        if not chunks:
            return self._simple_chunk(content, metadata)
        
        return chunks

    def _simple_chunk(self, content: str, metadata: Dict = None) -> List[Dict]:
        chunks = []
        start = 0
        idx = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end].strip()
            chunk_metadata = {"chunk_index": idx}
            if metadata:
                chunk_metadata.update(metadata)
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})
            idx += 1
            if end >= len(content):
                break
            start = end - self.overlap
        return chunks

class EnhancedDocumentProcessor:
    def __init__(self):
        self.chunker = SmartChunker()

    def extract_text_from_pdf(self, filepath: str) -> str:
        try:
            doc = fitz.open(filepath)
            full_text = ""
            for page in doc:
                try:
                    full_text += page.get_text("text") + "\n"
                except Exception:
                    continue
            doc.close()
            return full_text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract structured entities like dates, tasks, people, etc."""
        entities = {
            "dates": [],
            "tasks": [],
            "people": [],
            "deadlines": [],
            "action_items": []
        }
        
        # Simple regex-based extraction (can be enhanced with NLP libraries)
        import re
        
        # Extract dates (simple patterns)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract action items (lines starting with action words)
        action_patterns = [
            r'(?:^|\n)\s*(?:TODO|Action|Task|Follow up|Next step)[:\s]+([^\n]+)',
            r'(?:^|\n)\s*[-•]\s*([^\n]*(?:complete|finish|implement|create|develop|schedule)[^\n]*)'
        ]
        
        for pattern in action_patterns:
            entities["action_items"].extend(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
        
        return entities

    def process_file(self, filepath: str, document_type: str = "general") -> Dict[str, Any]:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            if p.suffix.lower() == ".pdf":
                full_text = self.extract_text_from_pdf(filepath)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    full_text = f.read()
            
            # Extract entities
            entities = self.extract_entities(full_text)
            
            # Create enhanced metadata
            metadata = {
                "document_type": document_type,
                "filename": p.name,
                "processed_at": datetime.now().isoformat(),
                "entities": entities
            }
            
            chunks = self.chunker.chunk(full_text, metadata)
            
            return {
                "full_text": full_text,
                "chunks": chunks,
                "entities": entities,
                "metadata": metadata
            }
        except Exception as e:
            print(f"File processing error: {e}")
            return {"full_text": "", "chunks": [], "entities": {}, "metadata": {}}

# --------- Enhanced In-Memory Store with Keyword Search ---------
class EnhancedDocumentStore:
    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.keyword_index: Dict[str, List[Tuple[str, int]]] = {}  # word -> [(doc_id, chunk_idx), ...]
        
        # Initialize hybrid search if available
        if HYBRID_SEARCH_AVAILABLE:
            self.hybrid_search = HybridSearchEngine()
        else:
            self.hybrid_search = None

    def _build_keyword_index(self, doc_id: str, chunks: List[Dict]):
        """Build keyword index for BM25-like search"""
        for i, chunk in enumerate(chunks):
            words = re.findall(r'\w+', chunk["content"].lower())
            for word in set(words):  # Use set to avoid duplicates
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                self.keyword_index[word].append((doc_id, i))

    def add_document(self, filepath: str, full_text: str, chunks: List[Dict], 
                    embeddings_list: List[Optional[List[float]]], entities: Dict = None):
        self.documents[filepath] = {
            "full_text": full_text,
            "chunks": chunks,
            "embeddings": embeddings_list,
            "entities": entities or {}
        }
        
        # Build keyword index
        self._build_keyword_index(filepath, chunks)
        
        # Add to hybrid search if available
        if self.hybrid_search:
            try:
                self.hybrid_search.add_documents([(filepath, full_text, chunks, embeddings_list)])
            except Exception as e:
                print(f"Error adding to hybrid search: {e}")

    def keyword_search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Simple BM25-like keyword search"""
        query_words = re.findall(r'\w+', query.lower())
        doc_scores = {}
        
        for word in query_words:
            if word in self.keyword_index:
                for doc_id, chunk_idx in self.keyword_index[word]:
                    key = (doc_id, chunk_idx)
                    if key not in doc_scores:
                        doc_scores[key] = 0
                    doc_scores[key] += 1
        
        # Convert to results format
        results = []
        for (doc_id, chunk_idx), score in doc_scores.items():
            if doc_id in self.documents:
                chunks = self.documents[doc_id]["chunks"]
                if chunk_idx < len(chunks):
                    results.append({
                        "source": doc_id,
                        "chunk_index": chunk_idx,
                        "content": chunks[chunk_idx]["content"],
                        "similarity": score / len(query_words)  # Normalize
                    })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]

    def hybrid_search_query(self, query_embedding: List[float], query_text: str, max_results: int = 5) -> List[Dict]:
        """Combine semantic and keyword search"""
        if self.hybrid_search:
            try:
                return self.hybrid_search.search(query_embedding, query_text, max_results)
            except Exception as e:
                print(f"Hybrid search error: {e}")
        
        # Fallback: combine semantic and keyword results
        semantic_results = self.search_similar(query_embedding, max_results)
        keyword_results = self.keyword_search(query_text, max_results)
        
        # Simple combination - could be improved with RRF (Reciprocal Rank Fusion)
        all_results = {}
        
        # Add semantic results with higher weight
        for i, result in enumerate(semantic_results):
            key = (result["source"], result["chunk_index"])
            score = result["similarity"] * 0.7 + (1.0 - i / len(semantic_results)) * 0.3
            all_results[key] = {**result, "similarity": score}
        
        # Add keyword results
        for i, result in enumerate(keyword_results):
            key = (result["source"], result["chunk_index"])
            if key in all_results:
                # Boost existing results that appear in both
                all_results[key]["similarity"] += result["similarity"] * 0.3
            else:
                score = result["similarity"] * 0.3
                all_results[key] = {**result, "similarity": score}
        
        # Sort and return top results
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        return final_results[:max_results]

    def search_similar(self, query_embedding: List[float], max_results: int = 5) -> List[Dict]:
        results = []
        for filepath, doc in self.documents.items():
            chunks = doc.get("chunks", [])
            embeddings_list = doc.get("embeddings", [])
            for i, emb in enumerate(embeddings_list):
                if not emb:
                    continue
                sim = self._cosine_similarity(query_embedding, emb)
                results.append({
                    "source": filepath, 
                    "chunk_index": i, 
                    "content": chunks[i]["content"], 
                    "similarity": sim,
                    "metadata": chunks[i].get("metadata", {})
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]

    def get_entities_dashboard(self) -> Dict[str, Any]:
        """Extract structured entities from all documents"""
        dashboard = {
            "dates": [],
            "tasks": [],
            "people": [],
            "deadlines": [],
            "action_items": [],
            "document_count": len(self.documents)
        }
        
        for doc_id, doc in self.documents.items():
            entities = doc.get("entities", {})
            for entity_type in dashboard.keys():
                if entity_type in entities:
                    dashboard[entity_type].extend([
                        {"text": item, "source": Path(doc_id).name} 
                        for item in entities[entity_type]
                    ])
        
        return dashboard

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        try:
            a = np.array(v1, dtype=float)
            b = np.array(v2, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        except Exception:
            return 0.0

# --------- Database Manager (Supabase) ---------
class DatabaseManager:
    def __init__(self, SUPABASE_URL: str, SUPABASE_KEY: str):
        self.supabase_url = SUPABASE_URL
        self.supabase_key = SUPABASE_KEY
        self.supabase = None
        self.connected = False
        if SUPABASE_AVAILABLE and create_client and SUPABASE_KEY:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                self.supabase.table('conversations').select('id').limit(1).execute()
                self.connected = True
                print("✅ Database connected successfully")
            except Exception as e:
                print(f"❌ Database connection failed: {e}")
                self.connected = False
                self.supabase = None

    def is_connected(self) -> bool:
        return self.connected and self.supabase is not None

    def save_conversation(self, session_id: str, query: str, response: str, context: str = None):
        if not self.is_connected():
            return None
        try:
            conversation_data = {
                'session_id': session_id,
                'query': query,
                'response': response,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            result = self.supabase.table('conversations').insert(conversation_data).execute()
            return result.data[0]['id'] if result.data else None
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
            return None

    def save_document_metadata(self, filepath: str, document_type: str, chunk_count: int, entities: Dict = None):
        if not self.is_connected():
            return None
        try:
            doc_data = {
                'filepath': filepath,
                'document_type': document_type,
                'chunk_count': chunk_count,
                'entities': json.dumps(entities) if entities else None,
                'processed_at': datetime.now().isoformat()
            }
            result = self.supabase.table('document_metadata').insert(doc_data).execute()
            return result.data[0]['id'] if result.data else None
        except Exception as e:
            print(f"Error saving document metadata: {str(e)}")
            return None

# --------- Enhanced Conversation Memory ---------
class ConversationMemory:
    def __init__(self, db_manager: DatabaseManager = None):
        self.history = []
        self.context_window = 15  # Increased context window
        self.db_manager = db_manager

    def add_exchange(self, query: str, response: str, context: str = None, session_id: str = None):
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "context": context
        }
        self.history.append(exchange)
        if len(self.history) > self.context_window:
            self.history = self.history[-self.context_window:]
        if self.db_manager and self.db_manager.is_connected() and session_id:
            self.db_manager.save_conversation(session_id, query, response, context)

    def get_context_string(self, max_exchanges: int = 5) -> str:
        if not self.history:
            return ""
        recent_history = self.history[-max_exchanges:]
        context_parts = []
        for exchange in recent_history:
            context_parts.append(f"Human: {exchange['query']}")
            context_parts.append(f"Assistant: {exchange['response'][:300]}...")
        return "\n".join(context_parts)

    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation"""
        if not self.history:
            return "No conversation history"
        
        topics = set()
        for exchange in self.history[-10:]:  # Last 10 exchanges
            # Simple keyword extraction
            words = re.findall(r'\w+', exchange['query'].lower())
            topics.update([w for w in words if len(w) > 4])
        
        return f"Recent topics discussed: {', '.join(list(topics)[:10])}"

# --------- Enhanced RAG Agent with Advanced Features ---------
class EnhancedRAGAgentV2:
    def __init__(self):
        self.db_manager = DatabaseManager(SUPABASE_URL, SUPABASE_KEY)
        self.processor = EnhancedDocumentProcessor()
        self.store = EnhancedDocumentStore()
        self.conversation_memory = ConversationMemory(self.db_manager)
        self.latest_doc_path: Optional[str] = None
        self.audio_results: Dict[str, Any] = {}
        self.processed_audio_files: List[str] = []
        
        # Initialize enhanced speech processor if available
        if SPEECH_PROCESSING_AVAILABLE:
            try:
                self.speech_processor = SpeechProcessorEnhanced()
                print("✅ Enhanced speech processor initialized")
            except Exception as e:
                print(f"⚠️ Speech processor initialization failed: {e}")
                self.speech_processor = None
        else:
            self.speech_processor = None

    def process_and_store(self, filepath: str, document_type: str = "general") -> str:
        try:
            result = self.processor.process_file(filepath, document_type)
            full_text = result["full_text"]
            chunks = result["chunks"]
            entities = result["entities"]
            
            if not full_text.strip():
                return f"⚠️ No text extracted from {Path(filepath).name}"
            
            embeddings_list = []
            for chunk in chunks:
                emb = embeddings.get_embedding([chunk["content"]])
                embeddings_list.append(emb[0] if emb else None)
            
            self.store.add_document(filepath, full_text, chunks, embeddings_list, entities)
            self.latest_doc_path = filepath
            
            if self.db_manager and self.db_manager.is_connected():
                self.db_manager.save_document_metadata(filepath, document_type, len(chunks), entities)
            
            # Generate processing summary
            entity_summary = []
            for key, values in entities.items():
                if values:
                    entity_summary.append(f"{len(values)} {key}")
            
            summary = f"✅ Successfully processed {len(chunks)} chunks from {Path(filepath).name}"
            if entity_summary:
                summary += f"\n📊 Extracted: {', '.join(entity_summary)}"
            
            return summary
        except Exception as e:
            return f"❌ Error processing file: {e}"

    def process_audio_file(self, audio_filepath: str, language: str = None) -> Dict[str, Any]:
        """Process audio file with enhanced features"""
        if not self.speech_processor:
            return {"error": "Speech processing not available"}
        
        try:
            print(f"🎵 Starting enhanced audio processing for: {Path(audio_filepath).name}")
            results = self.speech_processor.process_audio(audio_filepath, language)
            
            if "error" in results:
                return results
            
            self.audio_results = results
            self.processed_audio_files.append(audio_filepath)
            
            # Store transcript as a document if available
            transcript_text = results.get('transcript', {}).get('text', '')
            if transcript_text:
                chunks = self.processor.chunker.chunk(transcript_text)
                embeddings_list = []
                for chunk in chunks:
                    emb = embeddings.get_embedding([chunk["content"]])
                    embeddings_list.append(emb[0] if emb else None)
                
                # Store with audio filename
                audio_doc_path = f"audio_{Path(audio_filepath).stem}"
                self.store.add_document(audio_doc_path, transcript_text, chunks, embeddings_list)
                self.latest_doc_path = audio_doc_path
                print(f"✅ Audio transcript stored as document: {audio_doc_path}")
            
            return results
        except Exception as e:
            error_msg = f"❌ Error processing audio: {e}"
            print(error_msg)
            return {"error": error_msg}

    def process_realtime_audio(self, audio_bytes: bytes, session_id: str = "default") -> Dict[str, Any]:
        """Process real-time audio recording"""
        if not audio_bytes or not self.speech_processor:
            return {"error": "No audio data or speech processor not available"}
        
        try:
            # Save audio bytes to temporary file
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_audio_path = temp_dir / f"realtime_audio_{timestamp}.wav"
            
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
            
            # Process the audio
            results = self.process_audio_file(str(temp_audio_path))
            
            # Clean up temp file
            try:
                temp_audio_path.unlink()
            except:
                pass
            
            return results
        except Exception as e:
            return {"error": f"Error processing real-time audio: {e}"}

    def _get_doc_excerpt_for_prompt(self, filepath: Optional[str], max_chars: int = 8000) -> str:
        if not filepath:
            return ""
        entry = self.store.documents.get(filepath)
        if not entry:
            return ""
        full = entry.get("full_text", "")
        if not full:
            chunks = entry.get("chunks", [])
            return " ".join(c["content"] for c in chunks[:7])  # More chunks
        return full[:max_chars] + ("..." if len(full) > max_chars else "")

    def _is_advice_query(self, q: str) -> bool:
        qlow = q.lower()
        advice_keywords = [
            "improv", "advice", "suggest", "strategy", "how to", "recommend", "optimi", "role", "which role",
            "cover letter", "apply", "cv", "resume", "format", "better", "strength", "weakness", "fix", "edit",
            "action item", "todo", "task", "follow up", "next step", "timeline", "visualization", "keyword"
        ]
        return any(k in qlow for k in advice_keywords)

    def _should_use_hybrid_search(self, query: str) -> bool:
        """Determine if hybrid search should be used"""
        # Use hybrid search for queries that might benefit from both semantic and keyword matching
        keyword_indicators = [
            "find", "search", "list", "show me", "what", "where", "when", "who",
            "specific", "exact", "mentioned", "contains"
        ]
        return any(indicator in query.lower() for indicator in keyword_indicators)

    def chat(self, query: str, session_id: str = "default", use_web: bool = False) -> Dict[str, Any]:
        if not GOOGLE_API_KEY or not llm:
            return {
                "response": "❌ Google Gemini API not available. Please check your API key configuration.",
                "context": "API unavailable",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sources_used": []
            }
        
        try:
            doc_path = self.latest_doc_path
            doc_excerpt = self._get_doc_excerpt_for_prompt(doc_path, max_chars=8000)
            advice_needed = self._is_advice_query(query)
            context_string = self.conversation_memory.get_context_string(max_exchanges=7)
            
            # Use hybrid search for better retrieval
            relevant_docs = []
            if self._should_use_hybrid_search(query) and doc_path:
                query_emb = embeddings.get_embedding([query])
                if query_emb and query_emb[0]:
                    relevant_docs = self.store.hybrid_search_query(query_emb[0], query, max_results=5)
            
            # Include audio analysis results if available
            audio_context = ""
            if self.audio_results and not self.audio_results.get('error'):
                audio_context = f"\nAUDIO ANALYSIS RESULTS:\n"
                if self.audio_results.get('summary'):
                    audio_context += f"Summary: {self.audio_results['summary']}\n"
                if self.audio_results.get('action_items'):
                    audio_context += f"Action Items: {', '.join(self.audio_results['action_items'][:5])}\n"
                if self.audio_results.get('keywords'):
                    kw_list = [kw for kw, score in self.audio_results['keywords'][:5]]
                    audio_context += f"Key Topics: {', '.join(kw_list)}\n"
                if self.audio_results.get('speakers') and len(self.audio_results['speakers']) > 1:
                    audio_context += f"Speakers Identified: {len(self.audio_results['speakers'])}\n"

            # Enhanced system instructions
            system_instructions = (
                "You are an expert assistant with access to documents and audio analysis. "
                "Use the DOCUMENT EXCERPTS, RELEVANT CHUNKS, and AUDIO ANALYSIS for factual answers.\n"
                "- When the user asks for improvements, editing suggestions, role advice, or strategies, "
                "provide recommendations based on available content and mark each as [FROM_DOCUMENT], [FROM_AUDIO], or [INFERRED].\n"
                "- For audio content, consider action items, key moments, speaker contributions, and extracted insights.\n"
                "- Provide concise actionable steps and examples with timeline suggestions when appropriate.\n"
                "- If you identify contradictions or uncertainties in the content, highlight them.\n"
                "- If you must guess context, provide confidence level (High/Medium/Low) and evidence."
            )

            prompt_parts = [system_instructions]
            
            if doc_excerpt:
                prompt_parts.append("DOCUMENT EXCERPTS:\n" + doc_excerpt)
            
            if relevant_docs:
                relevant_content = "\n\n".join([
                    f"RELEVANT CHUNK (similarity: {doc['similarity']:.3f}):\n{doc['content']}"
                    for doc in relevant_docs[:3]
                ])
                prompt_parts.append("RELEVANT CHUNKS:\n" + relevant_content)
            
            if audio_context:
                prompt_parts.append(audio_context)
            
            if not doc_excerpt and not audio_context and not relevant_docs:
                prompt_parts.append("No document or audio content available.")
            
            if context_string:
                prompt_parts.append("Recent conversation context:\n" + context_string)
            
            prompt_parts.append("\nUSER QUESTION:\n" + query)

            if advice_needed:
                prompt_parts.append(
                    "\nUSER REQUEST TYPE: ADVICE/IMPROVEMENT. Provide:\n"
                    "1) Short diagnosis based on available content.\n"
                    "2) Top actionable improvements with source tags and timeline suggestions.\n"
                    "3) Example implementations if applicable.\n"
                    "4) Next steps or action items with priority levels."
                )

            final_prompt = "\n\n".join(prompt_parts)
            
            try:
                response = llm.generate_content(final_prompt)
                text = response.text if hasattr(response, "text") else str(response)
                clean_text = re.sub(r"\n{3,}", "\n\n", text).strip()
            except Exception as e:
                clean_text = f"❌ Error generating response with Gemini: {e}"

            self.conversation_memory.add_exchange(
                query, clean_text, 
                f"Used document: {Path(doc_path).name if doc_path else 'None'}", 
                session_id
            )

            sources_used = []
            if doc_excerpt:
                sources_used.append("📄 Uploaded document")
            if relevant_docs:
                sources_used.append(f"🔍 {len(relevant_docs)} relevant chunks")
            if self.audio_results and not self.audio_results.get('error'):
                sources_used.append("🎵 Audio analysis")

            return {
                "response": clean_text,
                "context": f"Used: {', '.join(sources_used) if sources_used else 'No sources'}",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sources_used": sources_used,
                "relevant_chunks": len(relevant_docs) if relevant_docs else 0
            }
        except Exception as e:
            return {
                "response": f"❌ Error processing query: {e}",
                "context": "",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sources_used": []
            }

    def get_audio_insights(self, session_id: str = "default") -> Dict[str, Any]:
        """Get detailed insights from processed audio with enhanced features"""
        if not self.audio_results or self.audio_results.get('error'):
            return {
                "response": "❌ No audio processed yet or processing failed. Please upload an audio file first.",
                "context": "",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sources_used": []
            }

        insights_text = "# 🎵 Enhanced Audio Analysis Results\n\n"
        
        if self.audio_results.get('summary'):
            insights_text += f"## 📝 Summary\n{self.audio_results['summary']}\n\n"
        
        if self.audio_results.get('speakers') and len(self.audio_results['speakers']) > 1:
            insights_text += f"## 👥 Speaker Analysis ({len(self.audio_results['speakers'])} speakers detected)\n"
            for speaker_id, speaker_info in self.audio_results['speakers'].items():
                duration = speaker_info.get('total_time', 0)
                insights_text += f"**Speaker {speaker_id}**: {duration:.1f}s total speaking time\n"
            insights_text += "\n"
        
        if self.audio_results.get('keywords'):
            kw_text = ", ".join([f"**{kw}** ({score:.2f})" for kw, score in self.audio_results['keywords'][:15]])
            insights_text += f"## 🔑 Key Topics & Keywords\n{kw_text}\n\n"
        
        if self.audio_results.get('action_items'):
            insights_text += "## ✅ Action Items & Tasks\n"
            for i, action in enumerate(self.audio_results['action_items'][:15], 1):
                insights_text += f"{i}. {action}\n"
            insights_text += "\n"
        
        if self.audio_results.get('key_moments'):
            insights_text += "## ⭐ Key Moments & Highlights\n"
            for moment in self.audio_results['key_moments'][:8]:
                start_time = moment['start']
                insights_text += f"**{start_time:.1f}s**: {moment['text']} (Importance: {moment['score']:.2f})\n"
            insights_text += "\n"

        if self.audio_results.get('contradictions'):
            insights_text += "## ⚠️ Contradictions & Uncertainties\n"
            for contradiction in self.audio_results['contradictions'][:5]:
                insights_text += f"- {contradiction}\n"
            insights_text += "\n"

        if self.audio_results.get('mindmap'):
            insights_text += "## 🧠 Mind Map Structure\n"
            mindmap = self.audio_results['mindmap']
            if mindmap.get('mermaid'):
                insights_text += "```mermaid\n" + mindmap['mermaid'] + "\n```\n\n"

        if self.audio_results.get('timeline'):
            insights_text += "## 📅 Suggested Timeline\n"
            timeline = self.audio_results['timeline']
            for item in timeline[:8]:
                insights_text += f"- **{item.get('timeframe', 'TBD')}**: {item.get('task', 'Task description')}\n"
            insights_text += "\n"

        return {
            "response": insights_text,
            "context": "Enhanced audio analysis insights",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "sources_used": ["🎵 Enhanced audio analysis"]
        }

    def get_entities_dashboard(self, session_id: str = "default") -> Dict[str, Any]:
        """Get structured entities dashboard"""
        dashboard = self.store.get_entities_dashboard()
        
        dashboard_text = "# 📊 Document Intelligence Dashboard\n\n"
        dashboard_text += f"**Total Documents Processed**: {dashboard['document_count']}\n\n"
        
        if dashboard['action_items']:
            dashboard_text += "## ✅ Extracted Action Items\n"
            for item in dashboard['action_items'][:10]:
                dashboard_text += f"- {item['text']} *(from {item['source']})*\n"
            dashboard_text += "\n"
        
        if dashboard['dates']:
            dashboard_text += "## 📅 Important Dates\n"
            for item in dashboard['dates'][:10]:
                dashboard_text += f"- {item['text']} *(from {item['source']})*\n"
            dashboard_text += "\n"
        
        if dashboard['deadlines']:
            dashboard_text += "## ⏰ Deadlines\n"
            for item in dashboard['deadlines'][:10]:
                dashboard_text += f"- {item['text']} *(from {item['source']})*\n"
            dashboard_text += "\n"
        
        if not any([dashboard['action_items'], dashboard['dates'], dashboard['deadlines']]):
            dashboard_text += "No structured entities found. Try processing documents with more structured content.\n"

        return {
            "response": dashboard_text,
            "context": "Document intelligence dashboard",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "sources_used": ["📊 Document intelligence"],
            "dashboard_data": dashboard
        }

    def generate_podcast_from_content(self, session_id: str = "default") -> str:
        """Generate podcast audio from processed content"""
        if not self.speech_processor:
            return "❌ Speech processing not available"
        
        if not self.latest_doc_path and not self.audio_results:
            return "❌ No content available to generate podcast"
        
        # Use summary if available, otherwise use document excerpt
        content_for_podcast = ""
        if self.audio_results and self.audio_results.get('summary'):
            content_for_podcast = self.audio_results['summary']
        elif self.latest_doc_path:
            content_for_podcast = self._get_doc_excerpt_for_prompt(self.latest_doc_path, max_chars=4000)
        
        if not content_for_podcast:
            return "❌ No suitable content found for podcast generation"
        
        # Generate podcast audio
        output_dir = Path("./outputs")
        ensure_dir(output_dir)
        podcast_path = output_dir / f"podcast_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        
        try:
            success = self.speech_processor.generate_podcast(content_for_podcast, str(podcast_path))
            if success:
                return f"✅ Podcast generated successfully: {podcast_path}"
            else:
                return "❌ Failed to generate podcast"
        except Exception as e:
            return f"❌ Error generating podcast: {e}"

    def export_content_summary(self, format_type: str = "markdown", session_id: str = "default") -> str:
        """Export summary of processed content"""
        if format_type not in ["markdown", "pdf"]:
            return "❌ Unsupported format. Use 'markdown' or 'pdf'"
        
        # Generate comprehensive summary
        summary_content = f"# Aurora AI Content Summary\n"
        summary_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Document summary
        if self.store.documents:
            summary_content += "## 📄 Processed Documents\n"
            for doc_path, doc_data in self.store.documents.items():
                filename = Path(doc_path).name
                chunk_count = len(doc_data.get('chunks', []))
                entities = doc_data.get('entities', {})
                
                summary_content += f"### {filename}\n"
                summary_content += f"- **Chunks**: {chunk_count}\n"
                
                for entity_type, items in entities.items():
                    if items:
                        summary_content += f"- **{entity_type.title()}**: {len(items)}\n"
                summary_content += "\n"
        
        # Audio summary
        if self.audio_results and not self.audio_results.get('error'):
            summary_content += "## 🎵 Audio Analysis Summary\n"
            if self.audio_results.get('summary'):
                summary_content += f"{self.audio_results['summary']}\n\n"
            
            if self.audio_results.get('action_items'):
                summary_content += "### Action Items from Audio\n"
                for item in self.audio_results['action_items'][:10]:
                    summary_content += f"- {item}\n"
                summary_content += "\n"
        
        # Conversation summary
        conv_summary = self.conversation_memory.get_conversation_summary()
        if conv_summary != "No conversation history":
            summary_content += f"## 💬 Conversation Summary\n{conv_summary}\n\n"
        
        # Save the summary
        output_dir = Path("./outputs")
        ensure_dir(output_dir)
        
        if format_type == "markdown":
            summary_path = output_dir / f"summary_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            save_text(summary_path, summary_content)
            return f"✅ Summary exported to: {summary_path}"
        elif format_type == "pdf":
            # For PDF export, you might want to use additional libraries
            return "📋 PDF export feature coming soon. Use markdown format for now."

    def get_status(self) -> Dict[str, Any]:
        return {
            "documents_in_memory": len(self.store.documents),
            "conversation_length": len(self.conversation_memory.history),
            "latest_doc": Path(self.latest_doc_path).name if self.latest_doc_path else None,
            "audio_processed": bool(self.audio_results and not self.audio_results.get('error')),
            "processed_audio_files": len(self.processed_audio_files),
            "google_gemini_available": GOOGLE_API_KEY is not None,
            "database_connected": self.db_manager.is_connected() if self.db_manager else False,
            "speech_processor_available": SPEECH_PROCESSING_AVAILABLE and self.speech_processor is not None,
            "hybrid_search_available": HYBRID_SEARCH_AVAILABLE,
            "audio_recorder_available": AUDIO_RECORDER_AVAILABLE,
            "keyword_index_size": len(self.store.keyword_index)
        }

# --------- Enhanced Streamlit UI ---------
def create_streamlit_app():
    st.set_page_config(
        page_title="Aurora AI - Advanced RAG + Speech AI", 
        page_icon="🧠", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced custom CSS
    st.markdown("""
    <style>
    .main > div {
        max-width: 100%;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .highlight-box {
        background-color: #f0f2f6;
        border-left: 4px solid #ff6b35;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if "rag_agent" not in st.session_state:
        with st.spinner("Initializing Aurora AI..."):
            st.session_state.rag_agent = EnhancedRAGAgentV2()
    rag = st.session_state.rag_agent

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().timestamp()}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("🧠 Aurora AI")
        st.markdown("*Advanced RAG + Speech AI Assistant*")
        
        # Enhanced System Status
        st.subheader("📊 System Status")
        status = rag.get_status()
        
        # Create metrics display
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", status.get("documents_in_memory", 0))
            st.metric("Conversations", status.get("conversation_length", 0))
        with col2:
            st.metric("Audio Files", status.get("processed_audio_files", 0))
            st.metric("Keywords", status.get("keyword_index_size", 0))
        
        # Feature availability
        features = [
            ("Google Gemini", status.get("google_gemini_available", False)),
            ("Speech Processing", status.get("speech_processor_available", False)),
            ("Database", status.get("database_connected", False)),
            ("Hybrid Search", status.get("hybrid_search_available", False)),
            ("Audio Recorder", status.get("audio_recorder_available", False))
        ]
        
        for feature, available in features:
            icon = "✅" if available else "❌"
            st.write(f"{icon} {feature}")

        st.divider()
        
        # Document Upload Section
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])
        document_type = st.selectbox("Document Type", 
            ["general", "technical", "research", "resume", "legal", "medical", "meeting_notes"])
        
        if uploaded_file is not None:
            if st.button("📄 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    
                    file_path = temp_dir / f"temp_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    result_msg = rag.process_and_store(str(file_path), document_type)
                    
                    if "✅" in result_msg:
                        st.success(result_msg)
                    else:
                        st.error(result_msg)

        st.divider()
        
        # Enhanced Audio Section
        st.subheader("🎵 Audio Processing")
        
        # Real-time audio recorder
        if AUDIO_RECORDER_AVAILABLE:
            st.write("**Real-time Audio Recording**")
            audio_bytes = audio_recorder(
                text="🎙️ Click to Record",
                recording_color="#e74c3c",
                neutral_color="#2ecc71",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=2.0
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎵 Process Recording", type="primary"):
                        with st.spinner("Processing real-time audio..."):
                            results = rag.process_realtime_audio(audio_bytes, st.session_state.session_id)
                            
                            if "error" in results:
                                st.error(f"❌ {results['error']}")
                            else:
                                st.success("✅ Real-time audio processed successfully!")
                                if results.get('summary'):
                                    st.info(f"📝 {results['summary'][:100]}...")
                
                with col2:
                    if st.button("💾 Save Recording"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"recorded_audio_{timestamp}.wav"
                        with open(filename, "wb") as f:
                            f.write(audio_bytes)
                        st.success(f"✅ Saved as {filename}")
        else:
            st.warning("⚠️ Audio recorder not available. Install audio-recorder-streamlit.")
        
        st.write("**Audio File Upload**")
        uploaded_audio = st.file_uploader("Choose an audio file", 
            type=["mp3", "wav", "m4a", "flac", "ogg"])
        audio_language = st.selectbox("Audio Language", 
            ["auto-detect", "en", "hi", "es", "fr", "de", "it", "pt", "ru", "ja", "ko"])
        
        if uploaded_audio is not None:
            if st.button("🎵 Process Audio File", type="primary"):
                with st.spinner("Processing audio file (this may take several minutes)..."):
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    
                    audio_extension = uploaded_audio.name.split('.')[-1]
                    temp_audio_path = temp_dir / f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{audio_extension}"
                    
                    with open(temp_audio_path, "wb") as f:
                        f.write(uploaded_audio.getvalue())
                    
                    language = None if audio_language == "auto-detect" else audio_language
                    audio_results = rag.process_audio_file(str(temp_audio_path), language)
                    
                    if "error" in audio_results:
                        st.error(f"❌ {audio_results['error']}")
                    else:
                        st.success("✅ Audio processed successfully!")
                        
                        # Enhanced results display
                        if audio_results.get('summary'):
                            with st.expander("📝 Quick Summary"):
                                st.write(audio_results['summary'])
                        
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            if audio_results.get('action_items'):
                                st.metric("Action Items", len(audio_results['action_items']))
                            if audio_results.get('speakers'):
                                st.metric("Speakers", len(audio_results['speakers']))
                        
                        with metrics_col2:
                            if audio_results.get('keywords'):
                                st.metric("Keywords", len(audio_results['keywords']))
                            if audio_results.get('key_moments'):
                                st.metric("Key Moments", len(audio_results['key_moments']))
                    
                    try:
                        temp_audio_path.unlink()
                    except:
                        pass

        st.divider()
        
        # Enhanced Quick Tools Section
        st.subheader("🚀 Advanced Tools")
        
        # Row 1
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Smart Analysis"):
                with st.spinner("Performing smart analysis..."):
                    if not rag.latest_doc_path and not rag.audio_results:
                        st.warning("⚠️ No content to analyze. Upload content first.")
                    else:
                        analysis_query = "Perform a comprehensive analysis of the available content. Identify key themes, action items, insights, and provide strategic recommendations with timeline suggestions."
                        outcome = rag.chat(analysis_query, st.session_state.session_id)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": outcome["response"], 
                            "sources": outcome["sources_used"]
                        })
        
        with col2:
            if st.button("🎵 Audio Insights"):
                with st.spinner("Extracting enhanced insights..."):
                    outcome = rag.get_audio_insights(st.session_state.session_id)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": outcome["response"], 
                        "sources": outcome["sources_used"]
                    })
        
        # Row 2
        col3, col4 = st.columns(2)
        with col3:
            if st.button("📊 Entity Dashboard"):
                with st.spinner("Building dashboard..."):
                    outcome = rag.get_entities_dashboard(st.session_state.session_id)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": outcome["response"], 
                        "sources": outcome["sources_used"]
                    })
        
        with col4:
            if st.button("📝 Export Summary"):
                with st.spinner("Generating summary..."):
                    result = rag.export_content_summary("markdown", st.session_state.session_id)
                    st.info(result)
        
        # Row 3
        col5, col6 = st.columns(2)
        with col5:
            if st.button("🎧 Generate Podcast", type="secondary"):
                with st.spinner("Generating podcast..."):
                    result = rag.generate_podcast_from_content(st.session_state.session_id)
                    st.info(result)
        
        with col6:
            if st.button("🔍 Keyword Search"):
                search_query = st.text_input("Search keywords:", key="keyword_search")
                if search_query:
                    with st.spinner("Searching..."):
                        results = rag.store.keyword_search(search_query, max_results=5)
                        if results:
                            search_response = "## 🔍 Keyword Search Results\n\n"
                            for i, result in enumerate(results, 1):
                                search_response += f"**{i}.** {result['content'][:200]}...\n"
                                search_response += f"*Source: {Path(result['source']).name}, Relevance: {result['similarity']:.3f}*\n\n"
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": search_response, 
                                "sources": ["🔍 Keyword search"]
                            })
                        else:
                            st.warning("No results found for your search.")

        st.divider()
        
        # Enhanced Processed Content Summary
        st.subheader("📚 Content Overview")
        if rag.store.documents:
            for i, (path, doc_data) in enumerate(list(rag.store.documents.items())[:5]):
                if path.startswith("audio_"):
                    icon = "🎵"
                    name = path
                else:
                    icon = "📄"
                    name = Path(path).name
                
                chunks = len(doc_data.get('chunks', []))
                entities = doc_data.get('entities', {})
                entity_count = sum(len(items) for items in entities.values())
                
                st.write(f"{icon} **{name}** ({chunks} chunks, {entity_count} entities)")
            
            if len(rag.store.documents) > 5:
                st.caption(f"... and {len(rag.store.documents) - 5} more documents")
        else:
            st.caption("No content processed yet.")
        
        # Recent audio files
        if rag.processed_audio_files:
            st.write("**Recent Audio Files:**")
            for audio_file in rag.processed_audio_files[-3:]:
                st.caption(f"🎵 {Path(audio_file).name}")

    # Main Chat Interface
    st.title("💬 Chat with Aurora AI")
    st.caption("Ask questions about your documents, get audio insights, request analysis and improvements, or search for specific information!")

    # Enhanced features info
    with st.expander("✨ Features Highlights"):
        st.markdown("""
        - **🎙️ Real-time Audio Recording**: Record audio directly in the browser
        - **🔍 Hybrid Search**: Combines semantic and keyword search for better results
        - **👥 Multi-speaker Diarization**: Identify and analyze multiple speakers
        - **📊 Entity Dashboard**: Extract structured data (dates, tasks, people)
        - **⚠️ Contradiction Detection**: Identify inconsistencies in content
        - **📅 Timeline Visualization**: Suggested timelines for action items
        - **🧠 Enhanced Mind Maps**: Better visualization of content structure
        - **💾 Export Summaries**: Download comprehensive content summaries
        """)

    # Display chat messages with enhanced formatting
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if message.get("sources"):
                with st.expander("📚 Sources Used"):
                    for s in message["sources"]:
                        st.write(s)

    # Enhanced chat input with suggestions
    if prompt := st.chat_input("Ask anything about your content (documents, audio, improvements, analysis, search)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = rag.chat(prompt, st.session_state.session_id, use_web=False)
                st.markdown(res["response"])
                
                # Enhanced source display
                if res.get("sources_used"):
                    with st.expander("📚 Sources Used"):
                        for s in res["sources_used"]:
                            st.write(s)
                        if res.get("relevant_chunks", 0) > 0:
                            st.info(f"🔍 Used {res['relevant_chunks']} relevant chunks from hybrid search")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": res["response"], 
                    "sources": res.get("sources_used", [])
                })

    # Quick action suggestions
    if not st.session_state.messages:
        st.markdown("### 💡 Try these example queries:")
        suggestions = [
            "Analyze the uploaded document and provide key insights",
            "What are the main action items from the audio?",
            "Show me a timeline for completing the tasks",
            "Extract all dates and deadlines from the content",
            "Find contradictions or uncertainties in the material",
            "Generate a summary with mindmap visualization"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    # Simulate clicking the suggestion
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.rerun()

    # Footer with enhanced tips
    st.markdown("---")
    st.markdown(
        "💡 **Enhanced Tips**: Upload documents for analysis • Record audio directly in browser • "
        "Process audio files with speaker diarization • Ask for improvements and strategic recommendations • "
        "Use keyword search for specific information • Generate podcasts and export summaries • "
        "View entity dashboards for structured data extraction"
    )

if __name__ == "__main__":
    create_streamlit_app()