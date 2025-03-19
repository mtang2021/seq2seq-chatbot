from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import os
import sys
import pickle
import re
import logging
import uuid
import html
from flask_cors import CORS
from collections import deque, defaultdict
import time

# Import the chatbot classes and functions
from chatbot_models import (
    Vocabulary, Seq2seqBaseline, Seq2seqAttention, 
    predict_greedy, predict_top_p, predict_beam,
    pad_id, bos_id, eos_id, unk_id
)

# Import enhancement functions
from postprocessing import (
    post_process_response, 
    enhance_response, is_quality_response
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chatbot.log"), logging.StreamHandler()]
)
logger = logging.getLogger("chatbot")

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for all routes

# Initialize global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = None
baseline_model = None
attention_model = None
models_loaded = False

# Store conversation history
sessions = {}

# Rate limiter implementation
class RateLimiter:
    def __init__(self, limit=100, window=60):
        self.limit = limit  # Requests per window
        self.window = window  # Time window in seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip):
        current_time = time.time()
        # Clean old requests
        self.requests[client_ip] = [t for t in self.requests[client_ip] 
                                  if current_time - t < self.window]
        
        # Check if limit is reached
        if len(self.requests[client_ip]) >= self.limit:
            return False
        
        # Add current request
        self.requests[client_ip].append(current_time)
        return True

# Create rate limiter instance
rate_limiter = RateLimiter(limit=100, window=60)  # 100 requests per minute

def load_models():
    """Load vocabulary and models from saved files"""
    global vocab, baseline_model, attention_model, models_loaded
    
    # Check if models are already loaded
    if models_loaded:
        return True
    
    try:
        logger.info("Loading vocabulary and models...")
        
        # 1. Load vocabulary
        vocab_file = "processed_CMDC.pkl"
        try:
            with open(vocab_file, "rb") as f:
                all_conversations = pickle.load(f)
            logger.info(f"Successfully loaded {len(all_conversations)} conversations")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            return False
        
        vocab = Vocabulary()
        for src, tgt in all_conversations[:-100]:  # Skip the last 100 for eval
            vocab.add_words_from_sentence(src)
            vocab.add_words_from_sentence(tgt)
        logger.info(f"Vocabulary created with {vocab.num_words} words")
        
        # 2. Load baseline model
        try:
            baseline_path = "models/baseline_model.pt"
            baseline_model = Seq2seqBaseline(vocab).to(device)
            baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
            baseline_model.eval()
            logger.info("Baseline model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading baseline model: {e}")
            return False
        
        # 3. Load attention model
        try:
            attention_path = "models/attention_model.pt"
            attention_model = Seq2seqAttention(vocab).to(device)
            attention_model.load_state_dict(torch.load(attention_path, map_location=device))
            attention_model.eval()
            logger.info("Attention model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading attention model: {e}")
            return False
        
        models_loaded = True
        logger.info("All models loaded successfully")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error in load_models: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_chat_input(data):
    """Validate the input data for chat API endpoint"""
    errors = []
    
    # Check required fields
    if 'message' not in data:
        errors.append("Message is required")
        return errors
    
    message = data.get('message', '')
    decode_method = data.get('decode_method', 'greedy')
    model_type = data.get('model_type', 'attention')
    
    # Validate message
    if not isinstance(message, str):
        errors.append("Message must be a string")
    elif not message.strip():
        errors.append("Message cannot be empty")
    elif len(message) > 500:  # Set reasonable maximum
        errors.append("Message is too long (maximum 500 characters)")
    
    # Check for potentially harmful content
    if re.search(r'<script.*?>.*?</script>', message, re.IGNORECASE | re.DOTALL):
        errors.append("Message contains potentially harmful content")
    
    # Validate decode method
    if decode_method not in ['greedy', 'top-p', 'beam']:
        errors.append("Invalid decode method")
    
    # Validate model type
    if model_type not in ['baseline', 'attention']:
        errors.append("Invalid model type")
    
    # Validate parameters based on decode method
    if decode_method == 'top-p':
        try:
            temperature = float(data.get('temperature', 0.9))
            if temperature < 0.1 or temperature > 2:
                errors.append("Temperature must be between 0.1 and 2")
        except (TypeError, ValueError):
            errors.append("Temperature must be a number")
        
        try:
            top_p = float(data.get('top_p', 0.9))
            if top_p < 0.1 or top_p > 1:
                errors.append("Top-p must be between 0.1 and 1")
        except (TypeError, ValueError):
            errors.append("Top-p must be a number")
    
    elif decode_method == 'beam':
        try:
            beam_size = int(data.get('beam_size', 5))
            if beam_size < 2 or beam_size > 10:
                errors.append("Beam size must be between 2 and 10")
        except (TypeError, ValueError):
            errors.append("Beam size must be an integer")
    
    return errors

def sanitize_input(text):
    """Sanitize input to prevent injection attacks"""
    if not isinstance(text, str):
        return ""
    
    # HTML escape and remove control characters
    text = html.escape(text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = deque(maxlen=5)  # Store last 5 exchanges
    return jsonify({"session_id": session_id})

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat messages with validation and enhancements"""
    try:
        # Check rate limit
        client_ip = request.remote_addr
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        
        # Get JSON data
        data = request.json
        
        # Log the request (omit full message for privacy)
        logger.info(f"Chat request received: {data.get('model_type')}, {data.get('decode_method')}")
        
        # Validate input
        validation_errors = validate_chat_input(data)
        if validation_errors:
            logger.warning(f"Input validation failed: {validation_errors}")
            return jsonify({"error": ". ".join(validation_errors)}), 400
        
        # Ensure models are loaded
        if not load_models():
            logger.error("Failed to load models")
            return jsonify({"error": "Failed to load models"}), 500
        
        # Sanitize and extract input
        message = sanitize_input(data.get('message', ''))
        decode_method = data.get('decode_method', 'greedy')
        model_type = data.get('model_type', 'attention')
        
        # Get conversation history if available
        session_id = data.get('session_id')
        history = list(sessions.get(session_id, []))
        
        # Select model
        model = attention_model if model_type == 'attention' else baseline_model
        
        # Try to generate a model response with multiple attempts if needed
        max_attempts = 3
        attempt = 0
        response = None
            
        while attempt < max_attempts and not response:
            try:
                # Generate response based on decode method
                if decode_method == 'greedy':
                    # No special parameters for greedy search
                    raw_response = predict_greedy(model, message)
                    
                elif decode_method == 'top-p':
                    # Extract and validate parameters for top-p sampling
                    temperature = min(max(float(data.get('temperature', 0.95)), 0.1), 2.0)
                    top_p = min(max(float(data.get('top_p', 0.92)), 0.1), 1.0)
                    
                    # Generate response with top-p sampling
                    raw_response = predict_top_p(model, message, temperature=temperature, top_p=top_p)
                    
                    # For retrying with top-p, adjust temperature
                    if not is_quality_response(raw_response):
                        attempt += 1
                        temperature = min(temperature + 0.1, 1.5)
                        logger.info(f"Retrying top-p with temperature: {temperature} (attempt {attempt})")
                        continue  # Skip to next iteration with new temperature
                        
                elif decode_method == 'beam':
                    # Extract and validate parameters for beam search
                    beam_size = min(max(int(data.get('beam_size', 5)), 2), 10)
                    
                    # Generate responses with beam search
                    beam_responses = predict_beam(model, message, k=beam_size)
                    
                    # Check if we got any responses
                    if not beam_responses:
                        raw_response = ""
                    else:
                        # Try each beam response until we find a good one
                        for resp in beam_responses:
                            if is_quality_response(resp):
                                raw_response = resp
                                break
                        else:
                            # If none were good quality, use the first one
                            raw_response = beam_responses[0]
                    
                    # For retrying with beam search, adjust beam size
                    if not is_quality_response(raw_response):
                        attempt += 1
                        beam_size = min(beam_size + 1, 10)  # Increase beam size for retry
                        logger.info(f"Retrying beam search with beam size: {beam_size} (attempt {attempt})")
                        continue  # Skip to next iteration with new beam size
                else:
                    return jsonify({"error": "Invalid decode method"}), 400
                
                # Check response quality and enhance
                if is_quality_response(raw_response):
                    response = enhance_response(message, raw_response, history)
                else:
                    # If we get here, we need to retry
                    attempt += 1
                    
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}", exc_info=True)
                attempt += 1
        
        # Validate output
        if not response or len(response.strip()) == 0:
            response = "I'm sorry, I couldn't generate a proper response."
        
        # Limit response length
        if len(response) > 1000:
            response = response[:997] + "..."
        
        # Update conversation history
        if session_id:
            if session_id not in sessions:
                sessions[session_id] = deque(maxlen=5)
            sessions[session_id].append((message, response))
        
        logger.info(f"Generated response: {response[:30]}...")
        return jsonify({
            "response": response,
            "session_id": session_id
        })
            
    except Exception as e:
        logger.error(f"General error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    # Try to load models at startup
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)