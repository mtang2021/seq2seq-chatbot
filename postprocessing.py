import re
import random
from collections import deque


def post_process_response(response):
    """Clean up and enhance model responses"""
    if not response or len(response.strip()) == 0:
        return "I'm sorry, I didn't understand that properly."
    
    # Fix common spelling errors
    spelling_fixes = {
        "mena": "mean", 
        "i m": "I'm", 
        "dont": "don't",
        "cant": "can't",
        "you re": "you're",
        "im": "I'm"
    }
    
    for typo, correction in spelling_fixes.items():
        response = re.sub(r'\b' + typo + r'\b', correction, response, flags=re.IGNORECASE)
    
    # Fix spaces before punctuation marks
    response = re.sub(r'\s+([,.!?;:])', r'\1', response)
    
    # Fix multiple spaces
    response = re.sub(r'\s{2,}', ' ', response)
    
    # Fix missing apostrophes
    response = fix_missing_apostrophes(response)
    
    # Fix spaces before punctuation marks
    response = re.sub(r'\s+([,.!?;:])', r'\1', response)
    
    # Capitalize first letter of sentences
    sentences = re.split(r'([.!?]\s*)', response)
    result = ""
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            if sentences[i]:
                sentences[i] = sentences[i][0].upper() + sentences[i][1:]
            result += sentences[i]
        if i+1 < len(sentences):
            result += sentences[i+1]
    
    # Add punctuation if missing
    if result and not result.rstrip()[-1] in ['.', '!', '?']:
        result += '.'
    
    # Expand very short responses
    word_count = len(result.split())
    if word_count <= 2 and "what" in result.lower():
        result = "I'm not sure I understand. Could you please elaborate on what you mean?"
    elif word_count <= 2 and "fine" in result.lower():
        result = "I'm glad to hear that. Is there anything I can help you with today?"
    
    return result

def enhance_response(user_input, model_response, conversation_history=None):
    """Enhance model responses based on input patterns and conversation history"""
    
    # Convert to lowercase for pattern matching
    input_lower = user_input.lower().strip()
    

    # Questions about the model
    if "who are you" in input_lower or "what are you" in input_lower:
        return "I'm a neural chatbot trained on movie dialogue data. I use a sequence-to-sequence model with attention to generate responses. How can I assist you?"
    
    # Handle very short or repetitive responses
    if len(model_response.split()) < 3 or model_response.lower().strip() == "what?":
        if "?" in user_input:
            return "That's an interesting question. Let me think about it for a moment..."
        else:
            return "I understand. Please tell me more about that."
    
    # If input seems confrontational
    confrontational = ["kidding me", "are you serious", "you're joking"]
    if any(phrase in input_lower for phrase in confrontational):
        return "I'm trying my best to understand and respond appropriately. Could you please rephrase that?"
    
    # Default: return the enhanced original response
    return post_process_response(model_response)

def fix_missing_apostrophes(response):
    """Fix missing apostrophes in common contractions"""
    import re
    
    # Fix pronouns with missing apostrophes (he s, she s, it s, etc.)
    pronoun_s_pattern = r'\b(he|she|it|that|there|here|who|what|where|this|I|you|we|they) s\b'
    response = re.sub(pronoun_s_pattern, r"\1's", response, flags=re.IGNORECASE)
    
    # Also fix other common contractions
    response = re.sub(r'\b(can|don|won|isn|aren|wasn|weren|haven|hasn|hadn|wouldn|couldn|shouldn|didn|ain) t\b', 
                     r"\1't", response, flags=re.IGNORECASE)
    
    # Fix "let s" → "let's"
    response = re.sub(r'\b(let) s\b', r"\1's", response, flags=re.IGNORECASE)
    
    # Fix will contractions (I ll, you ll, etc.)
    response = re.sub(r'\b(I|you|we|they|he|she|it|that|there|who|what) ll\b', 
                     r"\1'll", response, flags=re.IGNORECASE)
    
    # Fix would/had contractions
    response = re.sub(r'\b(I|you|we|they|he|she|it|that|who|what) d\b', 
                     r"\1'd", response, flags=re.IGNORECASE)
    
    # Fix have contractions
    response = re.sub(r'\b(I|you|we|they|could|should|would|must|might) ve\b', 
                     r"\1've", response, flags=re.IGNORECASE)
    
    # Fix are contractions
    response = re.sub(r'\b(you|we|they) re\b', r"\1're", response, flags=re.IGNORECASE)
    
    return response

def is_quality_response(response):
    """Check if a response meets quality standards"""
    # Too short
    if len(response.split()) < 3:
        return False
    
    # Too repetitive
    words = response.lower().split()
    if len(words) > 3 and len(set(words)) / len(words) < 0.5:
        return False
    
    # Contains only questions or very generic responses
    low_quality = ["what?", "i don't know", "what do you mean"]
    if response.lower().strip() in low_quality:
        return False
        
    return True