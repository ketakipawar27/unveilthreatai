import logging
import spacy

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

def analyze_caption(caption):
    """Analyze the tone, intent, and risks in a given caption using spaCy."""
    try:
        doc = nlp(caption)
        tone = "Neutral"
        intent = "Informative"
        risks = []
        for sent in doc.sents:
            if any(token.text.lower() in ['sad', 'sorry', 'hurt'] for token in sent):
                tone = "Emotional"
            if any(token.text.lower() in ['party', 'celebrate'] for token in sent):
                intent = "Celebratory"
            if any(token.text.lower() in ['sarcasm', 'joke'] for token in sent):
                intent = "Sarcastic"
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'PERSON']:
                risks.append((f"Contains {ent.label_.lower()}: {ent.text}", 0.8))
        logger.debug(f"Caption analysis: Tone={tone}, Intent={intent}, Risks={risks}")
        return tone, intent, risks
    except Exception as e:
        logger.error(f"Error in caption analysis: {str(e)}", exc_info=True)
        return "Unknown", "Unknown", []



