"""Module for analyzing conversational flows in video transcripts."""
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
def download_nltk_data():
    """Download required NLTK data if it's not already available."""
    resources = [
        'punkt',
        'averaged_perceptron_tagger'
    ]

    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"NLTK resource '{resource}' is already available")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=False)
            logger.info(f"Successfully downloaded {resource}")

# Download required NLTK data
download_nltk_data()

@dataclass
class ConversationalInsights:
    """Container for the results of conversational analysis."""
    summary: str
    sarcasm_detected: List[Dict[str, str]]
    slang_terms: List[Dict[str, str]]
    top_adjectives: List[Tuple[str, int]]
    transcript_snippet: str

class ConversationalAnalyzer:
    """Analyzes transcripts for conversational flows including sarcasm, slang, and word usage."""

    def __init__(self):
        """Initialize the conversational analyzer."""
        # Common slang terms to identify in transcripts
        self.slang_dictionary = {
            # Internet/Social Media Slang
            "lol": "laughing out loud",
            "rofl": "rolling on floor laughing",
            "lmao": "laughing my a** off",
            "ngl": "not gonna lie",
            "tbh": "to be honest",
            "fyi": "for your information",
            "imo": "in my opinion",
            "idk": "I don't know",
            "idc": "I don't care",
            "btw": "by the way",
            "smh": "shaking my head",
            "irl": "in real life",
            "tfw": "that feeling when",
            "ftw": "for the win",
            "gg": "good game",
            "dm": "direct message",
            "sus": "suspicious",

            # TikTok Specific Slang
            "fyp": "For You Page",
            "pov": "point of view",
            "ib": "inspired by",
            "dc": "dance credit",
            "fit": "outfit",
            "dnc": "did not consume",
            "fl": "for later",
            "mid": "mediocre",
            "fr": "for real",
            "gyat": "damn (exclamation of surprise)",
            "ate": "did really well",
            "slay": "did exceptionally well",
            "moots": "mutuals (mutual followers)",
            "szn": "season",
            "rizz": "charisma or charm",
            "bussin": "really good",
            "no cap": "no lie",
            "cap": "a lie",
            "heather": "desirable person",
            "caught in 4k": "caught doing something",
            "living rent free": "constantly thinking about",
            "main character": "someone who acts like they're the protagonist",
            "ratio": "when a reply gets more likes than the original post",
            "cheugy": "uncool or outdated",
            "a vibe": "something enjoyable or cool",
            "simp": "someone who shows excessive affection",
            "understood the assignment": "did well at a task",
        }

        # Patterns for identifying sarcasm
        self.sarcasm_patterns = [
            r"(?i)(yeah|sure|right|totally|obviously|clearly|of course).*(eye roll|rolling my eyes|smh|not)",
            r"(?i)(oh|ah|wow|gee|golly) (really|wow|great|nice|wonderful|fantastic)",
            r"(?i)(sarcasm|sarcastic|being sarcastic|just kidding|jk|kidding|j/k)",
            r"(?i)(air quotes|quote unquote|so-called|so called)",
            r"(?i)(thanks|thank you) (a lot|so much).*(not|didn't|don't|won't|can't)",
            r"(?i)what a (surprise|shocker)",
            r"(?i)(you don't say|no kidding|who would have thought|shock horror)",
            r"(?i)(because|cause|cos) that('s| is) (totally|exactly) what (we|I|they) needed",
            r"(?i)good (luck|job) with that",
            r"(?i)well isn't that (special|nice|great)",
            r"\*\*(.*?)\*\*",  # Text between asterisks often denotes sarcasm
            r"/s",  # Common internet marker for sarcasm
            r"(?i)i'm (so|totally|absolutely) (impressed|shocked|surprised)",
            r"(?i)(oh|ah) (how|what) (wonderful|lovely|nice|great)",
        ]

        logger.info("Conversational analyzer initialized with slang dictionary and sarcasm patterns")

    def analyze_transcript(self, full_text: str, segments: List[Dict]) -> ConversationalInsights:
        """
        Analyze a transcript for conversational flows.

        Args:
            full_text: The complete transcript text
            segments: List of transcript segments with timestamps

        Returns:
            ConversationalInsights: Results of the conversational analysis
        """
        logger.info("Starting conversational analysis")

        # Find slang terms in the transcript
        slang_findings = self._identify_slang(full_text, segments)
        logger.info(f"Found {len(slang_findings)} slang terms")

        # Detect sarcasm in the transcript
        sarcasm_findings = self._detect_sarcasm(segments)
        logger.info(f"Detected {len(sarcasm_findings)} potential instances of sarcasm")

        # Extract adjectives from the transcript
        top_adjectives = self._extract_adjectives(full_text)
        logger.info(f"Extracted top {len(top_adjectives)} adjectives")

        # Generate a summary of the content
        summary = self._generate_summary(full_text, slang_findings, sarcasm_findings, top_adjectives)
        logger.info("Generated video summary")

        # Extract a representative snippet
        snippet = self._extract_snippet(segments, slang_findings, sarcasm_findings)
        logger.info("Extracted representative transcript snippet")

        return ConversationalInsights(
            summary=summary,
            sarcasm_detected=sarcasm_findings,
            slang_terms=slang_findings,
            top_adjectives=top_adjectives[:5],  # Limit to top 5 adjectives
            transcript_snippet=snippet
        )

    def _identify_slang(self, text: str, segments: List[Dict]) -> List[Dict[str, str]]:
        """
        Identify slang terms in the transcript.

        Args:
            text: The complete transcript text
            segments: List of transcript segments with timestamps

        Returns:
            List of dictionaries with slang terms and their meanings
        """
        found_slang = []
        lower_text = text.lower()

        # Look for each slang term in the text
        for slang, meaning in self.slang_dictionary.items():
            # Use word boundaries to match whole words only
            pattern = r'\b' + re.escape(slang) + r'\b'
            matches = list(re.finditer(pattern, lower_text))

            for match in matches:
                # Find which segment this slang occurs in
                timestamp = 0
                for segment in segments:
                    segment_lower = segment["text"].lower()
                    if slang in segment_lower:
                        timestamp = segment["start"]
                        break

                found_slang.append({
                    "term": slang,
                    "meaning": meaning,
                    "timestamp": timestamp,
                })

        return found_slang

    def _detect_sarcasm(self, segments: List[Dict]) -> List[Dict[str, str]]:
        """
        Detect potential sarcasm in the transcript segments.

        Args:
            segments: List of transcript segments with timestamps

        Returns:
            List of dictionaries with sarcastic phrases and their timestamps
        """
        sarcasm_findings = []

        for segment in segments:
            text = segment["text"]

            for pattern in self.sarcasm_patterns:
                matches = re.finditer(pattern, text)

                for match in matches:
                    sarcasm_findings.append({
                        "phrase": match.group(0),
                        "context": text,
                        "timestamp": segment["start"]
                    })

        return sarcasm_findings

    def _extract_adjectives(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract and count adjectives from the transcript.

        Args:
            text: The complete transcript text

        Returns:
            List of tuples containing (adjective, count)
        """
        try:
            # Tokenize the text
            tokens = word_tokenize(text)

            # Get POS tags
            tagged_tokens = pos_tag(tokens)

            # Extract adjectives (JJ, JJR, JJS tags)
            adjectives = [word.lower() for word, tag in tagged_tokens if tag.startswith('JJ')]

            # Count occurrences
            adjective_counts = Counter(adjectives)

            # Return most common adjectives
            return adjective_counts.most_common(10)  # Get top 10 adjectives
        except Exception as e:
            logger.error(f"Error extracting adjectives: {str(e)}")
            return []  # Return empty list on error

    def _generate_summary(self, text: str, slang: List[Dict], sarcasm: List[Dict],
                          adjectives: List[Tuple[str, int]]) -> str:
        """
        Generate a two-line summary of the video content.

        Args:
            text: The complete transcript text
            slang: List of identified slang terms
            sarcasm: List of identified sarcastic phrases
            adjectives: List of extracted adjectives

        Returns:
            A two-line summary string
        """
        try:
            # Use a simple extractive approach for the summary
            # Take first 100 characters and ensure it ends with a full sentence
            if not text or text == "Analysis failed":
                return "Analysis failed. No transcript was available to summarize."

            first_part = text[:min(100, len(text))]
            if len(text) > 100:
                # Find the last sentence end
                last_period = max([first_part.rfind('.'), first_part.rfind('!'), first_part.rfind('?')])
                if last_period > 0:
                    first_part = first_part[:last_period+1]

            # Create the tone description based on findings
            tone = "neutral"
            if len(sarcasm) > 0:
                tone = "sarcastic"
            elif len(adjectives) > 0 and adjectives[0][1] > 2:  # If top adjective appears more than twice
                tone = f"{adjectives[0][0]}"  # Use the most common adjective

            slang_text = ""
            if len(slang) > 0:
                slang_terms = [item["term"] for item in slang[:2]]
                slang_text = f" using slang like {', '.join(slang_terms)}"

            second_part = f"The video has a {tone} tone{slang_text} and focuses on "

            # Extract main subject from text (simple approach - take first 2-3 nouns)
            try:
                tokens = word_tokenize(text[:200])  # Look at first 200 chars for subjects
                tagged_tokens = pos_tag(tokens)
                nouns = [word for word, tag in tagged_tokens if tag.startswith('NN')][:3]
            except Exception:
                nouns = []

            if nouns:
                second_part += ', '.join(nouns) + "."
            else:
                second_part += "the presented content."

            return f"{first_part} {second_part}"
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Content summary could not be generated due to an error."

    def _extract_snippet(self, segments: List[Dict], slang: List[Dict],
                          sarcasm: List[Dict]) -> str:
        """
        Extract a representative snippet from the transcript.

        Args:
            segments: List of transcript segments with timestamps
            slang: List of identified slang terms
            sarcasm: List of identified sarcastic phrases

        Returns:
            A representative snippet from the transcript
        """
        # Prefer segments with slang or sarcasm
        interesting_timestamps = []

        # Add timestamps from slang findings
        for item in slang:
            interesting_timestamps.append(item["timestamp"])

        # Add timestamps from sarcasm findings
        for item in sarcasm:
            interesting_timestamps.append(item["timestamp"])

        # If we have interesting timestamps, find the segment
        if interesting_timestamps:
            # Get the most common timestamp (in case several slang/sarcasm instances are in the same segment)
            timestamp_counts = Counter(interesting_timestamps)
            most_common_timestamp = timestamp_counts.most_common(1)[0][0]

            # Find the segment with this timestamp
            for segment in segments:
                if segment["start"] == most_common_timestamp:
                    return segment["text"]

        # Fallback: return the middle segment or the longest one
        if segments:
            middle_idx = len(segments) // 2
            longest_segment = max(segments, key=lambda x: len(x["text"]))

            # Prefer the longest segment if it's significantly longer
            if len(longest_segment["text"]) > len(segments[middle_idx]["text"]) * 1.5:
                return longest_segment["text"]
            else:
                return segments[middle_idx]["text"]

        return "No transcript available."
