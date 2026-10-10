from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent

def inspect(text):
    clean_text = re.sub(r'```[\s\S]*?```', '', text)
    long_sentences, long_paragraphs, lengths, steps = [], [], [], []
    for block in re.split(r'\n\s*\n', clean_text):
        if not block.strip() or block.lstrip().startswith(('#', '![')):
            continue
        chunks = block.splitlines() if block.lstrip().startswith(('|', '- ')) or re.match(r'^\d+\.', block) else [block]
        for chunk in chunks:
            if re.match(r'^\|[- :|]+\|$', chunk):
                continue
            numbered = bool(re.match(r'^\d+\.', chunk))
            cells = chunk.strip('|').split('|') if chunk.startswith('|') else [chunk]
            for cell in cells:
                cell = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cell)
                cell = re.sub(r'`[^`]+`', 'CODE', cell)
                cell = re.sub(r'^\s*(?:\d+\.\s*|-\s*)', '', cell)
                cell = re.sub(r'^\*Figure \d+\.\s*', '', cell).replace('*', '').strip()
                sentences = [x.strip() for x in re.split(r'(?<=[.!?])\s+', cell) if x.strip()]
                if len(sentences) > 6:
                    long_paragraphs.append({'sentences': len(sentences), 'text': cell})
                for sentence in sentences:
                    words = len(sentence.split())
                    lengths.append(words)
                    if numbered:
                        steps.append(words)
                    if words > (20 if numbered else 25):
                        long_sentences.append({'words': words, 'limit': 20 if numbered else 25, 'text': sentence})
    prose = re.sub(r'`[^`]+`|\([^)]*https?[^)]*\)', '', clean_text)
    uk = re.findall(r'\b(?:grey|colour|colours|labelled|labelling|licence|licences|programme|centre|micrometre|micrometres|metre|favourable)\b', prose, flags=re.I)
    contractions = re.findall(r"\b(?:don't|doesn't|didn't|isn't|aren't|wasn't|weren't|can't|couldn't|won't|wouldn't|shouldn't|haven't|hasn't|we're|it's|that's)\b", prose, flags=re.I)
    return {'maximum_sentence_words': max(lengths), 'maximum_numbered_step_words': max(steps), 'sentence_units_checked': len(lengths), 'numbered_steps_checked': len(steps), 'long_sentences': long_sentences, 'long_paragraphs': long_paragraphs, 'selected_british_spelling_occurrences': uk, 'contractions': contractions, 'em_dashes': text.count('\u2014')}

current = (HERE / 'kaggle_writeup.md').read_text()
before = (HERE / 'before_more_figures/kaggle_writeup.md').read_text()
result = {'verdict': 'STE-guided; not fully compliant. See ASD_STE100_review.md for the contextual review.', 'mechanical_check_status': 'PASS within the stated scope', 'scope': 'Conservative whitespace count of prose, table cells, captions and numbered steps. Headers, code, URLs and figure-number labels are excluded. This is an editing aid, not a full STE parser or dictionary checker.', 'before': inspect(before), 'after': inspect(current), 'current_sha256': hashlib.sha256(current.encode()).hexdigest()}
assert not result['after']['long_sentences']
assert not result['after']['long_paragraphs']
assert not result['after']['selected_british_spelling_occurrences']
assert not result['after']['contractions']
assert not result['after']['em_dashes']
(HERE / 'ste_style_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
