from difflib import SequenceMatcher


def candidate(source, text, producer, rank=0, truncated=False):
    flags = []
    if not text.strip():
        flags.append("empty")
    if truncated and producer!='copy':
        flags.append("truncated")
    if producer!='copy' and any(t in text for t in ("<s>","</s>","<pad>","<think>","</think>","<|im_start|>","<|im_end|>","<|endoftext|>")):
        flags.append("special_token_or_rationale")
    if producer!='copy' and (text.lstrip().startswith(('```','Câu cần kiểm tra:', 'Câu sửa:', 'Giải thích:'))):
        flags.append('prompt_echo_or_rationale')
    return dict(text_raw=text, producer_id=producer, rank_in_producer=rank,
                is_copy=producer=="copy", valid=not flags, flags=flags, common_logprob=None,
                independent_agreement=1, edit_distance_ratio=1-SequenceMatcher(None,source,text).ratio())


def select_s0(candidates):
    valid = [c for c in candidates if c["valid"]]
    return next((c for c in valid if not c["is_copy"]), next(c for c in valid if c["is_copy"]))
