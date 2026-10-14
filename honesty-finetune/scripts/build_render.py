def render(messages, gen=False):
    s = ""
    for m in messages:
        if m["role"] == "user":
            s += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
        else:
            s += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{m['content']}<|im_end|>\n"
    if gen:
        s += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return s
