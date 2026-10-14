import json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad")
from build_render import render

SP = "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad"
M = f"{SP}/Qwen3.5-2B"
ADAPTER = f"{SP}/lora_honesty_v2"
HIST = f"{SP}/chat_v2_history.json"

msg = sys.argv[1] if len(sys.argv) > 1 else ""
if msg == "--reset":
    if os.path.exists(HIST):
        os.remove(HIST)
    print("history cleared")
    sys.exit(0)

history = json.load(open(HIST)) if os.path.exists(HIST) else []
history.append({"role": "user", "content": msg})

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(M)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
eos = set([tok.convert_tokens_to_ids("<|im_end|>")])
eos_cfg = model.generation_config.eos_token_id
eos |= set(eos_cfg if isinstance(eos_cfg, list) else [eos_cfg])

enc = tok(render(history, gen=True), return_tensors="pt", add_special_tokens=False)
t0 = time.time()
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=400, do_sample=False, repetition_penalty=1.05,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=list(eos))
gen = out[0][enc["input_ids"].shape[1]:].tolist()
reply = tok.decode(gen, skip_special_tokens=True).strip()
history.append({"role": "assistant", "content": reply})
json.dump(history, open(HIST, "w"), ensure_ascii=False, indent=1)
print(reply)
print(f"\n[{len(gen)} tokens in {time.time()-t0:.0f}s]")
