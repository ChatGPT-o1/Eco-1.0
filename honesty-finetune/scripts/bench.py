import time, torch, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad/Qwen3.5-2B"
torch.set_num_threads(4)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16)
print("loaded in %.1fs" % (time.time() - t0), type(model).__name__, flush=True)
names = collections.Counter(n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear))
print("linear leaf names:", dict(names), flush=True)
msgs = [{"role": "user", "content": "What is the capital of Australia?"}]
t0 = time.time()
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**ids, max_new_tokens=40, do_sample=False)
dt = time.time() - t0
print("gen 40 tok in %.1fs (%.2f tok/s)" % (dt, 40 / dt), flush=True)
print("OUTPUT:", repr(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)), flush=True)
x = torch.randint(0, 1000, (1, 300))
model.train()
t0 = time.time()
loss = model(input_ids=x, labels=x).loss
loss.backward()
print("fwd+bwd 300 tok (full, no LoRA) in %.1fs" % (time.time() - t0), flush=True)
