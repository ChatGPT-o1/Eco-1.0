import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
SP = "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad"
torch.set_num_threads(4); torch.manual_seed(0)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]
def run(name, lora_bf16=False, top_only=False, batch=1, seqlen=210, iters=2):
    model = AutoModelForCausalLM.from_pretrained(f"{SP}/Qwen3.5-2B", dtype=torch.bfloat16)
    kw = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=TARGETS)
    if top_only:
        kw["layers_to_transform"] = list(range(12, 24))
    model = get_peft_model(model, LoraConfig(**kw))
    if lora_bf16:
        for n, p in model.named_parameters():
            if p.requires_grad: p.data = p.data.to(torch.bfloat16)
    model.train()
    x = torch.randint(0, 1000, (batch, seqlen))
    times = []
    for i in range(iters + 1):
        t0 = time.time()
        out = model(input_ids=x, labels=x, use_cache=False)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        if i > 0: times.append(time.time() - t0)
    per_ex = sum(times) / len(times) / batch
    print(f"{name}: {per_ex:.2f} s per example (batch={batch}, seq={seqlen})", flush=True)
    del model
which = sys.argv[1:]
if "base" in which: run("baseline")
if "bf16" in which: run("lora-bf16", lora_bf16=True)
if "top" in which: run("top12-layers-only", top_only=True)
if "batch4" in which: run("batch4", batch=4)
