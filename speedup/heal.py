"""heal.py - self-healing: distill the edited student toward the ORIGINAL model's next-token distribution.
Loss is KL(teacher || student) only: no cross-entropy on wikitext labels, so healing cannot overfit the
benchmark text and "beat" the original on it. Text comes from the healing split only."""
import time, torch, torch.nn.functional as F
import judge as J

_teacher = None
def teacher():
    global _teacher
    if _teacher is None: _teacher = J.load_original().eval()
    return _teacher

def heal(student, steps=200, batch=4, seq=128, lr=5e-5, start=0, log=print):
    x = J.split('healing')[:, :seq]; T = teacher()
    student.train(); opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=0)
    t0 = time.time()
    for s in range(steps):
        i = (start + s * batch) % (len(x) - batch); b = x[i:i + batch]
        with torch.no_grad(): tl = T(b).logits.float()
        sl = student(b).logits.float()
        loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction='batchmean') / seq
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if s % 50 == 0: log(f'  heal step {s} kl {loss.item():.4f} ({(time.time()-t0)/(s+1):.2f}s/step)')
    student.eval(); return student, start + steps * batch
