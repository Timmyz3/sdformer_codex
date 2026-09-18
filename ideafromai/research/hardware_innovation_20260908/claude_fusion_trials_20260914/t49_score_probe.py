import torch
torch.manual_seed(0)
D, T, B, H, N = 32, 2, 2, 2, 16
alpha0, mismatch, single = 0.02, 0.0, 0.0
motion_a = 0.125
rho = 0.057

def make(dtype):
    q = (torch.rand(T, B, H, N, D) < rho).float()
    k = (torch.rand(B, H, T * N, D) < rho).float()
    if dtype == "ternary":
        q = q * (2 * (torch.randint(0, 2, q.shape).float()) - 1)
        k = k * (2 * (torch.randint(0, 2, k.shape).float()) - 1)
    return q, k

def sign_ste(x): return (x.sign() - x).detach() + x
def bin_ste(x):  return x.gt(0).to(x.dtype)
def tok_q(q_orig): return q_orig.permute(1, 2, 0, 3, 4).reshape(B, H, T * N, D)

def parts(q_orig, k_orig, ste):
    qe, ke = ste(tok_q(q_orig)), ste(k_orig)
    qa, ka = qe.ne(0), ke.ne(0)
    same = (qe == ke) & qa & ka
    samez = (~qa) & (~ka)
    opp = (qe == -ke) & qa & ka
    singleact = (qa ^ ka)
    return (same.float().sum(-1, keepdim=True), samez.float().sum(-1, keepdim=True),
            opp.float().sum(-1, keepdim=True), singleact.float().sum(-1, keepdim=True))

def motion(q_orig, k_orig, ste):
    ke = ste(k_orig).reshape(B, H, T, N, D)
    return (ke - ke.flip(2)).abs().sum(-1, keepdim=True).reshape(B, H, T * N, 1)

def shiftmax(s, dim, eps=1e-6):
    sh = s - s.amax(dim=dim, keepdim=True)
    num = torch.pow(2.0, sh)
    return num / torch.pow(2.0, torch.ceil(torch.log2(num.sum(dim=dim, keepdim=True).clamp_min(eps))))

def report(name, q, k, ste):
    qe_tok = ste(tok_q(q))
    S, Z, O, SA = parts(q, k, ste)
    score = S + alpha0 * Z - mismatch * O - single * SA
    score = score + motion_a * motion(q, k, ste)
    score = score / D
    # exact decomposed model: 1.02*overlap - 0.02*|K| + (0.02*D - 0.02*|Q|) + 0.125*hamming
    ka = ste(k).ne(0).float().sum(-1, keepdim=True)
    qa = qe_tok.ne(0).float().sum(-1, keepdim=True)
    model = (1.02 * S - 0.02 * ka + (0.02 * D - 0.02 * qa) + motion_a * motion(q, k, ste)) / D
    print("%-24s |O| mean=%.3f  maxdev(score vs model)=%.3e" % (name, O.mean(), (score - model).abs().max()))
    centered = score - score.mean(dim=2, keepdim=True)
    g = shiftmax(centered, dim=2)
    g_nocenter = shiftmax(score, dim=2)
    print("%-24s shiftmax(centered) vs shiftmax(unc): maxdev=%.3e" % ("", (g - g_nocenter).abs().max()))
    # how much does alpha0 matter AT ALL (centered shiftmax path)?
    score_noalpha0 = (S - mismatch * O - single * SA) + motion_a * motion(q, k, ste)
    score_noalpha0 = score_noalpha0 / D
    g2 = shiftmax(score_noalpha0 - score_noalpha0.mean(dim=2, keepdim=True), dim=2)
    print("%-24s gate change when alpha0: 0.02->0 : max=%.3e rel=%.3e" % ("", (g - g2).abs().max(), ((g-g2).abs().sum()/g.abs().sum())))

for dt in ("binary", "ternary"):
    q, k = make(dt)
    ste = sign_ste
    report(dt + " / sign_ste(ATLIF path)", q, k, ste)
    if dt == "binary":
        report("binary / binary_event_ste", q, k, bin_ste)
