import glob, json, numpy as np, struct, sys

FILES = sorted(glob.glob("local-model/mlx-community/Qwen3.5-35B-A3B-4bit/main/*.safetensors"))
TENS = {}  # name -> (path, base, s, e, dtype, shape)
for F in FILES:
    n = int.from_bytes(open(F,'rb').read(8),'little')
    hdr = json.loads(open(F,'rb').read(8+n)[8:])
    for k,v in hdr.items():
        if k.startswith('__'): continue
        s,e = v['data_offsets']
        TENS[k] = (F, 8+n, s, e, v['dtype'], tuple(v['shape']))

def bf16_to_fp32(u16):
    u = u16.astype(np.uint32) << 16
    return u.view(np.float32)

def get(name):
    F,base,s,e,dt,shape = TENS[name]
    raw = open(F,'rb').read(base+e)[base+s: base+e]
    if dt == 'U32':
        return np.frombuffer(raw, dtype=np.uint32).reshape(shape)
    if dt == 'BF16':
        return bf16_to_fp32(np.frombuffer(raw, dtype=np.uint16).reshape(shape))
    if dt == 'F32':
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    raise Exception(dt)

def dequant(w_u32, scales, biases, N):
    M = w_u32.shape[0]
    raw = np.frombuffer(w_u32.tobytes(), dtype=np.uint8).reshape(M, N//2)
    low = (raw & 0xF).astype(np.float32)
    high = (raw >> 4).astype(np.float32)
    nib = np.empty((M, N), dtype=np.float32)
    nib[:, 0::2] = low
    nib[:, 1::2] = high
    num_groups = N // 64
    g = np.arange(N) // 64
    sc = scales[:, g].astype(np.float32)
    bi = biases[:, g].astype(np.float32)
    return nib * sc + bi

def rmsnorm(x, w, eps=1e-6):
    ms = np.mean(x*x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * w

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)

L=128; KV=16; VH=32; convDim=8192; 
P="language_model.model.layers.0."
# load weights
embed = np.fromfile("/tmp/qwen_raw_embed.bin", dtype=np.float16).astype(np.float32).reshape(5,2048)
in_ln = get(P+"input_layernorm.weight").astype(np.float32)
post_ln = get(P+"post_attention_layernorm.weight").astype(np.float32) if (P+"post_attention_layernorm.weight") in hdr else None
# in_proj
qkv_u = get(P+"linear_attn.in_proj_qkv.weight"); qkv_s=get(P+"linear_attn.in_proj_qkv.scales"); qkv_b=get(P+"linear_attn.in_proj_qkv.biases")
Wqkv = dequant(qkv_u, qkv_s, qkv_b, 2048)
z_u=get(P+"linear_attn.in_proj_z.weight"); z_s=get(P+"linear_attn.in_proj_z.scales"); z_b=get(P+"linear_attn.in_proj_z.biases")
Wz = dequant(z_u,z_s,z_b,2048)
a_u=get(P+"linear_attn.in_proj_a.weight"); a_s=get(P+"linear_attn.in_proj_a.scales"); a_b=get(P+"linear_attn.in_proj_a.biases")
Wa = dequant(a_u,a_s,a_b,2048)
b_u=get(P+"linear_attn.in_proj_b.weight"); b_s=get(P+"linear_attn.in_proj_b.scales"); b_b=get(P+"linear_attn.in_proj_b.biases")
Wb = dequant(b_u,b_s,b_b,2048)
A_log = get(P+"linear_attn.A_log").astype(np.float32)
dt_bias = get(P+"linear_attn.dt_bias").astype(np.float32)
conv_w = get(P+"linear_attn.conv1d.weight").reshape(8192,4).astype(np.float32)
norm_w = get(P+"linear_attn.norm.weight").astype(np.float32)
out_u=get(P+"linear_attn.out_proj.weight"); out_s=get(P+"linear_attn.out_proj.scales"); out_b=get(P+"linear_attn.out_proj.biases")
Wout = dequant(out_u,out_s,out_b,4096)

# GDN
t=5
normed = rmsnorm(embed, in_ln)
mixed = (Wqkv @ normed.T).T
zz = (Wz @ normed.T).T
aa = (Wa @ normed.T).T
bb = (Wb @ normed.T).T
S = np.zeros((VH,L,L))
conv_state = np.zeros((3,convDim))
out = np.zeros((t,VH*L))
input_scale = 1.0/np.sqrt(L)
for tt in range(t):
    mx = mixed[tt]
    window = np.stack([conv_state[0],conv_state[1],conv_state[2],mx],0)  # [4,8192]
    acc = (conv_w * window.T).sum(1)
    qkv = acc/(1+np.exp(-acc))
    conv_state[0]=conv_state[1]; conv_state[1]=conv_state[2]; conv_state[2]=mx
    for h in range(VH):
        grp=h//2
        qc=grp*L; kc=KV*L+grp*L; vc=2*KV*L+h*L
        qv=qkv[qc:qc+L]; kv=qkv[kc:kc+L]; vv=qkv[vc:vc+L]
        qv=qv/(np.linalg.norm(qv)+1e-12)
        kv=kv/(np.linalg.norm(kv)+1e-12)
        gamma=np.exp(-np.exp(A_log[h])*softplus(aa[tt,h]+dt_bias[h]))
        beta=1/(1+np.exp(-bb[tt,h]))
        r=gamma*(S[h].T@kv)
        u=beta*(vv-r)
        S[h]=gamma*S[h]+np.outer(kv,u)
        o=S[h].T@qv
        zg=zz[tt,h*L:(h+1)*L]
        silu_z=zg/(1+np.exp(-zg))
        inv=1/np.sqrt(np.mean(o*o)+1e-6)
        out[tt,h*L:(h+1)*L]=silu_z*o*input_scale*inv*norm_w
gdnProj = (Wout @ out.T).T
projout_swift = np.fromfile("/tmp/qwen_raw_projout0.bin", dtype=np.float16).astype(np.float32).reshape(5,2048)
np.save("/tmp/np_gdnProj.npy", gdnProj)
print("GDN gdnProj maxabs diff vs swift:", np.max(np.abs(gdnProj-projout_swift)))
print("  numpy gdnProj stats:", gdnProj.min(), gdnProj.max(), gdnProj.mean())
print("  swift gdnProj stats:", projout_swift.min(), projout_swift.max(), projout_swift.mean())

# Full layer 0: residual + gdn, then MoE on post_attn norm
h1 = embed + gdnProj
moe_in = rmsnorm(h1, post_ln) if post_ln is not None else rmsnorm(h1, in_ln)
# shared expert
sg_u=get(P+"mlp.shared_expert.gate_proj.weight"); sg_s=get(P+"mlp.shared_expert.gate_proj.scales"); sg_b=get(P+"mlp.shared_expert.gate_proj.biases")
Wsg=dequant(sg_u,sg_s,sg_b,2048)
su_u=get(P+"mlp.shared_expert.up_proj.weight"); su_s=get(P+"mlp.shared_expert.up_proj.scales"); su_b=get(P+"mlp.shared_expert.up_proj.biases")
Wsu=dequant(su_u,su_s,su_b,2048)
sd_u=get(P+"mlp.shared_expert.down_proj.weight"); sd_s=get(P+"mlp.shared_expert.down_proj.scales"); sd_b=get(P+"mlp.shared_expert.down_proj.biases")
Wsd=dequant(sd_u,sd_s,sd_b,512)
sgate=dequant(get(P+"mlp.shared_expert_gate.weight"),get(P+"mlp.shared_expert_gate.scales"),get(P+"mlp.shared_expert_gate.biases"),2048)  # [1,2048]
shared = (Wsd @ (np.expand_dims(np.maximum(Wsg@moe_in.T,0),0)*np.expand_dims(Wsu@moe_in.T,0)).reshape(5,-1).T) if False else None
# proper:
sh_gate = 1.0/(1+np.exp(-(sgate @ moe_in.T).T))  # [5,1]
gs = Wsg@moe_in.T
act = (gs/(1+np.exp(-gs)))*(Wsu@moe_in.T)  # SiLU gate * up  [512,5]
shared = (Wsd @ act).T  # [5,2048]
# routed MoE
gate_w = dequant(get(P+"mlp.gate.weight"),get(P+"mlp.gate.scales"),get(P+"mlp.gate.biases"),2048)  # [256,2048]
router = (gate_w @ moe_in.T).T / np.sqrt(2048.0)  # [5,256]
probs = np.exp(router - router.max(-1,keepdims=True)); probs/=probs.sum(-1,keepdims=True)
topk=8
idx = np.argsort(-probs,axis=-1)[:,:topk]
routed=np.zeros((5,2048))
# load switch weights per selected expert
sgp=get(P+"mlp.switch_mlp.gate_proj.weight").astype(np.uint32).reshape(256,512,256)
sup=get(P+"mlp.switch_mlp.up_proj.weight").astype(np.uint32).reshape(256,512,256)
sdp=get(P+"mlp.switch_mlp.down_proj.weight").astype(np.uint32).reshape(256,2048,64)
sgs=get(P+"mlp.switch_mlp.gate_proj.scales").reshape(256,512,32)
sus=get(P+"mlp.switch_mlp.up_proj.scales").reshape(256,512,32)
sds=get(P+"mlp.switch_mlp.down_proj.scales").reshape(256,2048,8)
sgb=get(P+"mlp.switch_mlp.gate_proj.biases").reshape(256,512,32)
sub=get(P+"mlp.switch_mlp.up_proj.biases").reshape(256,512,32)
sdb=get(P+"mlp.switch_mlp.down_proj.biases").reshape(256,2048,8)
def dexpert(u,s,b,N,expert):
    R,C=u[expert].shape
    wu=np.frombuffer(u[expert].tobytes(),dtype=np.uint8).reshape(R,C*4)
    low=(wu&0xF).astype(np.float32); high=(wu>>4).astype(np.float32)
    nib=np.empty((R,N),dtype=np.float32); nib[:,0::2]=low; nib[:,1::2]=high
    g=np.arange(N)//64; sc=s[expert].astype(np.float32)[:,g]; bi=b[expert].astype(np.float32)[:,g]
    return nib*sc+bi
for ti in range(5):
    for k in range(topk):
        e=idx[ti,k]
        Wg=dexpert(sgp,sgs,sgb,2048,e); Wu=dexpert(sup,sus,sub,2048,e); Wd=dexpert(sdp,sds,sdb,512,e)
        gv=Wg@moe_in[ti]; act_e=(gv/(1+np.exp(-gv)))*(Wu@moe_in[ti])
        routed[ti]+=probs[ti,idx[ti,k]]*(Wd@act_e)
hidden0 = h1 + shared*sh_gate + routed
hidden0_swift = np.fromfile("/tmp/qwen_raw_hidden0.bin", dtype=np.float16).astype(np.float32).reshape(5,2048)
projout_sw = np.fromfile("/tmp/qwen_raw_projout0.bin", dtype=np.float16).astype(np.float32).reshape(5,2048)
h1_sw = embed + projout_sw
delta_np = hidden0 - h1
delta_sw = hidden0_swift - h1_sw
print("h1 np vs swift maxabs:", np.max(np.abs(h1-h1_sw)))
print("delta (shared+routed) np vs swift maxabs:", np.max(np.abs(delta_np-delta_sw)), "corr", np.corrcoef(delta_np.flatten(),delta_sw.flatten())[0,1])
print("  shared stats:", shared.min(), shared.max(), shared.mean())
print("  routed stats:", routed.min(), routed.max(), routed.mean())
print("  sh_gate:", sh_gate.ravel()[:6])
print("FULL layer0 maxabs diff vs swift:", np.max(np.abs(hidden0-hidden0_swift)))
print("  numpy hidden0:", hidden0.min(), hidden0.max(), hidden0.mean())
print("  swift hidden0:", hidden0_swift.min(), hidden0_swift.max(), hidden0_swift.mean())
# per-component where biggest diff
d = np.abs(hidden0-hidden0_swift)
print("  max diff at token", int(np.argmax(d.max(1))), "value", float(d.max()))
