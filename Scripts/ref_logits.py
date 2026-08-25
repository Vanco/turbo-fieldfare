import struct, glob, json, numpy as np
SD="local-model/mlx-community/Qwen3.5-35B-A3B-4bit/main"
GROUP=64
def bf16_to_f32(u16): return (u16.astype(np.uint32)<<16).view(np.float32)
def load_st(key):
    for x in glob.glob(SD+"/*.safetensors"):
        with open(x,'rb') as fh:
            n=struct.unpack('<Q',fh.read(8))[0]; hdr=json.loads(fh.read(n)); base=8+n
            if key in hdr:
                s,e=hdr[key]['data_offsets']; fh.seek(base+s); raw=fh.read(e-s)
                dt=hdr[key]['dtype']
                if dt=='U32': return np.frombuffer(raw,dtype=np.uint32)
                if dt=='BF16': return bf16_to_f32(np.frombuffer(raw,dtype=np.uint16))
    raise KeyError(key)

# final hidden last token
h=np.fromfile("/tmp/qwen_raw_final_hidden.bin",dtype=np.float16).astype(np.float32).reshape(13,2048)
last=h[-1]
# final norm
fnw=load_st("language_model.model.norm.weight") if "language_model.model.norm.weight" in open(glob.glob(SD+"/*.safetensors")[0],'rb').read().decode('latin1','ignore') else None
# find norm key
normkey="language_model.model.norm.weight"
print("normkey",normkey)
fnw=load_st(normkey)
ms=np.mean(last*last); normed=last/np.sqrt(ms+1e-6)*fnw
print("normed finite",np.isfinite(normed).all(),"maxabs",np.max(np.abs(normed)))

# lm_head
w=load_st("language_model.lm_head.weight").reshape(248320, 256)
sc=load_st("language_model.lm_head.scales").reshape(248320,32)
bi=load_st("language_model.lm_head.biases").reshape(248320,32)
print("lm shapes",w.shape,sc.shape,bi.shape)
# dequant rows lazily: logits[r] = sum_i nib(r,i)*sc[r,i//64]*normed[i] + bi[r,i//64]*normed[i]
# build full W is 2GB; do per-row loop in chunks
N=2048
logits=np.zeros(248320,dtype=np.float32)
rows=w.shape[0]
# vectorized: for each group g (0..31), elems [g*64:(g+1)*64]
# weight nibble matrix Wn (rows, N) ; value = Wn*sc[:,g None] + bi[:,g None]; dot with normed
# build Wn row by row is expensive; instead accumulate group contributions
for g in range(N//GROUP):
    i0=g*GROUP
    u0=i0//8; bb0=(i0%8)//2; lo0=i0%2
    # 64 elements = 8 bytes = 8 nibbles -> indices i=i0..i0+63, u=i//8, etc.
    nibs=np.zeros((rows, GROUP),dtype=np.float32)
    for j in range(GROUP):
        i=i0+j
        u=i//8; bb=(i%8)//2; lo=(i%8)%2
        # MLX affine: nibble is UNSIGNED 0..15 (matches GEMV kernel)
        bv=((w[:,u]>>(8*bb))&0xFF).astype(np.uint32)
        nv=((bv>>(4*lo))&0xF).astype(np.float32)
        nibs[:,j]=nv
    contrib=(nibs*sc[:,g,None]+bi[:,g,None])*normed[i0:i0+GROUP]
    logits+=contrib.sum(axis=1)
print("logits_ref finite",np.isfinite(logits).all(),"maxabs",np.max(np.abs(logits)),"top5 ids",np.argsort(-logits)[:5])

lg=np.fromfile("/tmp/tf_logits.bin",dtype=np.float16).astype(np.float32)[:248320]
print("dumped logits: max",lg.max(),"min",lg.min())
print("corr ref vs dumped", np.corrcoef(logits, lg)[0,1])
