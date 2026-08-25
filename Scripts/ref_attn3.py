import numpy as np, os

def rmsnorm(x, w, eps=1e-6):
    ms=np.mean(x*x, axis=-1, keepdims=True)
    return x/np.sqrt(ms+eps)*w

def rope(x, pos):
    # x: (t, heads, 256); rope first 64 dims, rp=32 pairs, theta=1e7
    t,heads,d=x.shape
    out=x.copy(); theta=1e7; rp=32
    for hh in range(heads):
        for i in range(t):
            p=pos[i]
            for pair in range(rp):
                angle=p*(theta**(-pair/rp))
                c=np.cos(angle); s=np.sin(angle)
                a=out[i,hh,pair]; b=out[i,hh,pair+rp]
                out[i,hh,pair]=a*c-b*s
                out[i,hh,pair+rp]=a*s+b*c
    return out

def load_raw(name, rows, cols):
    a=np.fromfile(f"/tmp/qwen_raw_{name}.bin", dtype=np.float16).astype(np.float32)
    return a.reshape(rows,cols)

def bf16_to_f32(u16): return (u16.astype(np.uint32)<<16).view(np.float32)
def load_bf16(key):
    import struct,json,glob
    for x in glob.glob(os.path.join("local-model/mlx-community/Qwen3.5-35B-A3B-4bit/main","*.safetensors")):
        with open(x,'rb') as fh:
            n=struct.unpack('<Q',fh.read(8))[0]; hdr=json.loads(fh.read(n)); base=8+n
            if key in hdr:
                s,e=hdr[key]['data_offsets']; fh.seek(base+s); raw=fh.read(e-s)
                return bf16_to_f32(np.frombuffer(raw,dtype=np.uint16))
    raise KeyError(key)

t=13
qraw=load_raw("L3_qraw", t, 8192)
kraw=load_raw("L3_kraw", t, 512)
vraw=load_raw("L3_vraw", t, 512)
attn_swift=load_raw("L3_attn", t, 4096)

qn=load_bf16("language_model.model.layers.3.self_attn.q_norm.weight").astype(np.float32)
kn=load_bf16("language_model.model.layers.3.self_attn.k_norm.weight").astype(np.float32)

qhead=np.zeros((t,16,256)); qgate=np.zeros((t,16,256))
for h in range(16):
    qhead[:,h,:]=qraw[:, h*512 : h*512+256]
    qgate[:,h,:]=qraw[:, h*512+256 : h*512+512]
k=kraw.reshape(t,2,256)
v=vraw.reshape(t,2,256)

for h in range(16): qhead[:,h,:]=rmsnorm(qhead[:,h,:], qn)
for h in range(2):  k[:,h,:]=rmsnorm(k[:,h,:], kn)

pos=np.arange(t)
qhead=rope(qhead,pos)
k=rope(k,pos)

qcompact_swift=load_raw("L3_qcompact", t, 4096)
qhead_flat=qhead.reshape(t,4096)
print("qCompact vs swift: maxabs", np.max(np.abs(qhead_flat-qcompact_swift)), "corr", np.corrcoef(qhead_flat.ravel(),qcompact_swift.ravel())[0,1])

scale=1.0/np.sqrt(256)
attn=np.zeros((t,16,256),dtype=np.float32)
print("qhead[0,0,:3]",qhead[0,0,:3],"finite",np.isfinite(qhead).all())
print("k[0,0,:3]",k[0,0,:3],"finite",np.isfinite(k).all())
print("v[0,0,:3]",v[0,0,:3],"finite",np.isfinite(v).all())
print("qn[:3]",qn[:3],"kn[:3]",kn[:3])
for i in range(t):
    for h in range(16):
        kvh=h//8
        Q=qhead[i,h,:]; K=k[:i+1,kvh,:]; V=v[:i+1,kvh,:]
        sc=(Q@K.T)*scale
        sc=sc-sc.max()
        e=np.exp(sc); p=e/e.sum()
        attn[i,h,:]=p@V
attn=attn.reshape(t,4096)
gate=1.0/(1.0+np.exp(-qgate.reshape(t,4096)))
out=gate*attn
print("attn_ref vs swift: maxabs", np.max(np.abs(out-attn_swift)), "corr", np.corrcoef(out.ravel(),attn_swift.ravel())[0,1])
print("swift attn sample[:8]", attn_swift[0,:8])
print("ref   attn sample[:8]", out[0,:8])
