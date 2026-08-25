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
w=load_st("language_model.model.layers.3.self_attn.q_proj.weight").reshape(8192,256)
sc=load_st("language_model.model.layers.3.self_attn.q_proj.scales").reshape(8192,32)
bi=load_st("language_model.model.layers.3.self_attn.q_proj.biases").reshape(8192,32)
print("w dt",w.dtype,"sc dt",sc.dtype,"bi dt",bi.dtype)
print("sc[0,0]=",float(sc[0,0]),"bi[0,0]=",float(bi[0,0]))
out=np.zeros(2048,dtype=np.float32)
for i in range(2048):
    u=i//8; bb=(i%8)//2; lo=(i%8)%2
    bv=int((w[0,u]>>(8*bb))&0xFF)
    nibv=(bv>>(4*lo))&0xF
    if nibv>=8: nibv-=16
    g=i//GROUP
    sval=float(sc[0,g]); bval=float(bi[0,g])
    out[i]=nibv*sval+bval
print("out[0]=",out[0],"out[3]=",out[3],"maxabs",np.max(np.abs(out)))
PY