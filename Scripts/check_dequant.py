import struct, glob, json, numpy as np, os
SD="local-model/mlx-community/Qwen3.5-35B-A3B-4bit/main"
GROUP=64
def bf16_to_f32(u16): return (u16.astype(np.uint32)<<16).view(np.float32)
def load_st(key):
    for x in glob.glob(os.path.join(SD,"*.safetensors")):
        with open(x,'rb') as fh:
            n=struct.unpack('<Q',fh.read(8))[0]; hdr=json.loads(fh.read(n)); base=8+n
            if key in hdr:
                s,e=hdr[key]['data_offsets']; fh.seek(base+s); raw=fh.read(e-s)
                dt=hdr[key]['dtype']
                if dt=='U32': return np.frombuffer(raw,dtype=np.uint32), hdr[key]['shape']
                if dt=='BF16': return bf16_to_f32(np.frombuffer(raw,dtype=np.uint16)), hdr[key]['shape']
    raise KeyError(key)

w,sh=load_st("language_model.model.layers.3.self_attn.q_proj.weight")
sc,_=load_st("language_model.model.layers.3.self_attn.q_proj.scales")
bi,_=load_st("language_model.model.layers.3.self_attn.q_proj.biases")
w=w.reshape(sh)
sc=sc.reshape(sh[0], sh[1]*8//GROUP)
bi=bi.reshape(sh[0], sh[1]*8//GROUP)
print("weight shape",sh,"scales shape",sc.shape,"biases shape",bi.shape)
print("sc max",np.nanmax(np.abs(sc)),"bi max",np.nanmax(np.abs(bi)))

rows=sh[0]; in_logical=sh[1]*8
print("in_logical",in_logical)
W=np.zeros((rows,in_logical),dtype=np.float32)
for i in range(in_logical):
    u=i//8; byte_idx=(i%8)//2; lo=(i%8)%2
    b=(w[:,u]>>(8*byte_idx))&0xFF
    nib=(b>>(4*lo))&0xF
    nib=np.where(nib>=8,nib-16,nib).astype(np.float32)
    g=i//GROUP
    W[:,i]=nib*sc[:,g]+bi[:,g]
print("Wq maxabs",np.nanmax(np.abs(W)),"finite",np.isfinite(W).all())

# swift qraw
qraw_swift=np.fromfile("/tmp/qwen_raw_L3_qraw.bin",dtype=np.float16).astype(np.float32).reshape(13,8192)
# normed
normed=np.fromfile("/tmp/qwen_raw_L3_normed.bin",dtype=np.float16).astype(np.float32).reshape(13,2048)
qraw_ref=normed @ W.T
print("qraw_ref vs swift: maxabs", np.max(np.abs(qraw_ref-qraw_swift)))
