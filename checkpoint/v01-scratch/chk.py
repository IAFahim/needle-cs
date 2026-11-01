import sys, numpy as np
sys.path.insert(0,'/home/user/needle-cs/.claude/worktrees/agent-a6027f285338e378c/tools')
import golden_whistle as gw
from golden import Cact
S='/tmp/claude-0/-home-user-needle-cs/b0ea2699-1951-51d8-a62a-5922b082c136/scratchpad/bench/'
V='/tmp/claude-0/-home-user-needle-cs/b0ea2699-1951-51d8-a62a-5922b082c136/scratchpad/v01b/'
name=sys.argv[1]
class M(gw.Whistle):
    extra=0
    def mel_features(self, pcm):
        n=len(pcm)//self.hop+self.extra
        padded=np.pad(pcm.astype(np.float32),(self.window//2,self.window//2+2000))
        index=np.arange(n)[:,None]*self.hop+np.arange(self.window)[None,:]
        window=np.hanning(self.window).astype(np.float32)
        spec=np.fft.rfft(padded[index]*window,self.fft)
        power=(spec.real**2+spec.imag**2).astype(np.float32)
        mel=power@self.dense(self.mel)
        x=np.log(mel+np.float32(1e-8*mel.max())).astype(np.float32)
        mean=x.mean(0,dtype=np.float64); std=np.sqrt(((x-mean)**2).sum(0,dtype=np.float64)/max(n-1,1))
        return ((x-mean)/(std+1e-5)).astype(np.float32)
pcm=gw.read_wav(S+name+'.wav')
nat=np.load(S+name+'.native.npy').reshape(-1,512)
ours=np.fromfile(V+name+'_cur/encoder.f32',np.float32).reshape(-1,512)
m=M(Cact(gw.CACT),quant=set(),rotated=False)
m.extra=int(sys.argv[2]) if len(sys.argv)>2 else 0
st=m.encode(pcm); ref=st['encoder']
print('ref frames',len(ref),'native',len(nat))
mel_cs=np.fromfile(V+name+'_cur/mel.f32',np.float32).reshape(-1,80)

def rel(a,b): return np.sqrt(((a-b)**2).sum(1))/np.sqrt((b**2).sum(1))
for lab,a,b in [('ref-nat',ref,nat),('ours-nat',ours,nat),('ours-ref',ours,ref)]:
    r=rel(a,b); print(lab,'overall %.4f median %.4f worst'%(np.sqrt(((a-b)**2).sum())/np.sqrt((b**2).sum()),np.median(r)),np.argsort(-r)[:6],np.sort(r)[::-1][:6].round(3))
