import sys, os, numpy as np
sys.path.insert(0,'/home/user/needle-cs/.claude/worktrees/agent-a6027f285338e378c/tools')
sys.path.insert(0,'/tmp/claude-0/-home-user-needle-cs/b0ea2699-1951-51d8-a62a-5922b082c136/scratchpad/v01')
import golden_whistle as gw
from golden import Cact
A='/home/user/needle-cs/ref/audio/'
S='/tmp/claude-0/-home-user-needle-cs/b0ea2699-1951-51d8-a62a-5922b082c136/scratchpad/bench/'
V='/tmp/claude-0/-home-user-needle-cs/b0ea2699-1951-51d8-a62a-5922b082c136/scratchpad/v01b/'
r=16000
def clips():
    import native
    long=native.wav16(A+'test_long.wav'); test=native.wav16(A+'test.wav'); rec=native.wav16(A+'record.wav'); hot=native.wav16(S+'hotword16.wav')
    c={'long_0':long[:8*r],'long_40':long[40*r:48*r],'test_0':test[:8*r],'test_10':test[10*r:18*r],'rec':rec,'hot_0':hot[:8*r],'hot_9':hot[9*r:17*r],
       'hot_q':hot[:8*r]*0.1,'synA':np.load(V+'synA.pcm.npy')}
    out={}
    for k,v in c.items():
        v=v.astype(np.float32); f=V+'c_'+k+'.native.npy'
        if not os.path.exists(f): np.save(f,native.embed(v))
        out[k]=(v,np.load(f).reshape(-1,512))
    return out
data=clips()
cact=Cact(gw.CACT)
def rel(a,b): return np.sqrt(((a-b)**2).sum())/np.sqrt((b**2).sum())
class M(gw.Whistle):
    model='abs'; ec=gw.FLOOR
    def mel_features(self, pcm):
        n=len(pcm)//self.hop
        padded=np.pad(pcm.astype(np.float32),(self.window//2,self.window//2))
        index=np.arange(n)[:,None]*self.hop+np.arange(self.window)[None,:]
        window=np.hanning(self.window).astype(np.float32)
        spec=np.fft.rfft(padded[index]*window,self.fft)
        power=(spec.real**2+spec.imag**2).astype(np.float32)
        mel=power@self.dense(self.mel)
        if self.model=='abs': eps=self.ec
        elif self.model=='mean': eps=self.ec*mel.mean()
        elif self.model=='max': eps=self.ec*mel.max()
        elif self.model=='maxpow': eps=self.ec*power.max()
        elif self.model=='maxmean': eps=self.ec*mel.max(0).mean()
        elif self.model=='bin': eps=self.ec*mel.mean(0,keepdims=True)
        elif self.model=='frame': eps=self.ec*mel.mean(1,keepdims=True)
        elif self.model=='absmean': eps=gw.FLOOR*0+self.ec*mel.mean()+gw.FLOOR*0.0
        if self.model=='clampmax': x=np.log(np.maximum(mel,np.float32(self.ec*mel.max()))).astype(np.float32)
        elif self.model=='clampmax_abs': x=np.log(np.maximum(mel,np.float32(max(self.ec*mel.max(),gw.FLOOR)))).astype(np.float32)
        elif self.model=='both': x=np.log(mel+np.float32(self.ec*mel.max()+gw.FLOOR)).astype(np.float32)
        else: x=np.log(mel+np.float32(eps)).astype(np.float32)
        mean=x.mean(0,dtype=np.float64); std=np.sqrt(((x-mean)**2).sum(0,dtype=np.float64)/max(n-1,1))
        return ((x-mean)/(std+1e-5)).astype(np.float32)
m=M(cact,quant=set(),rotated=False)
model=sys.argv[1]; exps=[float(a) for a in sys.argv[2].split(',')]
names=sys.argv[3].split(',') if len(sys.argv)>3 else list(data)
print(model,'clips',names,flush=True)
for e in exps:
    m.model=model; m.ec=2.0**e
    print('c=2^%.1f'%e,' '.join('%s %.4f'%(k,rel(m.encode(data[k][0])['encoder'],data[k][1])) for k in names),flush=True)
