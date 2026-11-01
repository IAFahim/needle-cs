import sys, numpy as np
sys.path.insert(0,'/home/user/needle-cs/.claude/worktrees/agent-a6027f285338e378c/tools')
import golden_whistle as gw
from golden import Cact
S='/tmp/claude-0/-home-user-needle-cs/b0ea2699-1951-51d8-a62a-5922b082c136/scratchpad/bench/'
cact=Cact(gw.CACT)
def cos(a,b):
    a=a.reshape(len(a),-1); b=b.reshape(len(b),-1)
    c=(a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1)); return round(float(c.mean()),5), round(float(c.min()),5)
for name in ['hotword16','long25']:
    pcm=gw.read_wav(S+name+'.wav')
    native=np.load(S+name+'.native.npy').reshape(-1,512)
    ours=np.fromfile(S+name+'.cs.f32',np.float32).reshape(-1,512)
    for q in [True, False]:
        m=gw.Whistle(cact,quant=q)
        st=m.encode(pcm)
        e=st['encoder']
        print(name,'quant',q,'ref-vs-native',cos(e,native),'ref-vs-ours',cos(e,ours),'native-vs-ours',cos(native,ours), flush=True)
