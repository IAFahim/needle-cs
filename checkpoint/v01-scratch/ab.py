import sys, time, numpy as np
sys.path.insert(0,'/home/user/needle-cs/.claude/worktrees/agent-a6027f285338e378c/tools')
import golden_whistle as gw
from golden import Cact
S='/tmp/claude-0/-home-user-needle-cs/b0ea2699-1951-51d8-a62a-5922b082c136/scratchpad/bench/'
cact=Cact(gw.CACT)
ALL=['stem','route','attn_in','qkv','attn_out','pw1_in','pw2_in']
def cos(a,b):
    a=a.reshape(len(a),-1); b=b.reshape(len(b),-1)
    c=(a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1)); return round(float(c.mean()),5), round(float(c.min()),5), 'rel %.4f'%(np.sqrt(((a-b)**2).sum())/np.sqrt((b*b).sum()))
name=sys.argv[1]
pcm=gw.read_wav(S+name+'.wav')
native=np.load(S+name+'.native.npy').reshape(-1,512)
ours=np.fromfile(S+name+'.cs.f32',np.float32).reshape(-1,512)
for spec in sys.argv[2:]:
    rot, sites = spec.split(':')
    mode = 'before' if rot.endswith('b') else 'after'
    rot = rot[0]
    sites = set(ALL) if sites=='all' else set() if sites=='none' else set(sites.split(',')) if not sites.startswith('-') else set(ALL)-set(sites[1:].split(','))
    m=gw.Whistle(cact,quant=sites,rotated=rot=='r')
    m.pw2_mode=mode
    t=time.time()
    e=m.encode(pcm)['encoder']
    print(name,spec,'native',cos(e,native),'ours',cos(e,ours),'%.0fs'%(time.time()-t),flush=True)
