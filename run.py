import argparse, hashlib, json, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
import torch
from torch import nn

ROOT=Path(__file__).parent

def prices(ticker, data=ROOT/'data'):
    data.mkdir(exist_ok=True); path=data/(ticker.lower()+'.csv')
    if not path.exists():
        url=f'https://stooq.com/q/d/l/?s={ticker.lower()}&d1=20100101&d2=20251231&i=d'
        path.write_bytes(urllib.request.urlopen(url,timeout=30).read())
    df=pd.read_csv(path,parse_dates=['Date']).set_index('Date').sort_index()
    return df['Close'].dropna().astype(float)

def ewma(x, lam=.94):
    out=np.empty(len(x)); out[0]=x.iloc[0]**2
    for i in range(1,len(x)): out[i]=lam*out[i-1]+(1-lam)*x.iloc[i-1]**2
    return out

def garch_fit(x):
    r=x.to_numpy(float); var=np.var(r);
    def nll(p):
        omega,a,b=p; v=np.empty(len(r)); v[0]=var
        for i in range(1,len(r)): v[i]=omega+a*r[i-1]**2+b*v[i-1]
        if (v<=0).any(): return 1e20
        return .5*np.sum(np.log(v)+r*r/v)
    fit=minimize(nll,[max(var*.05,1e-8),.08,.9],bounds=[(1e-10,var),(1e-5,.3),(1e-5,.999)],constraints={'type':'ineq','fun':lambda p: .999-p[1]-p[2]})
    return fit.x

def garch_forecast(x,p):
    o,a,b=p; v=np.var(x); out=[]
    for r in x:
        out.append(o+a*r*r+b*v); v=out[-1]
    return np.asarray(out)

class TCN(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,16,3,padding=2),nn.ReLU(),nn.Conv1d(16,8,3,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Linear(8,1)
    def forward(self,x): return self.head(self.net(x)[:,:,-1]).squeeze(-1)

def tcn_fit(r, lookback=20, epochs=8, seed=7):
    torch.manual_seed(seed); x=np.asarray(r,float); X=np.array([x[i-lookback:i] for i in range(lookback,len(x))]); y=x[lookback:]**2
    scale=max(np.std(X),1e-8); X=torch.tensor(X/scale,dtype=torch.float32)[:,None]; y=torch.tensor(y/scale**2,dtype=torch.float32)
    m=TCN(); opt=torch.optim.Adam(m.parameters(),lr=.003)
    for _ in range(epochs): opt.zero_grad(); loss=((m(X)-y)**2).mean(); loss.backward(); opt.step()
    return m,scale

def evaluate(r, epochs=8):
    r=np.asarray(r,float); split=int(len(r)*.7); test=r[split:]; p=garch_fit(pd.Series(r[:split]));
    ew=ewma(pd.Series(r)); gf=garch_forecast(r,p); m,scale=tcn_fit(r[:split],epochs=epochs); lb=20
    with torch.no_grad(): pred=m(torch.tensor(r[split-lb:][:,None][None],dtype=torch.float32)) if False else None
    # walk-forward neural predictions use only the fixed training model and past returns
    X=np.array([r[i-lb:i] for i in range(split,len(r))]);
    with torch.no_grad(): nnv=m(torch.tensor(X/scale,dtype=torch.float32)[:,None]).numpy()*scale**2
    methods={'EWMA':ew[split:], 'GARCH':gf[split:], 'TCN':nnv}; actual=test**2
    return {k:{'mse':float(np.mean((v-actual)**2)),'qlike':float(np.mean(np.log(np.maximum(v,1e-12))+actual/np.maximum(v,1e-12)))} for k,v in methods.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tickers',nargs='+',default=['SPY']); ap.add_argument('--epochs',type=int,default=8); a=ap.parse_args(); report={'settings':vars(a),'results':{}}
    for t in a.tickers:
        s=prices(t); report['results'][t]=evaluate(np.log(s).diff().dropna().to_numpy(),a.epochs)
    (ROOT/'results').mkdir(exist_ok=True); (ROOT/'results/report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
