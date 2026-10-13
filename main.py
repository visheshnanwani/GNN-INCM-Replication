"""Single-file, auditable practical reproduction of Huang et al. (2022).
See paper_analysis.md for equation corrections and explicit deviations.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, pickle, random, sys, time, urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAST_DEBUG = True  # Used only when no explicit --mode is supplied.
ROOT = Path(__file__).resolve().parent
EXPECTED = {'cora': (2708,1433,7,5429), 'citeseer': (3327,3703,6,4732), 'pubmed': (19717,500,3,44338)}
SOURCE = 'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/'
MODELS = ['gcn','gnn_incm','no_cluster','no_reconstruction','no_hard']
PAPER = { # Table 2, ordered F1 mean, F1 SD, MCC mean, MCC SD; ten trials.
 'cora': {'gcn': [.6861,.0023,.6146,.0014], 'gnn_incm':[.7508,.0045,.7237,.0051], 'no_cluster':[.7407,.0071,.7141,.0061], 'no_reconstruction':[.7431,.0064,.7143,.0049], 'no_hard':[.7439,.0058,.7131,.0057]},
 'citeseer': {'gcn':[.6158,.0029,.5549,.0014], 'gnn_incm':[.6490,.0048,.6274,.0036], 'no_cluster':[.6465,.0061,.6231,.0056], 'no_reconstruction':[.6449,.0094,.6215,.0085], 'no_hard':[.6458,.0052,.6220,.0063]},
 'pubmed': {'gcn':[.7524,.0023,.6265,.0038], 'gnn_incm':[.7704,.0039,.6493,.0059], 'no_cluster':[.7618,.0054,.6478,.0048], 'no_reconstruction':[.7663,.0051,.6449,.0044], 'no_hard':[.7630,.0048,.6419,.0052]}}

@dataclass
class Config:
    dataset: str = 'cora'
    model: str = 'gnn_incm'
    seed: int = 42
    epochs: int = 1000
    patience: int = 200
    hidden: int = 100
    dropout: float = .3
    lr: float = .002
    weight_decay: float = .03
    gamma: float = 1.01
    beta_cap: float = 100.
    edge_samples: int = 10000
    cache: str = str(ROOT / 'data')
    output: str = str(ROOT / 'results')
    threads: int = 2


def seed_everything(seed, threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def paths(output):
    p = Path(output)
    for sub in ['metrics','tables','figures','logs','checkpoints']:
        (p/sub).mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_data(name, cache):
    """Official Planetoid files; preserve all graph nodes and fix CiteSeer gaps.
    Only load pickle files obtained from the named trusted research repository.
    Original Planetoid masks are deliberately replaced by the paper's split.
    """
    cache = Path(cache)/name; cache.mkdir(parents=True, exist_ok=True)
    raw = {}; hashes = {}
    for part in ['x','y','tx','ty','allx','ally','graph','test.index']:
        file = cache/f'ind.{name}.{part}'
        if not file.exists():
            url = SOURCE+file.name
            print('Downloading', url, flush=True)
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                temp = file.with_suffix(file.suffix+'.part'); temp.write_bytes(data); temp.replace(file)
            except Exception as exc:
                raise RuntimeError(f'Download failed. Copy {url} to {file}; see data/README.md.') from exc
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        if part == 'test.index':
            raw[part] = np.loadtxt(file, dtype=int)
        else:
            with file.open('rb') as handle: raw[part] = pickle.load(handle, encoding='latin1')
    order = raw['test.index']; sorted_order = np.sort(order)
    tx, ty = raw['tx'], raw['ty']
    if name == 'citeseer':
        full = np.arange(sorted_order.min(), sorted_order.max()+1)
        ext = sp.lil_matrix((len(full), tx.shape[1])); ext[sorted_order-full[0]] = tx; tx = ext
        ext_y = np.zeros((len(full), ty.shape[1])); ext_y[sorted_order-full[0]] = ty; ty = ext_y
    features = sp.vstack([raw['allx'],tx]).tolil()
    features[order] = features[sorted_order]
    labels = np.vstack([raw['ally'],ty]); labels[order] = labels[sorted_order]
    known = labels.sum(1)>0
    y = labels.argmax(1); y[~known] = -1
    n = features.shape[0]
    row, col = [], []
    for i, neighbors in raw['graph'].items():
        row.extend([i]*len(neighbors)); col.extend(neighbors)
    adj = sp.coo_matrix((np.ones(len(row)), (row,col)), shape=(n,n)).tocsr()
    adj = adj.maximum(adj.T); adj.setdiag(0); adj.eliminate_zeros(); adj.data[:] = 1
    edges = np.stack(adj.nonzero()).astype(np.int64)
    loop_adj = adj + sp.eye(n)
    inv = np.asarray(loop_adj.sum(1)).ravel()**-.5
    norm = (sp.diags(inv) @ loop_adj @ sp.diags(inv)).tocoo()
    tensor_adj = torch.sparse_coo_tensor(np.stack([norm.row,norm.col]), norm.data.astype('float32'), norm.shape, check_invariants=True).coalesce()
    # Row normalization is a declared preprocessing assumption, not stated in paper.
    features = features.tocsr().astype(np.float32)
    sums = np.asarray(features.sum(1)).ravel(); sums[sums==0] = 1
    features = sp.diags(1/sums) @ features
    expected = EXPECTED[name]
    assert (n, features.shape[1],labels.shape[1]) == expected[:3]
    stats = dict(dataset=name,nodes=n,features=features.shape[1],classes=labels.shape[1],
                 undirected_edges=adj.nnz//2,directed_entries=adj.nnz,paper_links=expected[3],
                 unknown_labels=int((~known).sum()),class_counts=np.bincount(y[known]).tolist(),
                 sha256=hashes,source=SOURCE)
    print(json.dumps({k:v for k,v in stats.items() if k!='sha256'}), flush=True)
    return dict(x=torch.from_numpy(features.toarray()), y=torch.tensor(y), adj=tensor_adj,
                edges=torch.from_numpy(edges), edge_codes=np.unique(edges[0]*n+edges[1]), stats=stats)


def split_data(y, seed):
    """Uniform random split, no balancing/stratification or retries based on labels."""
    ids = np.flatnonzero(y.numpy()>=0)
    ids = np.random.default_rng(seed).permutation(ids)
    n_train = int(np.floor(.03*len(ids)))
    n_val = int(np.floor(.10*(len(ids)-n_train)))
    split = dict(train=torch.tensor(ids[:n_train]), val=torch.tensor(ids[n_train:n_train+n_val]), test=torch.tensor(ids[n_train+n_val:]))
    all_ids = torch.cat(list(split.values()))
    assert len(torch.unique(all_ids)) == len(ids)
    return split


class GCN(nn.Module):
    """Equation (1): sparse A X W0 -> ReLU -> dropout -> sparse A H W1.
    K-dimensional output serves both as logits and INCM node embeddings.
    """
    def __init__(self, features, hidden, classes, dropout=.3):
        super().__init__()
        self.first = nn.Linear(features,hidden,bias=False)
        self.second = nn.Linear(hidden,classes,bias=False)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.first.weight); nn.init.xavier_uniform_(self.second.weight)
        self.register_buffer('centers',torch.empty(0,classes))
    def forward(self,x,adj):
        h = F.relu(torch.sparse.mm(adj,self.first(x)))
        return torch.sparse.mm(adj,self.second(F.dropout(h,self.dropout,self.training)))
    def initialize_centers(self,x,adj,y,train,seed):
        self.eval()
        with torch.no_grad(): o = self(x,adj)[train].numpy()
        counts = torch.bincount(y[train],minlength=self.second.out_features)
        if (counts==0).any():
            raise ValueError('A class is absent in training: paper cluster count undefined. Choose a different seed explicitly.')
        k = min(len(train),math.ceil(len(train)/int(counts.min())))
        # Paper only optimizes W0,W1: centers fixed after pre-training k-means.
        self.centers = torch.tensor(KMeans(n_clusters=k,n_init=10,random_state=seed).fit(o).cluster_centers_,dtype=torch.float32)
        self.train()


def soft_assignments(o, centers):
    distance = ((o[:,None,:]-centers[None,:,:])**2).sum(-1)
    q = (1+distance).reciprocal(); q = q/q.sum(1,keepdim=True)
    weighted = q.detach().square()/q.detach().sum(0,keepdim=True).clamp_min(1e-12)
    p = weighted/weighted.sum(1,keepdim=True).clamp_min(1e-12)
    return q,p


def hard_weights(beta,q,p,gamma=1.01,cap=100.):
    """Equation (11), floor 0.5; explicit stability cap, no focal-loss replacement."""
    agree = q.detach().argmax(1)==p.argmax(1)
    return torch.where(agree,beta/gamma,beta*gamma).clamp(.5,cap).detach()


def sample_negative_edges(n,count,edge_codes,rng):
    # Ordered off-diagonal nonedges sampled uniformly with replacement.
    chunks=[]; found=0
    while found<count:
        i=rng.integers(n,size=max(100,2*(count-found))); j=rng.integers(n,size=len(i))
        valid=(i!=j)&~np.isin(i*n+j,edge_codes)
        chunk=np.stack([i[valid],j[valid]]); chunks.append(chunk); found+=chunk.shape[1]
    return torch.from_numpy(np.concatenate(chunks,axis=1)[:,:count])


def reconstruction_loss(o,beta,data,rng,count):
    # Balanced positive/negative sampled BCE, not the dense n^2 sum in Eq.14.
    edges=data['edges']; count=min(count,edges.shape[1])
    pos=edges[:,torch.from_numpy(rng.choice(edges.shape[1],count,replace=False))]
    neg=sample_negative_edges(len(o),count,data['edge_codes'],rng)
    pair=torch.cat([pos,neg],dim=1)
    targets=torch.cat([torch.ones(count),torch.zeros(count)])
    logits=(o[pair[0]]*o[pair[1]]).sum(1)
    weights=(beta[pair[0]]+beta[pair[1]])/2
    return (weights*F.binary_cross_entropy_with_logits(logits,targets,reduction='none')).mean()


def losses(model,o,beta,data,split,cfg,rng,update=True):
    train=split['train']; y=data['y']; classes=o.shape[1]
    if cfg.model=='gcn':
        ce=F.cross_entropy(o[train],y[train]); zero=o.sum()*0
        return ce,dict(classification=ce,clustering=zero,reconstruction=zero),beta
    q,p=soft_assignments(o,model.centers)
    if cfg.model=='no_hard': beta=torch.ones_like(beta)
    elif update: beta=hard_weights(beta,q,p,cfg.gamma,cfg.beta_cap)
    counts=torch.bincount(y[train],minlength=classes).float()
    alpha=len(train)/(classes*counts.clamp_min(1)) # Correct sign in Eq.4.
    ce=(beta[train]*alpha[y[train]]*F.cross_entropy(o[train],y[train],reduction='none')).mean()
    kl=(beta*(p*(p.clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum(1)).mean()
    re=o.sum()*0 if cfg.model=='no_reconstruction' else reconstruction_loss(o,beta,data,rng,cfg.edge_samples)
    if cfg.model=='no_cluster': kl=o.sum()*0
    return ce+kl+re,dict(classification=ce,clustering=kl,reconstruction=re),beta


def metrics(y,pred,classes):
    labels=list(range(classes))
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    return dict(accuracy=float(accuracy_score(y,pred)),macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()),macro_f1=float(f1.mean()),mcc=float(matthews_corrcoef(y,pred)),
                per_class=[dict(class_id=i,precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i in labels],
                confusion_matrix=confusion_matrix(y,pred,labels=labels).tolist())


def run_id(cfg): return f'{cfg.dataset}_{cfg.model}_seed{cfg.seed}'


def plot_run(result,history,data,split,out):
    rid=result['run_id']; figdir=out/'figures'
    cm=np.array(result['confusion_matrix']); fig,ax=plt.subplots(figsize=(6.5,5))
    im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax)
    for i in range(len(cm)):
        for j in range(len(cm)): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xlabel='Predicted class ID',ylabel='True class ID',title=f'{rid}: test confusion matrix',xticks=range(len(cm)),yticks=range(len(cm)))
    fig.tight_layout(); fig.savefig(figdir/f'{rid}_confusion.png',dpi=150); plt.close(fig)
    if history:
        h=pd.DataFrame(history); fig,axes=plt.subplots(1,2,figsize=(11,4))
        for col in ['loss','classification','clustering','reconstruction']: axes[0].plot(h.epoch,h[col],label=col)
        axes[0].set(xlabel='Epoch',ylabel='Training objective'); axes[0].legend(fontsize=8)
        axes[1].plot(h.epoch,h.val_macro_f1,label='Validation macro F1'); axes[1].axvline(result['best_epoch'],ls='--',c='gray',label='Selected epoch')
        axes[1].set(xlabel='Epoch',ylabel='Score'); axes[1].legend(fontsize=8)
        fig.suptitle(rid); fig.tight_layout(); fig.savefig(figdir/f'{rid}_curves.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['train','test']):
        counts=np.bincount(data['y'][split[key]].numpy(),minlength=cm.shape[0]); ax.bar(range(len(counts)),counts,color='#286b89'); ax.set(title=key.capitalize()+' class distribution',xlabel='Class ID',ylabel='Nodes')
    fig.tight_layout(); fig.savefig(figdir/f'{rid}_classes.png',dpi=150); plt.close(fig)


def train(cfg,data=None,debug=False):
    seed_everything(cfg.seed,cfg.threads); out=paths(cfg.output); rid=run_id(cfg)
    data=load_data(cfg.dataset,cfg.cache) if data is None else data
    split=split_data(data['y'],cfg.seed); c=data['stats']['classes']
    model=GCN(data['x'].shape[1],cfg.hidden,c,cfg.dropout)
    if cfg.model!='gcn': model.initialize_centers(data['x'],data['adj'],data['y'],split['train'],cfg.seed)
    print(model, '\nParameters:',sum(p.numel() for p in model.parameters()),'Input:',tuple(data['x'].shape),'Output:',(len(data['y']),c),flush=True)
    beta=.5+.5*torch.rand(len(data['y']))
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    rng=np.random.default_rng(cfg.seed+10000); history=[]; best=-1.; stale=0; start=time.perf_counter(); checkpoint=out/'checkpoints'/f'{rid}.pt'
    before=model.first.weight.detach().clone()
    for epoch in range(1,cfg.epochs+1):
        model.train(); optimizer.zero_grad(); o=model(data['x'],data['adj'])
        loss,parts,beta=losses(model,o,beta,data,split,cfg,rng)
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        loss.backward()
        if debug:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert sum(p.grad.abs().sum().item() for p in model.parameters())>0
        optimizer.step(); model.eval()
        with torch.no_grad(): output=model(data['x'],data['adj']); pred=output.argmax(1)
        val=metrics(data['y'][split['val']].numpy(),pred[split['val']].numpy(),c)
        row=dict(epoch=epoch,loss=float(loss.detach()),val_macro_f1=val['macro_f1'],beta_min=float(beta.min()),beta_max=float(beta.max()),**{k:float(v.detach()) for k,v in parts.items()}); history.append(row)
        if val['macro_f1']>best+1e-9:
            best=val['macro_f1']; stale=0
            torch.save(dict(model=model.state_dict(),config=asdict(cfg),split=split,beta=beta,epoch=epoch,val_macro_f1=best,optimizer=optimizer.state_dict(),dataset_sha256=data['stats']['sha256']),checkpoint)
        else: stale+=1
        if epoch==1 or epoch%50==0: print(rid,json.dumps(row),flush=True)
        if stale>=cfg.patience: break
    assert not torch.equal(before,model.first.weight.detach())
    pd.DataFrame(history).to_csv(out/'logs'/f'{rid}_history.csv',index=False)
    result=evaluate_checkpoint(checkpoint,data,out,history)
    result['epochs_ran']=len(history); result['seconds']=time.perf_counter()-start
    write_json(out/'metrics'/f'{rid}.json',result)
    write_json(out/'logs'/f'{rid}_dataset.json',data['stats'])
    print('RESULT',rid,json.dumps({k:result[k] for k in ['accuracy','macro_f1','mcc','best_epoch','seconds']}),flush=True)
    return result


def evaluate_checkpoint(checkpoint,data=None,out=None,history=None):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True); cfg=Config(**saved['config']); seed_everything(cfg.seed,cfg.threads)
    data=load_data(cfg.dataset,cfg.cache) if data is None else data
    if saved['dataset_sha256']!=data['stats']['sha256']: raise ValueError('Dataset hash differs from training checkpoint')
    model=GCN(data['x'].shape[1],cfg.hidden,data['stats']['classes'],cfg.dropout)
    model.centers=torch.empty_like(saved['model']['centers']); model.load_state_dict(saved['model']); model.eval()
    with torch.no_grad(): o=model(data['x'],data['adj']); pred=o.argmax(1)
    split=saved['split']; test=split['test']
    result=metrics(data['y'][test].numpy(),pred[test].numpy(),o.shape[1])
    result.update(run_id=run_id(cfg),dataset=cfg.dataset,model=cfg.model,seed=cfg.seed,best_epoch=saved['epoch'],val_macro_f1=saved['val_macro_f1'],config=asdict(cfg),split_sizes={k:len(v) for k,v in split.items()})
    out=paths(cfg.output) if out is None else out
    pd.DataFrame(result['per_class']).to_csv(out/'metrics'/f'{run_id(cfg)}_per_class.csv',index=False)
    pd.DataFrame({'node_id':test.numpy(),'true':data['y'][test].numpy(),'prediction':pred[test].numpy()}).to_csv(out/'metrics'/f'{run_id(cfg)}_predictions.csv',index=False)
    plot_run(result,history,data,split,out)
    return result


def markdown_table(df):
    values=df.fillna('Not reported').astype(str)
    return '| '+' | '.join(values.columns)+' |\n|'+'|'.join(['---']*len(values.columns))+'|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in values.to_numpy())+'\n'


def summarize(output):
    out=paths(output); runs=[]
    for file in (out/'metrics').glob('*_seed*.json'):
        r=json.loads(file.read_text()); runs.append({k:r[k] for k in ['dataset','model','seed','accuracy','macro_precision','macro_recall','macro_f1','mcc','best_epoch','epochs_ran','seconds'] if k in r})
    df=pd.DataFrame(runs)
    if df.empty: return df
    df=df.sort_values(['dataset','model','seed']); df.to_csv(out/'tables'/'individual_runs.csv',index=False)
    scores=['accuracy','macro_precision','macro_recall','macro_f1','mcc']
    agg=df.groupby(['dataset','model'])[scores].agg(['mean','std','count']); agg.columns=['_'.join(x) for x in agg.columns]; agg=agg.reset_index()
    agg.to_csv(out/'tables'/'summary.csv',index=False)
    (out/'tables'/'summary.md').write_text(markdown_table(agg.round(5)),encoding='utf-8')
    comparisons=[]
    for dataset,models in PAPER.items():
        for model,values in models.items():
            ours=df[(df.dataset==dataset)&(df.model==model)]
            for metric,idx in [('F1',0),('MCC',2)]:
                col='macro_f1' if metric=='F1' else 'mcc'; mean=ours[col].mean() if len(ours) else None
                comparisons.append({'Dataset':dataset,'Model':model,'Metric':metric,'Paper Result':f'{values[idx]:.4f} +/- {values[idx+1]:.4f}', 'Our Result':f'{mean:.5f}' if mean is not None else '[RESULT TO BE GENERATED]', 'Difference':f'{mean-values[idx]:+.5f}' if mean is not None else '[RESULT TO BE GENERATED]'})
    compare=pd.DataFrame(comparisons); compare.to_csv(out/'tables'/'paper_vs_ours.csv',index=False)
    (out/'tables'/'paper_vs_ours.md').write_text('F1: our macro F1; the paper does not specify multiclass averaging, so F1 differences are descriptive only. MCC: our standard multiclass MCC. Paper Table 2; 10 trials versus the run counts in summary.csv.\n\n'+markdown_table(compare),encoding='utf-8')
    for dataset,group in df.groupby('dataset'):
        g=group.groupby('model').macro_f1.agg(['mean','std']).reindex([m for m in MODELS if m in group.model.values])
        fig,ax=plt.subplots(figsize=(9,4.5)); ax.bar(g.index,g['mean'],yerr=g['std'].fillna(0),capsize=4,color=['#879ca7' if m=='gcn' else '#247d91' for m in g.index]); ax.set(ylabel='Test macro F1 (mean +/- sample SD)',title=f'{dataset}: completed experiments',ylim=(0,1)); ax.tick_params(axis='x',rotation=15)
        fig.tight_layout(); fig.savefig(out/'figures'/f'{dataset}_comparison.png',dpi=160); plt.close(fig)
    return df


def debug_tests(cfg):
    cfg=copy.deepcopy(cfg); cfg.model='gnn_incm'; cfg.epochs=4; cfg.patience=4; cfg.output=str(Path(cfg.output)/'debug')
    seed_everything(cfg.seed,cfg.threads); data=load_data(cfg.dataset,cfg.cache)
    idx=data['adj'].indices().numpy(); val=data['adj'].values().numpy()
    normalized=sp.coo_matrix((val,(idx[0],idx[1])),shape=data['adj'].shape).tocsr()
    assert (normalized-normalized.T).nnz==0
    assert np.all(normalized.diagonal()>0)
    split=split_data(data['y'],cfg.seed)
    audit=GCN(data['x'].shape[1],cfg.hidden,data['stats']['classes'],0)
    audit.initialize_centers(data['x'],data['adj'],data['y'],split['train'],cfg.seed)
    audit.eval(); embedding=audit(data['x'],data['adj']); beta=torch.ones(len(data['y']))
    original=losses(audit,embedding,beta,data,split,cfg,np.random.default_rng(123),False)[0]
    altered=dict(data); altered['y']=data['y'].clone()
    holdout=torch.cat([split['val'],split['test']]); altered['y'][holdout]=(altered['y'][holdout]+1)%data['stats']['classes']
    changed=losses(audit,embedding,beta,altered,split,cfg,np.random.default_rng(123),False)[0]
    assert torch.equal(original,changed), 'Holdout labels leaked into training objective'
    # Controlled convex example: decreasing loss is meaningful here, not imposed on stochastic training.
    parameter=nn.Parameter(torch.tensor([4.])); opt=torch.optim.SGD([parameter],lr=.1)
    initial=float((parameter.detach()-1).square())
    for _ in range(20):
        opt.zero_grad(); loss=(parameter-1).square().sum(); loss.backward(); opt.step()
    final=float((parameter.detach()-1).square()); assert final<initial/100
    q=torch.tensor([[.6,.4],[.6,.4]]); p=torch.tensor([[.7,.3],[.2,.8]])
    b=hard_weights(torch.ones(2),q,p,2,100); assert torch.allclose(b,torch.tensor([.5,2.]))
    o=torch.randn(10,3,requires_grad=True); q,p=soft_assignments(o,torch.randn(4,3)); assert torch.allclose(q.sum(1),torch.ones(10)) and torch.allclose(p.sum(1),torch.ones(10))
    neg=sample_negative_edges(len(data['y']),100,data['edge_codes'],np.random.default_rng(0)); assert not np.isin((neg[0]*len(data['y'])+neg[1]).numpy(),data['edge_codes']).any()
    result=train(cfg,data,debug=True); restored=evaluate_checkpoint(Path(cfg.output)/'checkpoints'/f'{run_id(cfg)}.pt',data)
    assert result['confusion_matrix']==restored['confusion_matrix']
    status=dict(status='passed',controlled_initial_loss=initial,controlled_final_loss=final,checks=['dataset dimensions','holdout-label invariance','disjoint splits','sparse normalization','forward pass','finite loss','nonzero finite gradients','parameter update','soft assignments','target distribution','easy/hard update','negative edges','reconstruction','metrics','checkpoint reload'])
    write_json(Path(cfg.output)/'debug_status.json',status); print('DEBUG PASSED',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['debug','train','evaluate','all','summarize'],default='debug' if FAST_DEBUG else 'train')
    parser.add_argument('--dataset',choices=list(EXPECTED),default='cora'); parser.add_argument('--model',choices=MODELS,default='gnn_incm')
    for name,typ,default in [('seed',int,42),('epochs',int,1000),('patience',int,200),('hidden',int,100),('dropout',float,.3),('lr',float,.002),('weight-decay',float,.03),('gamma',float,1.01),('beta-cap',float,100.),('edge-samples',int,10000),('threads',int,2)]: parser.add_argument('--'+name,type=typ,default=default)
    parser.add_argument('--cache',default=Config.cache); parser.add_argument('--output',default=Config.output)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]); parser.add_argument('--skip-ablations',action='store_true')
    args=parser.parse_args(argv); values=vars(args).copy(); mode=values.pop('mode'); seeds=values.pop('seeds'); skip=values.pop('skip_ablations'); cfg=Config(**values)
    if cfg.epochs<1 or cfg.patience<1 or cfg.gamma<=1 or cfg.beta_cap<.5 or cfg.edge_samples<1: parser.error('Invalid positive iteration/weight settings')
    paths(cfg.output)
    if mode=='debug': debug_tests(cfg)
    elif mode=='train': train(cfg); summarize(cfg.output)
    elif mode=='evaluate':
        result=evaluate_checkpoint(Path(cfg.output)/'checkpoints'/f'{run_id(cfg)}.pt', data=load_data(cfg.dataset,cfg.cache), out=paths(cfg.output)); print(json.dumps(result,indent=2))
    elif mode=='summarize': summarize(cfg.output)
    elif mode=='all':
        data=load_data(cfg.dataset,cfg.cache)
        for model in (MODELS[:2] if skip else MODELS):
            for seed in seeds:
                current=copy.deepcopy(cfg); current.model=model; current.seed=seed; train(current,data)
                summarize(cfg.output)

if __name__ == '__main__':
    main()
