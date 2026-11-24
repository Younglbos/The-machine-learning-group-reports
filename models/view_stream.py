"""Bounded summaries for a fixed-object view stream; model updates receive no labels."""

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize
from models.kmeans import fit_kmeans


def align_to_summary(features,reference):
    features=normalize(np.asarray(features,dtype=np.float32))
    if reference is None: return features
    if features.shape!=reference.shape: raise ValueError('Object alignment or representation dimension changed')
    rotation,_=orthogonal_procrustes(features-features.mean(0),reference-reference.mean(0))
    return normalize((features-features.mean(0))@rotation+reference.mean(0)).astype(np.float32)


class ViewEncoder:
    def __init__(self,components,seed,forgetting=0.5):
        self.components,self.seed,self.forgetting=components,seed,forgetting
        self.reference=None

    def update(self,scaled):
        rank=min(self.components,scaled.shape[1],len(scaled)-1)
        reduced=PCA(n_components=rank,svd_solver='randomized',random_state=self.seed).fit_transform(scaled)
        padded=np.zeros((len(scaled),self.components),dtype=np.float32)
        padded[:,:rank]=reduced
        result=align_to_summary(padded,self.reference)
        self.reference=result.copy() if self.reference is None else normalize(
            self.forgetting*self.reference+(1-self.forgetting)*result).astype(np.float32)
        return result


class ViewBaseline:
    def __init__(self,method,clusters,seed,options,components=16,window=2):
        self.method,self.clusters,self.seed,self.options=method,clusters,seed,options
        self.encoder=ViewEncoder(components,seed)
        self.window=window
        self.history=[]
        self.model=None
        self.first_probe=None

    def update(self,scaled):
        if self.method=='full_retrain':
            self.history.append(scaled.copy())
            representation=np.concatenate(self.history,axis=1)
            training=representation
        else:
            representation=self.encoder.update(scaled)
            if self.method=='sliding_window':
                self.history=(self.history+[representation.copy()])[-self.window:]
                training=np.concatenate(self.history)
            else: training=representation
        centers=self.model.cluster_centers_.copy() if self.model is not None and self.method=='warm_start' else None
        self.model,_=fit_kmeans(training,self.clusters,self.seed,self.options,centers)
        return self.model.predict(representation),representation

    def predict(self,features):
        if self.method=='full_retrain' and features.shape[1]<self.model.n_features_in_:
            features=np.pad(features,((0,0),(0,self.model.n_features_in_-features.shape[1])))
        return self.model.predict(features)

    @property
    def state_bytes(self):
        n=sum(x.nbytes for x in self.history)
        if self.encoder.reference is not None: n+=self.encoder.reference.nbytes
        if self.model is not None: n+=self.model.cluster_centers_.nbytes+self.model.labels_.nbytes
        return n


class DSVCCourse:
    """DSVC networks/losses with bounded aligned consensus and stable prototype groups.

    Course adaptation, not the preserved author-code reproduction: only a prototype
    buffer and one N-by-D summary survive a view; inference reuses learned attention.
    """
    def __init__(self,config,seed):
        import torch
        from models.author_import import author_path
        author_path('DSVC')
        from network import Autoencoder
        from Knowledge_Generation import dual_attention
        from Loss import Loss_Reconstruction,Loss_Konw_Guide,Cross_contrast
        self.torch=torch
        self.Autoencoder,self.Attention=Autoencoder,dual_attention
        self.reconstruction=Loss_Reconstruction().cuda()
        self.guide=Loss_Konw_Guide().cuda()
        self.cross=Cross_contrast().cuda()
        self.config,self.seed=config,seed
        self.buffer=None
        self.summary=None
        self.model=None
        self.view=0
        self.loss_trace=[]
        torch.set_num_threads(1)
        torch.backends.cudnn.enabled=False

    def update(self,scaled):
        import itertools
        torch=self.torch
        cfg=self.config
        torch.manual_seed(self.seed+self.view)
        torch.cuda.manual_seed_all(self.seed+self.view)
        decoder=self.Autoencoder(scaled.shape[1],cfg['feature_dim'],activation='relu',batchnorm=True).cuda()
        attention=self.Attention(dim=cfg['feature_dim'],num=cfg['prototype_count']).cuda()
        optimizer=torch.optim.Adam(itertools.chain(decoder.parameters(),attention.parameters()),lr=cfg['learning_rate'])
        groups=[np.arange(i,min(i+cfg['batch_size'],len(scaled))) for i in range(0,len(scaled),cfg['batch_size'])]
        if len(groups[-1])<2: groups[-2]=np.concatenate([groups[-2],groups.pop()])
        if self.buffer is not None and len(self.buffer)!=len(groups): raise ValueError('Object groups changed')
        previous=self.buffer
        data=torch.as_tensor(scaled,device='cuda',dtype=torch.float32)
        for epoch in range(cfg['epochs']):
            current=[]
            epoch_loss=0.0
            for group_id,indices in enumerate(groups):
                batch=data[indices]
                latent,reconstructed=decoder(batch)
                guided,prototype,_=attention(latent,None)
                loss=cfg['beta1']*self.reconstruction(batch,reconstructed)
                if previous is None:
                    loss=loss+cfg['beta1']*self.cross(latent,guided)+cfg['beta2']*self.guide(prototype,guided)
                else:
                    old=torch.as_tensor(previous[group_id],device='cuda')
                    loss=loss+self.cross(old,prototype)+cfg['beta2']*self.guide(prototype,latent)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite DSVC course loss')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current.append(prototype.detach().cpu().numpy())
                epoch_loss+=float(loss.detach())
            if epoch==cfg['epochs']-1: self.buffer=current
            if (epoch+1)%50==0 or epoch==cfg['epochs']-1:
                self.loss_trace.append({'view':self.view+1,'epoch':epoch+1,'loss':epoch_loss})
        decoder.eval(),attention.eval()
        with torch.no_grad():
            features=attention(decoder(data)[0],None)[0].cpu().numpy()
        aligned=align_to_summary(features,self.summary)
        self.summary=aligned if self.summary is None else normalize(
            cfg['forgetting']*self.summary+(1-cfg['forgetting'])*aligned).astype(np.float32)
        centers=None if self.model is None else self.model.cluster_centers_.copy()
        self.model,predicted=fit_kmeans(self.summary,cfg['clusters'],self.seed,cfg['kmeans'],centers)
        self.view+=1
        return predicted,self.summary.copy()

    def predict(self,features): return self.model.predict(features)

    @property
    def state_bytes(self):
        n=0 if self.summary is None else self.summary.nbytes
        if self.buffer is not None: n+=sum(x.nbytes for x in self.buffer)
        if self.model is not None: n+=self.model.cluster_centers_.nbytes+self.model.labels_.nbytes
        return n
