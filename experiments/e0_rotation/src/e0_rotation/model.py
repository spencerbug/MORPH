import torch
from torch import nn
from torch.nn import functional as F


class Predictor(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.encoder = nn.Sequential(nn.Conv2d(1,16,4,2,1), nn.ReLU(),
            nn.Conv2d(16,32,4,2,1), nn.ReLU(), nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Flatten(), nn.Linear(1024,48))
        self.decoder = nn.Sequential(nn.Linear(48,1024), nn.ReLU(), nn.Unflatten(1,(64,4,4)),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(), nn.ConvTranspose2d(32,16,4,2,1),
            nn.ReLU(), nn.ConvTranspose2d(16,1,4,2,1), nn.Sigmoid())
        if method == 'U':
            self.transition = nn.Sequential(nn.Linear(49,64), nn.ReLU(), nn.Linear(64,16))
        self.register_buffer('harmonics', torch.arange(1,5,dtype=torch.float32).view(1,4,1))

    def encode(self, x):
        z = self.encoder(x)
        return z[:, :32], z[:, 32:]

    def decode(self, u, p):
        return self.decoder(torch.cat((u,p), dim=-1))

    def step(self, u, p, action):
        if self.method == 'U':
            return u, p+self.transition(torch.cat((u,p,action), dim=-1))
        angle = action.reshape(-1,1,1)*self.harmonics
        c,s = angle.cos(), angle.sin()
        blocks = p.reshape(-1,4,2,2)
        x,y = blocks[...,0], blocks[...,1]
        return u, torch.stack((c*x-s*y,s*x+c*y),dim=-1).reshape(-1,16)

    def rollout(self, x0, actions):
        u,p = self.encode(x0)
        zs=[]
        for a in actions.unbind(1):
            u,p = self.step(u,p,a)
            zs.append(torch.cat((u,p),-1))
        z=torch.stack(zs,1)
        return self.decoder(z.flatten(0,1)).reshape(x0.shape[0],actions.shape[1],1,32,32)

    def loss(self, images, actions):
        b = images.shape[0]
        z = self.encoder(images.flatten(0,1)).reshape(b,3,48)
        reconstruction = self.decoder(z.flatten(0,1)).reshape_as(images)
        u,p = z[:,0,:32], z[:,0,32:]
        predictions=[]
        for a in actions.unbind(1):
            u,p=self.step(u,p,a)
            predictions.append(torch.cat((u,p),-1))
        zh=torch.stack(predictions,1)
        future=self.decoder(zh.flatten(0,1)).reshape(b,2,1,32,32)
        losses = torch.stack((F.mse_loss(reconstruction,images), F.mse_loss(future,images[:,1:]),
            F.mse_loss(zh,z[:,1:].detach()), F.mse_loss(z[:,0:1,:32].expand(-1,2,-1),z[:,1:,:32])))
        return losses[0]+losses[1]+.1*losses[2]+.1*losses[3], losses.detach()


def warp(x0, actions):
    out=[]
    for alpha in actions.squeeze(-1).cumsum(1).unbind(1):
        c,s=alpha.cos(),alpha.sin()
        # Inverse sample in image coordinates (y down).
        mat=torch.stack((c,-s,torch.zeros_like(c),s,c,torch.zeros_like(c)),1).reshape(-1,2,3)
        grid=F.affine_grid(mat,x0.shape,align_corners=False)
        out.append(F.grid_sample(x0,grid,align_corners=False,mode='bilinear'))
    return torch.stack(out,1)
