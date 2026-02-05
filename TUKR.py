from tqdm import tqdm
import numpy as np
import sys
import Data_kura
import torch
import math
# 描画関連
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt


class TUKR:
    def __init__(self, X, latent_dim, sigma):
        self.X = X
        self.X=torch.tensor(X.copy())


        self.latent_dim = latent_dim
        self.Tsize = list(X.shape) #テンソルのサイズを格納
        self.sigma1 = sigma[0]
        self.sigma2 = sigma[1]
        self.sigma3 = sigma[2]


        # 潜在変数の初期値設定(一様分布)
        self.zeta1 = 0.2 * self.sigma1 * np.random.rand(self.Tsize[0], self.latent_dim) - 0.1 * self.sigma1
        self.zeta1=torch.tensor(self.zeta1,requires_grad=True,dtype=torch.float64)
        self.zeta2 = 0.2 * self.sigma2 * np.random.rand(self.Tsize[1], self.latent_dim) - 0.1 * self.sigma2
        self.zeta2=torch.tensor(self.zeta2,requires_grad=True,dtype=torch.float64)
        self.zeta3 = 0.2 * self.sigma3 * np.random.rand(self.Tsize[2], self.latent_dim) - 0.1 * self.sigma3
        self.zeta3=torch.tensor(self.zeta3,requires_grad=True,dtype=torch.float64)


        self.history = {} #学習結果を記録


    # 潜在変数の差を求める
    def _Delta(self, Z1, Z2):
        Delta = Z1[:, np.newaxis, :] - Z2[np.newaxis, :, :]
        return Delta
    
    # カーネル平滑化
    def _smoothing_Kernel(self, Z1, Z2, sigma):
        Delta = self._Delta(Z1,Z2)
        Delta2 = torch.sum(Delta*Delta, dim=2)
        K = torch.exp(-0.5 * Delta2 / (sigma**2))
        K_sum = torch.sum(K, dim=1, keepdim=True)


        return K / K_sum
    
    # 写像
    def _function(self, Z1, Z2, Z3):
        KernelZ1 = self._smoothing_Kernel(Z1, Z1, self.sigma1)
        KernelZ2 = self._smoothing_Kernel(Z2, Z2, self.sigma2)
        KernelZ3 = self._smoothing_Kernel(Z3, Z3, self.sigma3)


        K1 = torch.einsum("li,ijk->ljk", KernelZ1, self.X)
        K2 = torch.einsum("mj,ljk->lmk", KernelZ2, K1)
        K3 = torch.einsum("nk,lmk->lmn", KernelZ3, K2)


        return K3
    
    # 正則化関数
    def _regularization(self, U):
        return torch.sum(U**10)


    # 損失関数
    def _error_function(self, Z1, Z2, Z3):
        self.f = self._function(Z1, Z2, Z3)
        ef = 1 / (math.prod(self.Tsize)) * torch.sum((self.X - self.f)**2)
        ef = ef + self.lambda1 * self._regularization(Z1) + self.lambda2 * self._regularization(Z2) +self.lambda3 * self._regularization(Z3)


        return ef
    
    def _create_Y(self, U, V, W):
        U = torch.from_numpy(U).clone()
        V = torch.from_numpy(V).clone()
        W = torch.from_numpy(W).clone()


        R1 = self._smoothing_Kernel(U, self.zeta1, self.sigma1)
        R2 = self._smoothing_Kernel(V, self.zeta2, self.sigma2)
        R3 = self._smoothing_Kernel(W, self.zeta3, self.sigma3)


        K1 = torch.einsum("li,ijk->ljk", R1, self.X)
        K2 = torch.einsum("mj,ljk->lmk", R2, K1)
        Y = torch.einsum("nk,lmk->lmn", R3, K2)


        return Y


    # 学習
    def fit(self, epoch_num, lambda_, eata):
        self.history["z1"] = np.zeros((epoch_num, self.Tsize[0], self.latent_dim))
        self.history["z2"] = np.zeros((epoch_num, self.Tsize[1], self.latent_dim))
        self.history["z3"] = np.zeros((epoch_num, self.Tsize[2], self.latent_dim))
        self.history["f"] = np.zeros((epoch_num, self.Tsize[0], self.Tsize[1], self.Tsize[2]))
        self.history["ef"] = np.zeros((epoch_num,))


        self.lambda1 = lambda_[0]
        self.lambda2 = lambda_[1]
        self.lambda3 = lambda_[2]
        eata1 = eata[0]
        eata2 = eata[1]
        eata3 = eata[2]


        for epoch in tqdm(np.arange(epoch_num)):
            func_E = self._error_function(self.zeta1, self.zeta2, self.zeta3)
            func_E.backward()


            # 勾配法による潜在変数の更新
            with torch.no_grad():
                dEdZ1 = self.zeta1.grad
                dEdZ2 = self.zeta2.grad
                dEdZ3 = self.zeta3.grad


                self.zeta1 = self.zeta1 - eata1 * dEdZ1
                self.zeta2 = self.zeta2 - eata2 * dEdZ2
                self.zeta3 = self.zeta3 - eata3 * dEdZ3


                # historyに保存
                self.history["z1"][epoch,:,:] = self.zeta1.detach().numpy()
                self.history["z2"][epoch,:,:] = self.zeta2.detach().numpy()
                self.history["z3"][epoch,:,:] = self.zeta3.detach().numpy()
                self.history["f"][epoch,:,:,:] = self.f.detach().numpy()
                self.history["ef"][epoch]=func_E.item()


            # 一度更新すると自動的にfalseになるため、計算できなくなる
            self.zeta1.requires_grad = True
            self.zeta2.requires_grad = True
            self.zeta3.requires_grad = True


def create_zeta(Z, resolution):
    # zetaの計算
    latent_dim = Z.shape[-1]
    mesh, step = np.linspace(Z.min(), Z.max(), resolution, endpoint=False, retstep=True)
    mesh += step / 2.0
    zeta = np.empty((resolution, latent_dim))
    if latent_dim == 1:
        zeta = mesh[:, None]
    elif latent_dim == 2:
        xx, yy = np.meshgrid(mesh, mesh)
        zeta = np.concatenate([xx.reshape(-1)[:, None], yy.reshape(-1)[:, None]], axis=1)
    return zeta


def _main():
    eata = [1,1,1]
    epoch_num = 200
    sigma = [0.1,0.1,0.1]
    lambda_ = [0,0,0]
    resolution = 25
    dir_results = '/results'
    dir_save = '/result'
    np.random.seed(0)


    X = Data_kura.kura()
    tukr = TUKR(X, latent_dim=2, sigma=sigma)
    tukr.fit(epoch_num=epoch_num, lambda_=lambda_, eata=eata)


    zeta1 = create_zeta(tukr.history['z1'][epoch_num-1, :, :], resolution)
    zeta2 = create_zeta(tukr.history['z2'][epoch_num-1, :, :], resolution)
    zeta3 = create_zeta(tukr.history['z3'][epoch_num-1, :, :], resolution)
    Y = tukr._create_Y(zeta1, zeta2, zeta3)
    Y = Y.to('cpu').detach().numpy().copy()




    np.save(dir_results + dir_save +'/u_history', tukr.history['z1'][:, :, :])
    np.save(dir_results + dir_save +'/v_history', tukr.history['z2'][:, :, :])
    np.save(dir_results + dir_save +'/w_history', tukr.history['z3'][:, :, :])
    np.save(dir_results + dir_save +'/Y_history', Y)
    np.save(dir_results + dir_save +'/zetau_history', zeta1)
    np.save(dir_results + dir_save +'/zetav_history', zeta2)
    np.save(dir_results + dir_save +'/zetaw_history', zeta3)


if __name__ == "__main__":  # このファイルを実行した時のみ
    _main()
