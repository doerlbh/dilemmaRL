# import multiprocessing
# from joblib import Parallel, delayed
# from tqdm import tqdm

from utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats.stats import pearsonr   
import pickle
import numpy as np
import pandas as pd

SMALL_SIZE = 40
MEDIUM_SIZE = 50
BIGGER_SIZE = 60

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

MAB_algs = ['UCB','TS','eGreedy','EXP3','HBTS']
CB_algs = ['LinUCB','CTS','EXP4','SCTS']
RL_algs = ['QL','DQL','SARSA','SQL']
HC_algs = ['Coop','Dfct','Tit4Tat']
all_algs = ['UCB','TS','eGreedy','EXP3','HBTS','LinUCB','CTS','EXP4','SCTS','QL','DQL','SARSA','SQL','Coop','Dfct','Tit4Tat']
agent_algs = ['UCB','TS','eGreedy','EXP3','HBTS','LinUCB','CTS','EXP4','SCTS','QL','DQL','SARSA','SQL']

mMAB_algs = ['HBTS','bAD','bADD','bAD``HD','bbvFTD','bCP','bM','bPD']
mCB_algs = ['SCTS','cAD','cADD','cADHD','cbvFTD','cCP','cM','cPD']
mRL_algs = ['SQL','AD','ADD','ADHD','bvFTD','CP','M','PD']

# Case with 2 agents

fd = 'ipd1_m5'
nMemory = 5
T = 50
ALGS = all_algs
tab_r = np.zeros((len(ALGS),len(ALGS),T))
tab_r_std = np.zeros((len(ALGS),len(ALGS),T))
tab_p = np.zeros((len(ALGS),len(ALGS),T))
tab_p_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_sum = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_dff = np.zeros((len(ALGS),len(ALGS),T))
tab_rd_std = np.zeros((len(ALGS),len(ALGS),T))

for i,alg1 in enumerate(ALGS):
    for j,alg2 in enumerate(ALGS):
        print(i,j,alg1,alg2)
        r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2],nMemory=nMemory,prefix=fd)
        tab_r[i,j,:] = r[0]
        tab_r_std[i,j,:] = r_std[0]
        tab_p[i,j,:] = p[0]
        tab_p_std[i,j,:] = p_std[0]
        tab_r[j,i,:] = r[1]
        tab_r_std[j,i,:] = r_std[1]
        tab_p[j,i,:] = p[1]
        tab_p_std[j,i,:] = p_std[1]
        tab_rs_sum[i,j,:] = tab_rs_sum[j,i,:] = rs_sum
        tab_rs_std[i,j,:] = tab_rs_std[j,i,:] = rs_std
        tab_rs_dff[i,j,:] = rs_dff[0]
        tab_rd_std[i,j,:] = rd_std[0]
        tab_rs_dff[j,i,:] = rs_dff[1]
        tab_rd_std[j,i,:] = rd_std[1]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m5.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Case with 3 agents

fd = 'ipd1_m5_3ag'
nMemory = 5
T=50
ALGS1 = MAB_algs
ALGS2 = CB_algs
ALGS3 = RL_algs
ALGSALL = agent_algs
MAB_range = np.arange(5)
CB_range = np.arange(5,9)
RL_range = np.arange(9,13)
tab_r = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_r_std = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_p = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_p_std = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_rs_sum = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_rs_std = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_rs_dff = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_rd_std = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))

for it,alg1 in enumerate(ALGS1):
    for jt,alg2 in enumerate(ALGS2):
        for kt,alg3 in enumerate(ALGS3):
            i = MAB_range[it]
            j = CB_range[jt]
            k = RL_range[kt]
            print(i,j,k,alg1,alg2,alg3)
            r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2,alg3],nMemory=nMemory,prefix=fd)
            tab_r[i,j,k,:] = r[0]
            tab_r_std[i,j,k,:] = r_std[0]
            tab_p[i,j,k,:] = p[0]
            tab_p_std[i,j,k,:] = p_std[0]
            tab_r[j,k,i,:] = r[1]
            tab_r_std[j,k,i,:] = r_std[1]
            tab_p[j,k,i,:] = p[1]
            tab_p_std[j,k,i,:] = p_std[1]
            tab_r[k,i,j,:] = r[2]
            tab_r_std[k,i,j,:] = r_std[2]
            tab_p[k,i,j,:] = p[2]
            tab_p_std[k,i,j,:] = p_std[2]
            tab_rs_sum[i,j,k,:] = tab_rs_sum[j,k,i,:] = tab_rs_sum[k,i,j,:] = rs_sum
            tab_rs_std[i,j,k,:] = tab_rs_std[j,k,i,:] = tab_rs_std[k,i,j,:] = rs_std
            tab_rs_dff[i,j,k,:] = rs_dff[0]
            tab_rd_std[i,j,k,:] = rd_std[0]
            tab_rs_dff[j,k,i,:] = rs_dff[1]
            tab_rd_std[j,k,i,:] = rd_std[1]
            tab_rs_dff[k,i,j,:] = rs_dff[2]
            tab_rd_std[k,i,j,:] = rd_std[2]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m5_3ag.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Mental MAB agents

fd = 'ipd1_m5_mMAB'
nMemory = 5
T=50
ALGS = mMAB_algs
tab_r = np.zeros((len(ALGS),len(ALGS),T))
tab_r_std = np.zeros((len(ALGS),len(ALGS),T))
tab_p = np.zeros((len(ALGS),len(ALGS),T))
tab_p_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_sum = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_dff = np.zeros((len(ALGS),len(ALGS),T))
tab_rd_std = np.zeros((len(ALGS),len(ALGS),T))

for i,alg1 in enumerate(ALGS):
    for j,alg2 in enumerate(ALGS):
        print(i,j,alg1,alg2)
        r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2],nMemory=nMemory,prefix=fd)
        tab_r[i,j,:] = r[0]
        tab_r_std[i,j,:] = r_std[0]
        tab_p[i,j,:] = p[0]
        tab_p_std[i,j,:] = p_std[0]
        tab_r[j,i,:] = r[1]
        tab_r_std[j,i,:] = r_std[1]
        tab_p[j,i,:] = p[1]
        tab_p_std[j,i,:] = p_std[1]
        tab_rs_sum[i,j,:] = tab_rs_sum[j,i,:] = rs_sum
        tab_rs_std[i,j,:] = tab_rs_std[j,i,:] = rs_std
        tab_rs_dff[i,j,:] = rs_dff[0]
        tab_rd_std[i,j,:] = rd_std[0]
        tab_rs_dff[j,i,:] = rs_dff[1]
        tab_rd_std[j,i,:] = rd_std[1]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m5_mMAB.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Mental CB agents

fd = 'ipd1_m5_mCB'
nMemory = 5
T=50
ALGS = mCB_algs
tab_r = np.zeros((len(ALGS),len(ALGS),T))
tab_r_std = np.zeros((len(ALGS),len(ALGS),T))
tab_p = np.zeros((len(ALGS),len(ALGS),T))
tab_p_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_sum = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_dff = np.zeros((len(ALGS),len(ALGS),T))
tab_rd_std = np.zeros((len(ALGS),len(ALGS),T))

for i,alg1 in enumerate(ALGS):
    for j,alg2 in enumerate(ALGS):
        print(i,j,alg1,alg2)
        r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2],nMemory=nMemory,prefix=fd)
        tab_r[i,j,:] = r[0]
        tab_r_std[i,j,:] = r_std[0]
        tab_p[i,j,:] = p[0]
        tab_p_std[i,j,:] = p_std[0]
        tab_r[j,i,:] = r[1]
        tab_r_std[j,i,:] = r_std[1]
        tab_p[j,i,:] = p[1]
        tab_p_std[j,i,:] = p_std[1]
        tab_rs_sum[i,j,:] = tab_rs_sum[j,i,:] = rs_sum
        tab_rs_std[i,j,:] = tab_rs_std[j,i,:] = rs_std
        tab_rs_dff[i,j,:] = rs_dff[0]
        tab_rd_std[i,j,:] = rd_std[0]
        tab_rs_dff[j,i,:] = rs_dff[1]
        tab_rd_std[j,i,:] = rd_std[1]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m5_mCB.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Mental RL agents

fd = 'ipd1_m5_mRL'
nMemory = 5
T=50
ALGS = mRL_algs
tab_r = np.zeros((len(ALGS),len(ALGS),T))
tab_r_std = np.zeros((len(ALGS),len(ALGS),T))
tab_p = np.zeros((len(ALGS),len(ALGS),T))
tab_p_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_sum = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_dff = np.zeros((len(ALGS),len(ALGS),T))
tab_rd_std = np.zeros((len(ALGS),len(ALGS),T))

for i,alg1 in enumerate(ALGS):
    for j,alg2 in enumerate(ALGS):
        print(i,j,alg1,alg2)
        r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2],nMemory=nMemory,prefix=fd)
        tab_r[i,j,:] = r[0]
        tab_r_std[i,j,:] = r_std[0]
        tab_p[i,j,:] = p[0]
        tab_p_std[i,j,:] = p_std[0]
        tab_r[j,i,:] = r[1]
        tab_r_std[j,i,:] = r_std[1]
        tab_p[j,i,:] = p[1]
        tab_p_std[j,i,:] = p_std[1]
        tab_rs_sum[i,j,:] = tab_rs_sum[j,i,:] = rs_sum
        tab_rs_std[i,j,:] = tab_rs_std[j,i,:] = rs_std
        tab_rs_dff[i,j,:] = rs_dff[0]
        tab_rd_std[i,j,:] = rd_std[0]
        tab_rs_dff[j,i,:] = rs_dff[1]
        tab_rd_std[j,i,:] = rd_std[1]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m5_mRL.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Behavioral Cloning

data = pd.read_csv('./data/all_data.csv')
trajs = np.array(data[data['period']==10].iloc[:,9:27]) # (8258, 18)
np.random.shuffle(trajs)
trajs = trajs.reshape((trajs.shape[0],2,9)) # (8258, 2, 9)
trajs[trajs==0] = -1
split = 8000
train_set = trajs[:split,:,:]
test_set = trajs[split:,:,:]
full_data = {'train':train_set,'test':test_set}
with open('./data/processed_train_test.pkl', 'wb') as handle:
    pickle.dump(full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

ALGS = agent_algs
fd = 'bclone_m5'
T = 9
nTrials = 10
nMemory = 5
ipd_scenario = 1
test_size = test_set.shape[0]
tab_r = np.zeros((test_size,len(ALGS),T))
tab_r_std = np.zeros((test_size,len(ALGS),T))
tab_p = np.zeros((test_size,len(ALGS),T))
tab_p_std = np.zeros((test_size,len(ALGS),T))
tab_rs_sum = np.zeros((test_size,len(ALGS),T))
tab_rs_std = np.zeros((test_size,len(ALGS),T))
tab_rs_dff = np.zeros((test_size,len(ALGS),T))
tab_rd_std = np.zeros((test_size,len(ALGS),T))
tab_pr = np.zeros((test_size,len(ALGS)))

for i,alg1 in enumerate(ALGS):
    algs = [alg1,'Human']
    _,reward_from_A,reward_from_B,reward_from_C,reward_from_D = load_IPD(ipd_scenario,prefix=fd)
    reward_functions = (reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    ipd_case = IPD(algs,reward_functions,nTrials,T,nMemory=nMemory)
    for j in np.arange(train_set.shape[0]):
        print('training: ',i,alg1,j)
        train_data = train_set[j,:,:]
        ipd_case.loadTraj(train_data,True)
        rep = ipd_case.run()
    ipd_case.pauseLearn()
    with open('./models/'+fd+'/trained_IPD_'+str(ipd_scenario)+'_m_'+str(nMemory)+'_p_'+ '_'.join(algs)+'.pkl', 'wb') as handle:
        pickle.dump(rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for k in np.arange(test_size):
        print('testing: ',i,alg1,k)
        test_data = test_set[k,:,:]
        p_cor = []
        for t in np.arange(T):
            p_cor.append(np.sum(test_data[0,:t+1]==1)*100/(t+1))
        print(p_cor)
        p_cor = np.array(p_cor)
        ipd_case.loadTraj(train_data,True)
        rep = ipd_case.run()
        rep['algs'] = algs
        rep['nTrials'] = nTrials
        rep['T'] = T
        rep['min_r'] = np.min([np.sum(reward_from_A(0)),np.sum(reward_from_B(0)),np.sum(reward_from_C(0)),np.sum(reward_from_D(0))])
        rep['max_r'] = np.max([np.sum(reward_from_A(0)),np.sum(reward_from_B(0)),np.sum(reward_from_C(0)),np.sum(reward_from_D(0))])
        fig_p = './figures/'+fd+'/test_'+str(k)+'_p_IPD_'+str(ipd_scenario)+'_m_'+str(nMemory)+'_'+ '_'.join(algs)
        plot_p(rep,fig_p)
        rs,r,p,r_std,p_std = [],[],[],[],[]
        for l in np.arange(len(algs)):
            rs.append(norm_r(rep['reward'+str(l)],rep['min_r'],rep['max_r']))
            r.append(np.cumsum(np.mean(rs[-1],0)))
            p.append(np.mean(rep['percent'+str(l)],0))
            r_std.append(np.std(rs[-1],0)/np.sqrt(nTrials))
            p_std.append(np.std(rep['percent'+str(l)],0)/np.sqrt(nTrials))
        r,p,r_std,p_std,rs = np.array(r),np.array(p),np.array(r_std),np.array(p_std),np.array(rs)
        r_sum = np.sum(rs,0)
        rs_sum = np.mean(r_sum,0)
        rs_std = np.std(r_sum,0)/np.sqrt(nTrials)
        p_dff = p - p_cor
        rs_dff = np.mean(p_dff,1)
        rd_std = np.std(p_dff,1)/np.sqrt(nTrials)
        tab_r[k,i,:] = r[0]
        tab_r_std[k,i,:] = r_std[0]
        tab_p[k,i,:] = p[0]
        tab_p_std[k,i,:] = p_std[0]
        tab_rs_sum[k,i,:] = rs_sum
        tab_rs_std[k,i,:] = rs_std
        tab_rs_dff[k,i,:] = rs_dff[0]
        tab_rd_std[k,i,:] = rd_std[0]
        tab_pr[k,i] = pearsonr(p[0],p_cor)[0] if np.isnan(pearsonr(p[0],p_cor)[0]) else 0
        # print("pearsonr:",pearsonr(p[0],p_cor)[0])
        
tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std,'pr':tab_pr}
with open('./models/bclone_m5.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)



# Case with 2 agents

fd = 'ipd1_m1'
nMemory = 1
T = 50
ALGS = all_algs
tab_r = np.zeros((len(ALGS),len(ALGS),T))
tab_r_std = np.zeros((len(ALGS),len(ALGS),T))
tab_p = np.zeros((len(ALGS),len(ALGS),T))
tab_p_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_sum = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_dff = np.zeros((len(ALGS),len(ALGS),T))
tab_rd_std = np.zeros((len(ALGS),len(ALGS),T))

for i,alg1 in enumerate(ALGS):
    for j,alg2 in enumerate(ALGS):
        print(i,j,alg1,alg2)
        r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2],nMemory=nMemory,prefix=fd)
        tab_r[i,j,:] = r[0]
        tab_r_std[i,j,:] = r_std[0]
        tab_p[i,j,:] = p[0]
        tab_p_std[i,j,:] = p_std[0]
        tab_r[j,i,:] = r[1]
        tab_r_std[j,i,:] = r_std[1]
        tab_p[j,i,:] = p[1]
        tab_p_std[j,i,:] = p_std[1]
        tab_rs_sum[i,j,:] = tab_rs_sum[j,i,:] = rs_sum
        tab_rs_std[i,j,:] = tab_rs_std[j,i,:] = rs_std
        tab_rs_dff[i,j,:] = rs_dff[0]
        tab_rd_std[i,j,:] = rd_std[0]
        tab_rs_dff[j,i,:] = rs_dff[1]
        tab_rd_std[j,i,:] = rd_std[1]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m1.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Case with 3 agents

fd = 'ipd1_m1_3ag'
nMemory = 1
T=50
ALGS1 = MAB_algs
ALGS2 = CB_algs
ALGS3 = RL_algs
ALGSALL = agent_algs
MAB_range = np.arange(5)
CB_range = np.arange(5,9)
RL_range = np.arange(9,13)
tab_r = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_r_std = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_p = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_p_std = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_rs_sum = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_rs_std = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_rs_dff = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))
tab_rd_std = np.zeros((len(ALGSALL),len(ALGSALL),len(ALGSALL),T))

for it,alg1 in enumerate(ALGS1):
    for jt,alg2 in enumerate(ALGS2):
        for kt,alg3 in enumerate(ALGS3):
            i = MAB_range[it]
            j = CB_range[jt]
            k = RL_range[kt]
            print(i,j,k,alg1,alg2,alg3)
            r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2,alg3],nMemory=nMemory,prefix=fd)
            tab_r[i,j,k,:] = r[0]
            tab_r_std[i,j,k,:] = r_std[0]
            tab_p[i,j,k,:] = p[0]
            tab_p_std[i,j,k,:] = p_std[0]
            tab_r[j,k,i,:] = r[1]
            tab_r_std[j,k,i,:] = r_std[1]
            tab_p[j,k,i,:] = p[1]
            tab_p_std[j,k,i,:] = p_std[1]
            tab_r[k,i,j,:] = r[2]
            tab_r_std[k,i,j,:] = r_std[2]
            tab_p[k,i,j,:] = p[2]
            tab_p_std[k,i,j,:] = p_std[2]
            tab_rs_sum[i,j,k,:] = tab_rs_sum[j,k,i,:] = tab_rs_sum[k,i,j,:] = rs_sum
            tab_rs_std[i,j,k,:] = tab_rs_std[j,k,i,:] = tab_rs_std[k,i,j,:] = rs_std
            tab_rs_dff[i,j,k,:] = rs_dff[0]
            tab_rd_std[i,j,k,:] = rd_std[0]
            tab_rs_dff[j,k,i,:] = rs_dff[1]
            tab_rd_std[j,k,i,:] = rd_std[1]
            tab_rs_dff[k,i,j,:] = rs_dff[2]
            tab_rd_std[k,i,j,:] = rd_std[2]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m1_3ag.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Mental MAB agents

fd = 'ipd1_m1_mMAB'
nMemory = 1
T=50
ALGS = mMAB_algs
tab_r = np.zeros((len(ALGS),len(ALGS),T))
tab_r_std = np.zeros((len(ALGS),len(ALGS),T))
tab_p = np.zeros((len(ALGS),len(ALGS),T))
tab_p_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_sum = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_dff = np.zeros((len(ALGS),len(ALGS),T))
tab_rd_std = np.zeros((len(ALGS),len(ALGS),T))

for i,alg1 in enumerate(ALGS):
    for j,alg2 in enumerate(ALGS):
        print(i,j,alg1,alg2)
        r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2],nMemory=nMemory,prefix=fd)
        tab_r[i,j,:] = r[0]
        tab_r_std[i,j,:] = r_std[0]
        tab_p[i,j,:] = p[0]
        tab_p_std[i,j,:] = p_std[0]
        tab_r[j,i,:] = r[1]
        tab_r_std[j,i,:] = r_std[1]
        tab_p[j,i,:] = p[1]
        tab_p_std[j,i,:] = p_std[1]
        tab_rs_sum[i,j,:] = tab_rs_sum[j,i,:] = rs_sum
        tab_rs_std[i,j,:] = tab_rs_std[j,i,:] = rs_std
        tab_rs_dff[i,j,:] = rs_dff[0]
        tab_rd_std[i,j,:] = rd_std[0]
        tab_rs_dff[j,i,:] = rs_dff[1]
        tab_rd_std[j,i,:] = rd_std[1]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m1_mMAB.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Mental CB agents

fd = 'ipd1_m1_mCB'
nMemory = 1
T=50
ALGS = mCB_algs
tab_r = np.zeros((len(ALGS),len(ALGS),T))
tab_r_std = np.zeros((len(ALGS),len(ALGS),T))
tab_p = np.zeros((len(ALGS),len(ALGS),T))
tab_p_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_sum = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_dff = np.zeros((len(ALGS),len(ALGS),T))
tab_rd_std = np.zeros((len(ALGS),len(ALGS),T))

for i,alg1 in enumerate(ALGS):
    for j,alg2 in enumerate(ALGS):
        print(i,j,alg1,alg2)
        r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2],nMemory=nMemory,prefix=fd)
        tab_r[i,j,:] = r[0]
        tab_r_std[i,j,:] = r_std[0]
        tab_p[i,j,:] = p[0]
        tab_p_std[i,j,:] = p_std[0]
        tab_r[j,i,:] = r[1]
        tab_r_std[j,i,:] = r_std[1]
        tab_p[j,i,:] = p[1]
        tab_p_std[j,i,:] = p_std[1]
        tab_rs_sum[i,j,:] = tab_rs_sum[j,i,:] = rs_sum
        tab_rs_std[i,j,:] = tab_rs_std[j,i,:] = rs_std
        tab_rs_dff[i,j,:] = rs_dff[0]
        tab_rd_std[i,j,:] = rd_std[0]
        tab_rs_dff[j,i,:] = rs_dff[1]
        tab_rd_std[j,i,:] = rd_std[1]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m1_mCB.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Mental RL agents

fd = 'ipd1_m1_mRL'
nMemory = 1
T=50
ALGS = mRL_algs
tab_r = np.zeros((len(ALGS),len(ALGS),T))
tab_r_std = np.zeros((len(ALGS),len(ALGS),T))
tab_p = np.zeros((len(ALGS),len(ALGS),T))
tab_p_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_sum = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_std = np.zeros((len(ALGS),len(ALGS),T))
tab_rs_dff = np.zeros((len(ALGS),len(ALGS),T))
tab_rd_std = np.zeros((len(ALGS),len(ALGS),T))

for i,alg1 in enumerate(ALGS):
    for j,alg2 in enumerate(ALGS):
        print(i,j,alg1,alg2)
        r,r_std,p,p_std,rs_sum,rs_std,rs_dff,rd_std,rep = run_ipd(1,[alg1,alg2],nMemory=nMemory,prefix=fd)
        tab_r[i,j,:] = r[0]
        tab_r_std[i,j,:] = r_std[0]
        tab_p[i,j,:] = p[0]
        tab_p_std[i,j,:] = p_std[0]
        tab_r[j,i,:] = r[1]
        tab_r_std[j,i,:] = r_std[1]
        tab_p[j,i,:] = p[1]
        tab_p_std[j,i,:] = p_std[1]
        tab_rs_sum[i,j,:] = tab_rs_sum[j,i,:] = rs_sum
        tab_rs_std[i,j,:] = tab_rs_std[j,i,:] = rs_std
        tab_rs_dff[i,j,:] = rs_dff[0]
        tab_rd_std[i,j,:] = rd_std[0]
        tab_rs_dff[j,i,:] = rs_dff[1]
        tab_rd_std[j,i,:] = rd_std[1]

tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std}
with open('./models/ipd1_m1_mRL.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Behavioral Cloning

data = pd.read_csv('./data/all_data.csv')
trajs = np.array(data[data['period']==10].iloc[:,9:27]) # (8258, 18)
np.random.shuffle(trajs)
trajs = trajs.reshape((trajs.shape[0],2,9)) # (8258, 2, 9)
trajs[trajs==0] = -1
split = 8000
train_set = trajs[:split,:,:]
test_set = trajs[split:,:,:]
full_data = {'train':train_set,'test':test_set}
with open('./data/processed_train_test.pkl', 'wb') as handle:
    pickle.dump(full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

ALGS = agent_algs
fd = 'bclone_m1'
T = 9
nTrials = 10
nMemory = 1
ipd_scenario = 1
test_size = test_set.shape[0]
tab_r = np.zeros((test_size,len(ALGS),T))
tab_r_std = np.zeros((test_size,len(ALGS),T))
tab_p = np.zeros((test_size,len(ALGS),T))
tab_p_std = np.zeros((test_size,len(ALGS),T))
tab_rs_sum = np.zeros((test_size,len(ALGS),T))
tab_rs_std = np.zeros((test_size,len(ALGS),T))
tab_rs_dff = np.zeros((test_size,len(ALGS),T))
tab_rd_std = np.zeros((test_size,len(ALGS),T))
tab_pr = np.zeros((test_size,len(ALGS)))

for i,alg1 in enumerate(ALGS):
    algs = [alg1,'Human']
    _,reward_from_A,reward_from_B,reward_from_C,reward_from_D = load_IPD(ipd_scenario,prefix=fd)
    reward_functions = (reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    ipd_case = IPD(algs,reward_functions,nTrials,T,nMemory=nMemory)
    for j in np.arange(train_set.shape[0]):
        print('training: ',i,alg1,j)
        train_data = train_set[j,:,:]
        ipd_case.loadTraj(train_data,True)
        rep = ipd_case.run()
    ipd_case.pauseLearn()
    with open('./models/'+fd+'/trained_IPD_'+str(ipd_scenario)+'_m_'+str(nMemory)+'_p_'+ '_'.join(algs)+'.pkl', 'wb') as handle:
        pickle.dump(rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for k in np.arange(test_size):
        print('testing: ',i,alg1,k)
        test_data = test_set[k,:,:]
        p_cor = []
        for t in np.arange(T):
            p_cor.append(np.sum(test_data[0,:t+1]==1)*100/(t+1))
        print(p_cor)
        p_cor = np.array(p_cor)
        ipd_case.loadTraj(train_data,True)
        rep = ipd_case.run()
        rep['algs'] = algs
        rep['nTrials'] = nTrials
        rep['T'] = T
        rep['min_r'] = np.min([np.sum(reward_from_A(0)),np.sum(reward_from_B(0)),np.sum(reward_from_C(0)),np.sum(reward_from_D(0))])
        rep['max_r'] = np.max([np.sum(reward_from_A(0)),np.sum(reward_from_B(0)),np.sum(reward_from_C(0)),np.sum(reward_from_D(0))])
        fig_p = './figures/'+fd+'/test_'+str(k)+'_p_IPD_'+str(ipd_scenario)+'_m_'+str(nMemory)+'_'+ '_'.join(algs)
        plot_p(rep,fig_p)
        rs,r,p,r_std,p_std = [],[],[],[],[]
        for l in np.arange(len(algs)):
            rs.append(norm_r(rep['reward'+str(l)],rep['min_r'],rep['max_r']))
            r.append(np.cumsum(np.mean(rs[-1],0)))
            p.append(np.mean(rep['percent'+str(l)],0))
            r_std.append(np.std(rs[-1],0)/np.sqrt(nTrials))
            p_std.append(np.std(rep['percent'+str(l)],0)/np.sqrt(nTrials))
        r,p,r_std,p_std,rs = np.array(r),np.array(p),np.array(r_std),np.array(p_std),np.array(rs)
        r_sum = np.sum(rs,0)
        rs_sum = np.mean(r_sum,0)
        rs_std = np.std(r_sum,0)/np.sqrt(nTrials)
        p_dff = p - p_cor
        rs_dff = np.mean(p_dff,1)
        rd_std = np.std(p_dff,1)/np.sqrt(nTrials)
        tab_r[k,i,:] = r[0]
        tab_r_std[k,i,:] = r_std[0]
        tab_p[k,i,:] = p[0]
        tab_p_std[k,i,:] = p_std[0]
        tab_rs_sum[k,i,:] = rs_sum
        tab_rs_std[k,i,:] = rs_std
        tab_rs_dff[k,i,:] = rs_dff[0]
        tab_rd_std[k,i,:] = rd_std[0]
        tab_pr[k,i] = pearsonr(p[0],p_cor)[0] if np.isnan(pearsonr(p[0],p_cor)[0]) else 0
        # print("pearsonr:",pearsonr(p[0],p_cor)[0])
        
tab = {'r':tab_r,'rstd':tab_r_std,'p':tab_p,'pstd':tab_p_std,'s':tab_rs_sum,'sstd':tab_rs_std,'d':tab_rs_dff,'dstd':tab_rd_std,'pr':tab_pr}
with open('./models/bclone_m1.pkl', 'wb') as handle:
    pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)



