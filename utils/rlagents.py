import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

# RL Algorithms:
# QL: Q-learning
# DQL: Double Q-learning
# SQL: Split Q-learning
# EDQL: Exponential Double Q-learning
# ESQL: Exponential Split Q-learning
# PQL: Positive Q-learning
# NQL: Negative Q-learning
# SARSA: State–Action–Reward–State–Action
# MP: MaxPain
# ADD: SQL for Addiction
# AD: SQL for Alzheimer's Disease
# ADHD: SQL for Attention-Deficit Hyperactivity Disorder
# PD: SQL for Parkinson's Disorder
# bvFTD: SQL for behavioral variant of FrontoTemporal Dementia
# CP: SQL for Chronic Pain
# M: SQL for Moderate

# Bandits Algorithms:
# TS: Thompson Sampling
# UCB: Upper Confidence Bound (implemented UCB1)
# HBTS: Human Behavior Thompson Sampling (Split Thompson Sampling)
# ETS: Exponential Thompson Sampling
# EHBTS: Exponential Human Behavior Thompson Sampling
# PTS: Positive Thompson Sampling
# NTS: Negative Thompson Sampling
# eGreedy: epsilon Greedy
# EXP3: EXP3
# EXP30: EXP3 with mu set as zero
# bADD: HBTS for Addiction
# bAD: HBTS for Alzheimer's Disease
# bADHD: HBTS for Attention-Deficit Hyperactivity Disorder
# bPD: HBTS for Parkinson's Disorder
# bbvFTD: HBTS for behavioral variant of FrontoTemporal Dementia
# bCP: HBTS for Chronic Pain
# bM: HBTS for Moderate

# Contexutal Bandits Algorithms:
# CTS: Contextual Thompson Sampling
# LinUCB: Upper Confidence Bound (implemented UCB1)
# SCTS: Split Contextual Thompson Sampling
# EXP4: EXP4
# PCTS: Positive Contextual Thompson Sampling
# NCTS: Negative Contextual Thompson Sampling
# cADD: SCTS for Addiction
# cAD: SCTS for Alzheimer's Disease
# cADHD: SCTS for Attention-Deficit Hyperactivity Disorder
# cPD: SCTS for Parkinson's Disorder
# cbvFTD: SCTS for behavioral variant of FrontoTemporal Dementia
# cCP: SCTS for Chronic Pain
# cM: SCTS for Moderate

# (Only for IPD) Hand-crafted strategies:
# Coop: always cooperate
# Dfct: always defect
# Tit4Tat: tit-for-tat

# Behavioral cloning:
# player 0 is by default our agents
# player 1 and so on can be behavioral trajectories from human data

class MDP():
    """
    MDP game setting
    """
    def __init__(self,algorithm,reward_functions,nTrials,T,nAct_B=20,nAct_C=20,GAMMA=0.95,Q1=None,Q2=None,Q1s=None,Q2s=None,Traj=None):

        self.nTrials= nTrials # number of experiments to run, large number means longer execution time
        self.T = T            # number of episodes per experiment, large number means longer execution time
        self.algorithm = algorithm
        self.STATE_A,self.STATE_B,self.STATE_C,self.STATE_D,self.STATE_E = 0,1,2,3,4
        self.ACTION_LEFT,self.ACTION_RIGHT,self.ACTION_DUMMY = 0,1,-1
        self.GAMMA = GAMMA
        self.reward_from_B = reward_functions[0]
        self.reward_from_C = reward_functions[1]
        self.nArms = 2
        self.initialState = self.STATE_A
        self.nP = 1
        
        if Q1 is not None:
            self.Q1 = Q1
        if Q2 is not None:
            self.Q2 = Q2
        if Q1s is not None:
            self.Q1s = Q1s
        if Q2s is not None:
            self.Q2s = Q2s
        if Traj is not None:
            self.trajs = Traj

        # behavioral cloning
        self.BC = False # behavioral cloning
        self.pauseLearning = False # for testing phase
        self.trained = False # to prevent from picking a random from start
        self.epsilon = 0.05 # to be turned off during testing phase

        # map actions to states
        self.actionsPerState = {}
        self.actionsPerState[self.STATE_A] = [self.ACTION_LEFT, self.ACTION_RIGHT]
        self.actionsPerState[self.STATE_B] = [i for i in range(nAct_B)]
        self.actionsPerState[self.STATE_C] = [i for i in range(nAct_C)]
        self.actionsPerState[self.STATE_D] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_E] = [self.ACTION_DUMMY]
        self.stateSpace = [self.STATE_A,self.STATE_B,self.STATE_C,self.STATE_D,self.STATE_E]
        self.fQprime = None

        # in MDP and MAB, the context is simply the one hot of current state
        self.feats = np.zeros((len(self.stateSpace),1))
        
        # init Q values
        self.Q1,self.Q2 = {},{}
        self.Q1s,self.Q2s = [],[]

    def pauseLearn(self,toSet=True):
        self.pauseLearning = toSet
        self.epsilon = 0

    def loadTraj(self,Traj,BC=False):
        self.trajs = Traj
        self.T = Traj.shape[1]
        self.BC = BC
        self.trained = True if BC else False

    def getQfunctions(self,player=None):
        Q1 = self.Q1 if player is None else self.Q1s[player]
        Q2 = self.Q2 if player is None else self.Q2s[player]
        return Q1, Q2
        
    def getQaddition(self,pw=1,nw=1,player=None):
        Qprime = {}
        Q1, Q2 = self.getQfunctions(player)
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                Qprime[s][a] = pw * Q1[s][a] + nw * Q2[s][a]
        return Qprime

    def getQbeta(self,player=None):
        Qprime = {}
        Q1, Q2 = self.getQfunctions(player)
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                q1 = Q1[s][a] if Q1[s][a] > 0 else 1
                q2 = Q2[s][a] if Q2[s][a] > 0 else 1
                Qprime[s][a] = np.random.beta(q1,q2)
        return Qprime

    def getQUCB(self,N,NSA,current_a,player=None):
        Qprime = {}
        Q1, Q2 = self.getQfunctions(player)
        for s in self.stateSpace:
            Qprime[s] = {}
            if N[s] < len(self.actionsPerState[s]):
                for a in self.actionsPerState[s]:
                    Qprime[s][a] = 0
                Qprime[s][self.actionsPerState[s][N[s]]] = 1
            else:
                for a in self.actionsPerState[s]:
                    if NSA[s][a] == 0: NSA[s][a] = 1
                    if N[s] is None: N[s] = 0
                    if a == current_a: 
                        Qprime[s][a] = Q1[s][a] * (NSA[s][a]-1) / NSA[s][a] + np.sqrt(2*np.log(N[s])/NSA[s][a])
                    else: 
                        Qprime[s][a] = Q1[s][a] + np.sqrt(2*np.log(N[s])/NSA[s][a])

        return Qprime
    
    def getQLinUCB(self,N,NSA,current_a,player=None,alpha=0.05):
        Qprime = {}
        A, b = self.getQfunctions(player)
        for s in self.stateSpace:
            Qprime[s] = {}
            if N[s] < len(self.actionsPerState[s]):
                for a in self.actionsPerState[s]:
                    Qprime[s][a] = 0
                Qprime[s][self.actionsPerState[s][N[s]]] = 1
            else:
                for a in self.actionsPerState[s]:
                    theta = np.linalg.inv(A[s][a]).dot(b[s][a])
                    Qprime[s][a] = theta.T.dot(self.feats.flatten()) + alpha*np.sqrt(self.feats.flatten().T.dot(np.linalg.inv(A[s][a])).dot(self.feats.flatten()))
        return Qprime
    
    def getQCTS(self,Q,NSA,player=None):
        gR = 0.5
        gepsilon = 0.05
        gdelta = 0.1
        v2 = float((gR**2) * 24 * 1 * math.log(1./gdelta) * (1./gepsilon))
        Qprime = {}
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                B, f = Q[s][a][0], Q[s][a][1]
                try:
                    Qprime[s][a] = self.feats.flatten().T.dot(np.random.multivariate_normal(np.linalg.inv(B).dot(f).flatten(), v2 * np.linalg.inv(B)))
                except:
                    Qprime[s][a] = self.feats.flatten().T.dot(np.random.multivariate_normal(np.linalg.inv(B+np.eye(len(self.feats.flatten()))).dot(f).flatten(), v2 * np.linalg.inv(B+np.eye(len(self.feats.flatten())))))
        return Qprime
        
    def getQSCTS(self,NSA,pw=1,nw=1,player=None):
        Qprime = {}
        Q1, Q2 = self.getQfunctions(player)
        Q1t = self.getQCTS(Q1,NSA)
        Q2t = self.getQCTS(Q2,NSA)
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                Qprime[s][a] = pw*Q1t[s][a] + nw*Q2t[s][a]
        return Qprime
    
    def getQEXP4(self,mu,player=None):
        Qprime = {}
        Q1, Y0 = self.getQfunctions(player)
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                p = self.feats.flatten().dot(Q1[s][a]) / np.sum(Q1[s][a])
                Qprime[s][a] = (1-mu) * p + mu / len(self.actionsPerState[s])
        return Qprime

    def getQGreedy(self,N,NSA,current_a,player=None):
        Qprime = {}
        Q1, Q2 = self.getQfunctions(player)
        for s in self.stateSpace:
            Qprime[s] = {}
            if N[s] < len(self.actionsPerState[s]):
                for a in self.actionsPerState[s]:
                    Qprime[s][a] = 0
                Qprime[s][self.actionsPerState[s][N[s]]] = 1
            else:
                for a in self.actionsPerState[s]:
                    if a == current_a:
                        Qprime[s][a] = Q1[s][a] * (NSA[s][a]-1) / NSA[s][a]
                    else:
                        Qprime[s][a] = Q1[s][a]
        return Qprime
   
    def getQEXP3(self,mu,player=None):
        Qprime = {}
        Q1, Q2 = self.getQfunctions(player)
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                Qprime[s][a] = (1-mu) * Q1[s][a] / np.sum(list(Q1[s].values())) + mu / len(self.actionsPerState[s])
        return Qprime

    def resetQprimeFunction(self,alg=None):
        if alg is None: alg = self.algorithm
        if alg in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M']:
            self.fQprime = self.getQaddition
        elif alg in ['UCB']:
            self.fQprime = self.getQUCB
        elif alg in ['LinUCB']:
            self.fQprime = self.getQLinUCB
        elif alg in ['eGreedy']:
            self.fQprime = self.getQGreedy
        elif alg in ['EXP3','EXP30']:
            self.fQprime = self.getQEXP3
        elif alg in ['EXP4']:
            self.fQprime = self.getQEXP4
        elif alg in ['CTS']:
            self.fQprime = self.getQCTS
        elif alg in ['SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
            self.fQprime = self.getQSCTS
        else: 
            self.fQprime = self.getQbeta
        return self.fQprime
    
    # reset the variables, to be called on each experiment
    def reset(self):
        self.resetQprimeFunction()
        if self.algorithm in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M','CTS','SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
            defaultQ = 0
        else:
            defaultQ = 1
        for s in self.stateSpace:
            self.Q1[s],self.Q2[s] = {},{}
            for a in self.actionsPerState[s]:
                if self.algorithm in ['LinUCB']:
                    self.Q1[s][a] = np.eye(len(self.feats.flatten()))
                    self.Q2[s][a] = np.zeros((len(self.feats.flatten()),1))
                elif self.algorithm in ['EXP4']:
                    self.Q1[s][a] = np.ones((len(self.feats.flatten()),1))/len(self.feats.flatten())
                    self.Q2[s][a] = np.zeros((len(self.feats.flatten()),1))
                elif self.algorithm in ['CTS','SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
                    self.Q1[s][a] = [np.eye(len(self.feats.flatten())),np.zeros((len(self.feats.flatten()),1))]
                    self.Q2[s][a] = [np.eye(len(self.feats.flatten())),np.zeros((len(self.feats.flatten()),1))]
                else:
                    self.Q1[s][a] = defaultQ
                    self.Q2[s][a] = defaultQ
      
    # epsilon greedy action
    def randomAction(self,s,a,eps=.1):
        p = np.random.random()
        if p < (1 - eps): return a
        else: return np.random.choice(self.actionsPerState[s])
      
    # move from state s using action a
    def move(self,s,a,t=None):
        if self.BC and self.trajs is not None:
            if(s==self.STATE_A):
                a_cor = self.trajs[0,self.T-t-1]
                if(a == a_cor): return 1, a_cor
                else: return 0, a_cor
            if s==self.STATE_B: return 1, self.STATE_D
            if s==self.STATE_C: return 1, self.STATE_E
        else:
            if(s==self.STATE_A):
                if(a == self.ACTION_LEFT): return 0, self.STATE_B
                elif(a == self.ACTION_RIGHT): return 0, self.STATE_C
            if s==self.STATE_B: return self.reward_from_B(), self.STATE_D
            if s==self.STATE_C: return self.reward_from_C(), self.STATE_E
        return 0, s
    
    # returns the action that makes the max Q value, as well as the max Q value
    def maxQA(self,q,s):
        max=float('-inf')
        sa = 0
        if len(q[s]) == 1:
            for k in q[s]: return k,q[s][k]
        for k in q[s]:
            if(q[s][k] > max):
                max = q[s][k]
                sa = k
            elif(q[s][k] == max):
                if(np.random.random() < 0.5): sa = k
        return sa, max
    
    # return true if this is a terminal state
    def isTerminal(self,s):
        return s == self.STATE_E or s == self.STATE_D

    # do the experiment by running T episodes and fill the results in the episodes parameter
    def experiment(self):
        episodes = {} 
        self.reset()
        ALeft = 0 #contains the number of times left action is chosen at A
        
        N = {} # contains the number of visits for each state
        for s in self.stateSpace: N[s] = 0

        NSA = {} # contains the number of visits for each state and action
        for s in self.stateSpace: 
            NSA[s] = {}
            for a in self.actionsPerState[s]:
                NSA[s][a] = 0
        
        t = 0    
        reward,pos_reward,neg_reward,actions = None,None,None,None

        last_a = None
            
        # loop for T episodes
        for i in range(self.T):
            gameover = False
            
            s = self.initialState
            if i == 0: a = self.selectInitialAction(self.initialState,True,last_a,N,NSA,t=i)
            else: a = self.selectInitialAction(self.initialState,False,last_a,N,NSA,t=i)
            
            #loop until game is over, this will be ONE episode
            while not gameover:
                actions = a
                t += 1  # record learning steps
                
                if self.algorithm in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M']:
                    a = self.randomAction(s, a, self.epsilon) # apply epsilon greedy selection (including for action chosen at STATE A)
                if self.algorithm == 'eGreedy' and N[s] > len(self.actionsPerState[s]):
                    a = self.randomAction(s, a, self.epsilon) # apply epsilon greedy selection (including for action chosen at STATE A)
                
                N[s] += 1 #update the number of visits for state s

                # if left action is chosen at state A, increment the counter
                self.feats = np.zeros((len(self.stateSpace),1))
                self.feats[s] = 1
                if (s == self.STATE_A and a == self.ACTION_LEFT): ALeft += 1

                #move to the next state and get the reward
                r, nxt_s = self.move(s,a,i)
                reward = r
                if r > 0: pos_reward,neg_reward = r,0
                else: pos_reward,neg_reward = 0,r                
#                 reward.append(r)

                #update the number of visits per state and action
                if not s in NSA: NSA[s] = {}
                NSA[s][a] += 1

                #compute alpha
                alpha = 1 / np.power(NSA[s][a], .8)

                #update the agents and get the best action for the next state
                nxt_a = self.updateAgent(s, a, r, nxt_s, alpha, t, N=N, NSA=NSA)             

                #if next state is terminal then mark as gameover (end of episode)
                gameover = self.isTerminal(nxt_s)
                last_a = a
                s, a = nxt_s, nxt_a

            #update stats for each episode
            if not (i in episodes):
                episodes[i] = {}
                episodes[i]["count"] = 0
                episodes[i]["Q1(A)l"] = episodes[i]["Q1(A)r"] = episodes[i]["Q2(A)l"] = episodes[i]["Q2(A)r"] = 0
            
            episodes[i]["count"],episodes[i]["percent"] = ALeft, ALeft/(i+1)
            episodes[i]["reward"] = reward
            episodes[i]["pos_reward"],episodes[i]["neg_reward"] = pos_reward,neg_reward
            episodes[i]["actions"] = actions
#             episodes[i]["cumreward"] = sum(reward)
            episodes[i]["Q1(A)l"] = (episodes[i]["Q1(A)l"] * i + self.Q1[self.STATE_A][self.ACTION_LEFT])/(i+1)
            episodes[i]["Q2(A)l"] = (episodes[i]["Q2(A)l"] * i + self.Q2[self.STATE_A][self.ACTION_LEFT])/(i+1)
            episodes[i]["Q1(A)r"] = (episodes[i]["Q1(A)r"] * i + self.Q1[self.STATE_A][self.ACTION_RIGHT])/(i+1)
            episodes[i]["Q2(A)r"] = (episodes[i]["Q2(A)r"] * i + self.Q2[self.STATE_A][self.ACTION_RIGHT])/(i+1)
        
        return episodes 
        
    # run the learning
    def run(self):
        #run batch of experiments
        report = {}
        count  = np.ndarray((self.nTrials,self.T))
        percent = np.ndarray((self.nTrials,self.T))
        Q1Al = np.ndarray((self.nTrials,self.T))
        Q2Al = np.ndarray((self.nTrials,self.T))
        Q1Ar = np.ndarray((self.nTrials,self.T))
        Q2Ar = np.ndarray((self.nTrials,self.T))
        reward = np.ndarray((self.nTrials,self.T))
        cumreward = pos_reward = neg_reward = actions = np.ndarray((self.nTrials,self.T))

        for k in range(self.nTrials):
            tmp = self.experiment()
        
            #aggregate every experiment result into the final report
            for i in range(self.T):
                count[k,i] = tmp[i]["count"]
                percent[k,i] = 100*tmp[i]["count"] / (i+1)
                Q1Al[k,i] = tmp[i]["Q1(A)l"]
                Q2Al[k,i] = tmp[i]["Q2(A)l"]
                Q1Ar[k,i] = tmp[i]["Q1(A)r"]
                Q2Ar[k,i] = tmp[i]["Q2(A)r"]
#                 cumreward[k,i] = tmp[i]["cumreward"]
                reward[k,i] = tmp[i]["reward"]
                pos_reward[k,i] = tmp[i]["pos_reward"]
                neg_reward[k,i] = tmp[i]["neg_reward"]
                actions[k,i] = tmp[i]["actions"]
            
        report["count"],report["percent"]  = count,percent
        report["Q1(A)l"],report["Q2(A)l"],report["Q1(A)r"],report["Q2(A)r"] = Q1Al,Q2Al,Q1Ar,Q2Ar
#         report["cumreward"] = cumreward
        report["reward"] = reward
        report["pos_reward"],report["neg_reward"],report["actions"] = pos_reward,neg_reward,actions
        
        return report

    def selectInitialAction(self,startState,veryFirst=False,last_a=None,N=None,NSA=None,alg=None,player=None,fQprime=None,t=None):
        
        if alg is None:
            alg = self.algorithm
            fQprime = self.fQprime
            Q1 = self.Q1
            Q2 = self.Q2
        else:
            Q1,Q2 = self.getQfunctions(player)
        if alg in ['Coop','Dfct','Human']:
            a,_,_ = self.act(startState,last_a,N,NSA,alg,fQprime,Q1,Q2,player,t)
            return a
        elif veryFirst and not self.trained:
            return np.random.choice(self.actionsPerState[startState])
        else:
            a,_,_ = self.act(startState,last_a,N,NSA,alg,fQprime,Q1,Q2,player,t)
            return a

    def draw(self,p,actions):
        p = np.array(p) / np.sum(p)
        a = np.random.choice(actions, 1, list(p))[0]
        return a
    
    def getExpected(self,Q,s): 
        r = []
        for q in Q[s]: r.append(q)
        return np.mean(r)

    def getBias(self,alg=None):
        if alg is None:
            alg = self.algorithm
        if alg in ['SQL','SQL2','HBTS','MP','ESQL','EHBTS','SCTS']: p1,p2,n1,n2 = 1,1,1,1
        elif alg in ['PQL','PTS','PCTS']: p1,p2,n1,n2 = 1,1,0,0
        elif alg in ['NQL','NTS','NCTS']: p1,p2,n1,n2 = 0,0,1,1 
        elif alg in ['ADD','bADD','cADD']: p1,p2,n1,n2 = np.random.normal(1,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        elif alg in ['ADHD','bADHD','cADHD']: p1,p2,n1,n2 = np.random.normal(0.2,0.1),np.random.normal(1,0.1),np.random.normal(0.2,0.1),np.random.normal(1,0.1)
        elif alg in ['AD','bAD','cAD']: p1,p2,n1,n2 = np.random.normal(0.1,0.1),np.random.normal(1,0.1),np.random.normal(0.1,0.1),np.random.normal(1,0.1)
        elif alg in ['CP','bCP','cCP']: p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(1,0.1)
        elif alg in ['bvFTD','bbvFTD','cbvFTD']: p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(100,10),np.random.normal(0.5,0.1),np.random.normal(1,0.1)  
        elif alg in ['PD','bPD','cPD']: p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(100,10)
        elif alg in ['M','bM','cM']: p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        else: p1,p2,n1,n2 = None,None,None,None
        return p1,p2,n1,n2
        
    def act(self,s,last_a,N,NSA,alg,fQprime,Q1,Q2,player=None,t=None):

        maxq = None
        isQ1forDQL = False
        
        p1,p2,n1,n2 = self.getBias(alg)

        if alg in ['SARSA','QL']:
            nxt_a, maxq = self.maxQA(Q1, s)    

        if alg in ['DQL','EDQL']:
            p = np.random.random()
            if (p < .5): 
                nxt_a, maxq = self.maxQA(Q1, s)
                isQ1forDQL = True
            else: 
                nxt_a, maxq = self.maxQA(Q2, s)
                isQ1forDQL = False

        if alg == 'MP':
            Qprime = fQprime(player=player)
            nxt_a, maxq = self.maxQA(Qprime, s)
                                   
        if alg in ['SQL','ESQL','PQL','NQL','AD','ADD','ADHD','CP','bvFTD','PD','M']:
            Qprime = fQprime(player=player)
            nxt_a, maxq = self.maxQA(Qprime, s)
            
        if alg in ['SQL2']:
            Qprime = fQprime(pw=p2,nw=n2,player=player)
            nxt_a, maxq = self.maxQA(Qprime, s)
            
        if alg in ['EXP3','EXP4']:
            mu = 0.05
            Qprime = fQprime(mu,player=player)
            p = list(Qprime[s].values())
            nxt_a = self.draw(p,self.actionsPerState[s])

        if alg in ['EXP30']:
            mu = 0
            Qprime = fQprime(mu,player=player)
            p = list(Qprime[s].values())
            nxt_a = self.draw(p,self.actionsPerState[s])

        if alg in ['eGreedy','UCB','LinUCB']:
            Qprime = fQprime(N,NSA,last_a,player=player)
            nxt_a, maxq = self.maxQA(Qprime, s)
            
        if alg in ['SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
            Qprime = fQprime(NSA,player=player)
            nxt_a, maxq = self.maxQA(Qprime, s)

        if alg in ['CTS']:
            Qprime = fQprime(Q1,NSA,player=player)
            nxt_a, maxq = self.maxQA(Qprime, s)
            
        if alg in ['TS','ETS','HBTS','PTS','NTS','EHBTS','bADD','bADHD','bAD','bCP','bbvFTD','bPD','bM']:
            Qprime = fQprime(player=player)
            nxt_a, maxq = self.maxQA(Qprime, s)
                
        if alg in ['Coop']:
            nxt_a = self.ACTION_LEFT

        if alg in ['Dfct']:
            nxt_a = self.ACTION_RIGHT

        if alg in ['Tit4Tat']:
            nxt_a = self.ACTION_RIGHT if -1 in list(self.feats[np.arange(self.nP)!=player,0]) else self.ACTION_LEFT

        if alg in ['Human']:
            nxt_a = self.code2action(self.trajs[player,self.T-t-1])

        return nxt_a, maxq, isQ1forDQL
    
    def code2action(self,c):
        return c

    def updateAgent(self, s, a, r, nxt_s, alpha, t, pr=None, nr=None, N=None, NSA=None, alg=None, player=None, fQprime=None):
        
        if alg is None:
            alg = self.algorithm
            fQprime = self.fQprime
            Q1 = self.Q1
            Q2 = self.Q2
            noalg = True
        else:
            Q1,Q2 = self.getQfunctions(player)
            noalg = False

        p1,p2,n1,n2 = self.getBias(alg)
        
        nxt_a, maxq, isQ1forDQL = self.act(nxt_s,a,N,NSA,alg,fQprime,Q1,Q2,player,t)

        if alg == 'SARSA':
            # Q1[s][a] = Q1[s][a] + alpha * (r + self.GAMMA * self.getExpected(Q1s) - Q1[s][a])
            Q1[s][a] = Q1[s][a] + alpha * (r + self.GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        
        if alg == 'QL':
            Q1[s][a] = Q1[s][a] + alpha * (r + self.GAMMA * maxq - Q1[s][a])
    
        if alg == 'DQL':
            p = np.random.random()
            if isQ1forDQL: Q1[s][a] = Q1[s][a] + alpha * (r + self.GAMMA * Q2[nxt_s][nxt_a] - Q1[s][a])
            else: Q2[s][a] = Q2[s][a] + alpha * (r + self.GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])

        if alg == 'MP':
            if pr is not None and nr is not None:
                Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + self.GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
                Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + self.GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
            else:
                if (r >= 0): Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + self.GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
                if (r <= 0): Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + self.GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
                       
        if alg == 'EDQL':
            rho = 1
            p = np.random.random()
            if (p < .5):
                try: r = r*math.exp(r/rho)
                except: r = r
                Q1[s][a] = Q1[s][a] + alpha * (r + self.GAMMA * Q2[nxt_s][nxt_a] - Q1[s][a])
            else:
                try: r = r*math.exp(-r/rho)
                except: r = r
                Q2[s][a] = Q2[s][a] + alpha * (r + self.GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])
    
        if alg == 'ESQL':
            rho = 1    
            nxt_a1, maxq1 = self.maxQA(Q1, nxt_s)
            nxt_a2, maxq2 = self.maxQA(Q2, nxt_s)
            if pr is not None and nr is not None:
                try: pr = pr*math.exp(pr/rho)
                except: pr = pr
                Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + self.GAMMA * Q1[nxt_s][nxt_a1] - Q1[s][a])
                try: nr = nr*math.exp(-nr/rho)
                except: nr = nr
                Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + self.GAMMA * Q2[nxt_s][nxt_a2] - Q2[s][a])
            else:
                if (r >= 0):
                    try: r = r*math.exp(r/rho)
                    except: r = r
                    Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + self.GAMMA * Q1[nxt_s][nxt_a1] - Q1[s][a])
                if (r <= 0):
                    try: r = r*math.exp(-r/rho)
                    except: r = r    
                    Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + self.GAMMA * Q2[nxt_s][nxt_a2] - Q2[s][a])
        
                    
        if alg in ['SQL','PQL','NQL','AD','ADD','ADHD','CP','bvFTD','PD','M']:
            nxt_a1, maxq1 = self.maxQA(Q1, nxt_s)
            nxt_a2, maxq2 = self.maxQA(Q2, nxt_s)
            if pr is not None and nr is not None:
                Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + self.GAMMA * Q1[nxt_s][nxt_a1] - Q1[s][a])
                Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + self.GAMMA * Q2[nxt_s][nxt_a2] - Q2[s][a])
            else:
                if (r >= 0): Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + self.GAMMA * Q1[nxt_s][nxt_a1] - Q1[s][a])
                if (r <= 0): Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + self.GAMMA * Q2[nxt_s][nxt_a2] - Q2[s][a])

        if alg in ['SQL2']:
            nxt_a1, maxq1 = self.maxQA(Q1, nxt_s)
            nxt_a2, maxq2 = self.maxQA(Q2, nxt_s)
            if pr is not None and nr is not None:
                Q1[s][a] = p1*Q1[s][a] + alpha * (pr + self.GAMMA * Q1[nxt_s][nxt_a1] - Q1[s][a])
                Q2[s][a] = n1*Q2[s][a] + alpha * (nr + self.GAMMA * Q2[nxt_s][nxt_a2] - Q2[s][a])
            else:
                if (r >= 0): Q1[s][a] = p1*Q1[s][a] + alpha * (r + self.GAMMA * Q1[nxt_s][nxt_a1] - Q1[s][a])
                if (r <= 0): Q2[s][a] = n1*Q2[s][a] + alpha * (r + self.GAMMA * Q2[nxt_s][nxt_a2] - Q2[s][a])
    
        if alg in ['EHBTS']:
            rho = 1
            if pr is not None and nr is not None:
                try: pr = pr*math.exp(pr/rho)
                except: pr = pr
                Q1[s][a] = p1*Q1[s][a] + p2*pr    
                try: nr = nr*math.exp(-nr/rho)
                except: nr = nr
                Q2[s][a] = n1*Q2[s][a] - n2*nr
            else:
                if (r >= 0):
                    try: r = r*math.exp(r/rho)
                    except: r = r
                    Q1[s][a] = p1*Q1[s][a] + p2*r 
                if (r <= 0):
                    try: r = r*math.exp(-r/rho)
                    except: r = r
                    Q2[s][a] = n1*Q2[s][a] - n2*r
 
        if alg in ['TS']:
            if pr is not None and nr is not None: r = pr + nr    
            if (r >= 0): Q1[s][a] = Q1[s][a] + 1
            if (r <= 0): Q2[s][a] = Q2[s][a] + 1

        if alg in ['ETS']:
            if pr is not None and nr is not None:
                try: pr = pr*math.exp(pr/rho)
                except: pr = pr
                try: nr = nr*math.exp(-nr/rho)
                except: nr = nr
                r = pr + nr    
            if (r >= 0): Q1[s][a] = Q1[s][a] + 1
            if (r <= 0): Q2[s][a] = Q2[s][a] + 1
        
        if alg in ['EXP3']:
            mu = 0.05
            Qprime = fQprime(mu,player=player)
            est_r = r / Qprime[s][a]
            Q1[s][a] = Q1[s][a] * np.exp(mu*est_r/len(self.actionsPerState[s]))

        if alg in ['EXP30']:
            mu = 0
            Qprime = fQprime(mu,player=player)
            est_r = r / Qprime[s][a]
            Q1[s][a] = Q1[s][a] * np.exp(mu*est_r/len(self.actionsPerState[s]))
                
        if alg in ['eGreedy','UCB']:
            Q1[s][a] = Q1[s][a] + (r - Q1[s][a]) / N[s]

        if alg in ['HBTS','PTS','NTS','EHBTS','bADD','bADHD','bAD','bCP','bbvFTD','bPD','bM']:
            if pr is not None and nr is not None:
                Q1[s][a] = p1*Q1[s][a] + p2*pr
                Q2[s][a] = n1*Q2[s][a] - n2*nr 
            else:
                if (r >= 0): Q1[s][a] = p1*Q1[s][a] + p2*r 
                if (r <= 0): Q2[s][a] = n1*Q2[s][a] - n2*r 
                
        if alg in ['EXP4']:
            mu = 0.05
            Qprime = fQprime(mu,player=player)
            p = Qprime[s][a] / np.sum(list(Qprime[s].values()))
            est_r = {}
            for at in self.actionsPerState[s]:
                if at == a:
                    est_r[at] = r * Q1[s][at]/ (p * np.sum(Q1[s][at]))
                else:
                    est_r[at] = 0 * Q1[s][at] / (p * np.sum(Q1[s][at]))
            Q2[s][a] = Q2[s][a] + est_r[a]
            Zt = np.sum([np.exp(mu*Q2[s][at]) for at in self.actionsPerState[s]])
            # for st in self.stateSpace:
            for at in self.actionsPerState[s]:
                Q1[s][at] = np.exp(mu*est_r[at])/Zt

        if alg in ['LinUCB']:
            Q1[s][a] = Q1[s][a] + self.feats.flatten().dot(self.feats.flatten().T)
            Q2[s][a] = Q2[s][a] + r * self.feats.flatten().reshape((len(self.feats.flatten()),1))

        if alg in ['CTS']:
            Q1[s][a] = [Q1[s][a][0] + self.feats.flatten().dot(self.feats.flatten().T), Q1[s][a][1] + self.feats.flatten().reshape((len(self.feats.flatten()),1))*r]

        if alg in ['SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
            if pr is not None and nr is not None:
                Q1[s][a] = [p1*Q1[s][a][0] + self.feats.flatten().dot(self.feats.flatten().T), p1*Q1[s][a][1] + p2*self.feats.flatten().reshape((len(self.feats.flatten()),1))*pr]
                Q2[s][a] = [n1*Q2[s][a][0] + self.feats.flatten().dot(self.feats.flatten().T), n1*Q2[s][a][1] + n2*self.feats.flatten().reshape((len(self.feats.flatten()),1))*nr]
            else:
                if (r >= 0): 
                    Q1[s][a] = [p1*Q1[s][a][0] + self.feats.flatten().dot(self.feats.flatten().T), p1*Q1[s][a][1] + p2*self.feats.flatten().reshape((len(self.feats.flatten()),1))*r]
                if (r <= 0):        
                    Q2[s][a] = [n1*Q2[s][a][0] + self.feats.flatten().dot(self.feats.flatten().T), n1*Q2[s][a][1] + n2*self.feats.flatten().reshape((len(self.feats.flatten()),1))*r]

        if noalg:
            self.Q1 = Q1
            self.Q2 = Q2
        else:
            self.Q1s[player] = Q1
            self.Q2s[player] = Q2
            
        return nxt_a


class MAB(MDP):
    """
    MAB game setting
    """
    def __init__(self,algorithm,reward_functions,nTrials,T,Q1=None,Q2=None,Q1s=None,Q2s=None,Traj=None):
        MDP.__init__(self,algorithm,reward_functions,nTrials,T,Q1=Q1,Q2=Q2,Q1s=Q1s,Q2s=Q2s,Traj=Traj)
                
        # map actions to states
        self.actionsPerState = {}
        self.actionsPerState[self.STATE_A] = [self.ACTION_LEFT, self.ACTION_RIGHT]
        self.actionsPerState[self.STATE_B] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_C] = [self.ACTION_DUMMY]
        self.stateSpace = [self.STATE_A,self.STATE_B,self.STATE_C]

    # move from state s using action a
    def move(self,s,a,t=None):
        if self.BC and self.trajs is not None:
            if(s==self.STATE_A):
                a_cor = self.trajs[0,self.T-t-1]
                if(a == a_cor): return 1, a_cor
                else: return 0, a_cor
            else:
                return 0, s
        else:
            if(s==self.STATE_A):
                if(a == self.ACTION_LEFT): return self.reward_from_B(), self.STATE_B
                elif(a == self.ACTION_RIGHT): return self.reward_from_C(), self.STATE_C
            else:
                return 0, s

    # return true if this is a terminal state
    def isTerminal(self,s):
        return s == self.STATE_B or s == self.STATE_C


class IGT(MDP):
    """
    IGT game setting
    """
    def __init__(self,algorithm,reward_functions,nTrials,T,Q1=None,Q2=None,Q1s=None,Q2s=None,Traj=None):
        MDP.__init__(self,algorithm,reward_functions,nTrials,T,Q1=Q1,Q2=Q2,Q1s=Q1s,Q2s=Q2s,Traj=Traj)
        
        self.reward_from_A = reward_functions[0]
        self.reward_from_B = reward_functions[1]
        self.reward_from_C = reward_functions[2]
        self.reward_from_D = reward_functions[3]
    
        # In IGT, the initial state is self.STATE_E
        self.ACTION_A,self.ACTION_B,self.ACTION_C,self.ACTION_D = 0,1,2,3
        # map actions to states
        self.actionsPerState = {}
        self.actionsPerState[self.STATE_E] = [self.ACTION_A,self.ACTION_B,self.ACTION_C,self.ACTION_D]
        self.actionsPerState[self.STATE_A] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_B] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_C] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_D] = [self.ACTION_DUMMY]
        
        self.nArms = 4
        self.initialState = self.STATE_E   

    def move(self,s,a,N,t=None): 
        if self.BC and self.trajs is not None:
            if(s==self.STATE_E):
                a_cor = self.trajs[0,self.T-t-1]
                if(a == a_cor): return 1, a_cor
                else: return 0, a_cor
            else:
                return 1,s
        else:
            if(s==self.STATE_E):
                if(a == self.ACTION_A):  return self.reward_from_A(N[self.STATE_A]), self.STATE_A
                elif(a == self.ACTION_B): return self.reward_from_B(N[self.STATE_B]), self.STATE_B
                elif(a == self.ACTION_C): return self.reward_from_C(N[self.STATE_C]), self.STATE_C
                elif(a == self.ACTION_D): return self.reward_from_D(N[self.STATE_D]), self.STATE_D
        return [0,0],s
            
    def isTerminal(self,s):
        return s == self.STATE_A or s == self.STATE_B or s == self.STATE_C or s == self.STATE_D

    def experiment(self):   
        episodes = {}
        t = 0
        self.reset()
        ILeft = 0 #contains the number of times left action is chosen at initial state I
        N={}    # contains the number of visits for each state
        for s in self.stateSpace: N[s] = 0
        NSA = {}        # contains the number of visits for each state and action
        for s in self.stateSpace: 
            NSA[s] = {}
            for a in self.actionsPerState[s]:
                NSA[s][a] = 0
        reward,pos_reward,neg_reward,actions = None,None,None,None
        
        last_a = None
        
        # loop for T episodes
        for i in range(self.T):

            s = self.initialState
            
            if i == 0: a = self.selectInitialAction(self.initialState,True,last_a,N,NSA,t=i)
            else: a = self.selectInitialAction(self.initialState,False,last_a,N,NSA,t=i)

            gameover = False

            #loop until game is over, this will be ONE episode
            while not gameover:
                actions = a
                # record learning steps
                t += 1
                
                if self.algorithm in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M']:
                    a = self.randomAction(s, a, self.epsilon) # apply epsilon greedy selection (including for action chosen at STATE A)
                if self.algorithm == 'eGreedy' and N[s] > len(self.actionsPerState[s]):
                    a = self.randomAction(s, a, self.epsilon) # apply epsilon greedy selection (including for action chosen at STATE A)

                #update the number of visits for state s
                N[s] += 1

                # if left action is chosen at state A, increment the counter
                if (s == self.STATE_E and a == self.ACTION_A) or (s == self.STATE_E and a == self.ACTION_B):
                    ILeft += 1

                #move to the next state and get the reward
                [pr,nr], nxt_s = self.move(s,a,N,i)
                r = pr+nr
                reward,pos_reward,neg_reward = r,pr,nr

                #update the number of visits per state and action
                if not s in NSA: NSA[s] = {}
                NSA[s][a] += 1

                #compute alpha
                alpha = 1 / np.power(NSA[s][a], .8)

                #update the Q values and get the best action for the next state
                nxt_a = self.updateAgent(s, a, r, nxt_s, alpha, t, pr, nr, N, NSA)

                #if next state is terminal then mark as gameover (end of episode)
                gameover = self.isTerminal(nxt_s)

                if gameover: N[nxt_s] += 1
                last_a = a
                s = nxt_s
                a = nxt_a

            #update stats for each episode
            if not (i in episodes):
                episodes[i] = {}
                episodes[i]["count"] = 0
                episodes[i]["Q1(I)a"] = episodes[i]["Q2(I)a"] = episodes[i]["Q1(I)b"] = episodes[i]["Q2(I)b"] = 0
                episodes[i]["Q1(I)c"] = episodes[i]["Q2(I)c"] = episodes[i]["Q1(I)d"] = episodes[i]["Q2(I)d"] = 0
            episodes[i]["count"],episodes[i]["percent"] = ILeft, ILeft / (i+1)
            episodes[i]["reward"],episodes[i]["pos_reward"],episodes[i]["neg_reward"] = reward,pos_reward,neg_reward
            episodes[i]["actions"] = actions
#             episodes[i]["cumreward"] = sum(reward)
            episodes[i]["Q1(I)a"] = ((episodes[i]["Q1(I)a"] * i) + self.Q1[self.STATE_E][self.ACTION_A])/(i+1)
            episodes[i]["Q2(I)a"] = ((episodes[i]["Q2(I)a"] * i) + self.Q2[self.STATE_E][self.ACTION_A])/(i+1)
            episodes[i]["Q1(I)b"] = ((episodes[i]["Q1(I)b"] * i) + self.Q1[self.STATE_E][self.ACTION_B])/(i+1)
            episodes[i]["Q2(I)b"] = ((episodes[i]["Q2(I)b"] * i) + self.Q2[self.STATE_E][self.ACTION_B])/(i+1)
            episodes[i]["Q1(I)c"] = ((episodes[i]["Q1(I)c"] * i) + self.Q1[self.STATE_E][self.ACTION_C])/(i+1)
            episodes[i]["Q2(I)c"] = ((episodes[i]["Q2(I)c"] * i) + self.Q2[self.STATE_E][self.ACTION_C])/(i+1)
            episodes[i]["Q1(I)d"] = ((episodes[i]["Q1(I)d"] * i) + self.Q1[self.STATE_E][self.ACTION_D])/(i+1)
            episodes[i]["Q2(I)d"] = ((episodes[i]["Q2(I)d"] * i) + self.Q2[self.STATE_E][self.ACTION_D])/(i+1)
        
        return episodes 
   
    def run(self):
        report = {}
        count = percent = np.ndarray((self.nTrials,self.T))
        Q1Ia = np.ndarray((self.nTrials,self.T))
        Q2Ia = np.ndarray((self.nTrials,self.T))
        Q1Ib = np.ndarray((self.nTrials,self.T))
        Q2Ib = np.ndarray((self.nTrials,self.T))
        Q1Ic = np.ndarray((self.nTrials,self.T))
        Q2Ic = np.ndarray((self.nTrials,self.T))
        Q1Id = np.ndarray((self.nTrials,self.T))
        Q2Id = np.ndarray((self.nTrials,self.T))
        cumreward = reward = pos_reward = neg_reward = actions = np.ndarray((self.nTrials,self.T))
    
        #run batch of experiments
        for k in range(self.nTrials):
            tmp = self.experiment()
            #aggregate every experiment result into the final report
            for i in range(self.T):
                count[k,i] = tmp[i]["count"]
                percent[k,i] = 100*tmp[i]["count"] / (i+1)
                Q1Id[k,i] = tmp[i]["Q1(I)a"]
                Q2Ia[k,i] = tmp[i]["Q2(I)a"]
                Q1Ib[k,i] = tmp[i]["Q1(I)b"]
                Q2Ib[k,i] = tmp[i]["Q2(I)b"]
                Q1Ic[k,i] = tmp[i]["Q1(I)c"]
                Q2Ic[k,i] = tmp[i]["Q2(I)c"]
                Q1Id[k,i] = tmp[i]["Q1(I)d"]
                Q2Id[k,i] = tmp[i]["Q2(I)d"]
#                 cumreward[k,i] = tmp[i]["cumreward"]
                reward[k,i] = tmp[i]["reward"]
                pos_reward[k,i] = tmp[i]["pos_reward"]
                neg_reward[k,i] = tmp[i]["neg_reward"]
                actions[k,i] = tmp[i]["actions"]

        report["count"],report["percent"] = count,percent
        report["Q1(I)a"],report["Q2(I)a"],report["Q1(I)b"],report["Q2(I)b"] = Q1Ia,Q2Ia,Q1Ib,Q2Ib
        report["Q1(I)c"],report["Q2(I)c"],report["Q1(I)d"],report["Q2(I)d"] = Q1Ic,Q2Ic,Q1Id,Q2Id
        report["cumreward"],report["reward"],report["pos_reward"],report["neg_reward"] = cumreward,reward,pos_reward,neg_reward
        report["actions"] = actions
        return report


class IPD(MDP):
    """
    IPD game setting
    """
    def __init__(self,algorithms,reward_functions,nTrials,T,nMemory,Q1=None,Q2=None,Q1s=None,Q2s=None,Traj=None):
        MDP.__init__(self,None,reward_functions,nTrials,T,Q1=Q1,Q2=Q2,Q1s=Q1s,Q2s=Q2s,Traj=Traj)
        
        self.nArms = 2
        self.initialState = self.STATE_E  
        self.algs = algorithms
        self.nM = nMemory
        self.nP = len(self.algs)
        self.moves = []
        self.fQprimes = []

        # In IPD, the context can be the history of length nMemory
        self.feats = np.zeros((self.nP,nMemory))

        # In IPD: T > R > P > S amd 2R > S+T
        self.reward_from_A = reward_functions[0] # R for reward
        self.reward_from_B = reward_functions[1] # S for sucker
        self.reward_from_C = reward_functions[2] # T for temptation
        self.reward_from_D = reward_functions[3] # P for penalty
    
        # In IPD (nPlayer == 2), the initial state is self.STATE_E
        # STATE_A: (R,R)
        # STATE_B: (S,T)
        # STATE_C: (T,S)
        # STATE_D: (P,P)

        # In mult-agent IPD (nPlayer >= 3), the initial state is self.STATE_E
        # STATE_A: (R,R,R,R,R..)
        # STATE_B: (S,T,S,T,T...)
        # STATE_D: (P,P,P,P,P...)

        # In IPD, each player can have two choices
        # ACTION_A: Cooperate
        # ACTION_B: Defect
        self.ACTION_A,self.ACTION_B = 0,1

        # map actions to states
        self.actionsPerState = {}
        self.actionsPerState[self.STATE_E] = [self.ACTION_A,self.ACTION_B]
        self.actionsPerState[self.STATE_A] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_B] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_C] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_D] = [self.ACTION_DUMMY]

        # save temporal reward for behavorial cloning
        self.temp_rs = None
    
    def isTerminal(self,s):
        return s == self.STATE_A or s == self.STATE_B or s == self.STATE_C or s == self.STATE_D

    def reset(self):
        self.fQprimes = []
        for i, alg in enumerate(self.algs):
            self.fQprimes.append(self.resetQprimeFunction(alg))
            if alg in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M','CTS','SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
                defaultQ = 0
            else:
                defaultQ = 1
            Q1,Q2 = {},{}
            for s in self.stateSpace:
                Q1[s],Q2[s] = {},{}
                for a in self.actionsPerState[s]:
                    if alg in ['LinUCB']:
                        Q1[s][a] = np.eye(len(self.feats.flatten()))
                        Q2[s][a] = np.zeros((len(self.feats.flatten()),1))
                    elif alg in ['EXP4']:
                        Q1[s][a] = np.ones((len(self.feats.flatten()),1))/len(self.feats.flatten())
                        Q2[s][a] = np.zeros((len(self.feats.flatten()),1))
                    elif alg in ['CTS','SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
                        Q1[s][a] = [np.eye(len(self.feats.flatten())),np.zeros((len(self.feats.flatten()),1))]
                        Q2[s][a] = [np.eye(len(self.feats.flatten())),np.zeros((len(self.feats.flatten()),1))]
                    else:
                        Q1[s][a] = defaultQ
                        Q2[s][a] = defaultQ
            self.Q1s.append(Q1)
            self.Q2s.append(Q2)

    def storeMemory(self,tnow):
        if self.BC and self.trajs is not None:
            for t in np.arange(self.nM-1): self.feats[:,t+1] = self.feats[:,t]
            self.feats[:,0] = self.trajs[:,self.T-tnow-1]
        else:
            for t in np.arange(self.nM-1): self.feats[:,t+1] = self.feats[:,t]
            self.feats[:,0] = self.moves

    def action2code(self,a):
        if a == self.ACTION_A:
            return 1
        elif a == self.ACTION_B:
            return -1
        else:
            print('wrong action codes in action2code:', a)
            return None

    def code2action(self,c):
        if c == 1:
            return self.ACTION_A
        elif c == -1:
            return self.ACTION_B
        else:
            print('wrong action codes in code2action:', c)
            return None

    def move2state(self,a,t):
        m = self.action2code(a)
        self.moves.append(m)
        if len(self.moves) < self.nP:
            return None
        else:
            if self.nP == 2:
                if int(self.moves[0]) == 1 and int(self.moves[1]) == 1: s = self.STATE_A
                elif int(self.moves[0]) == 1 and int(self.moves[1]) == -1: s = self.STATE_B
                elif int(self.moves[0]) == -1 and int(self.moves[1]) == 1: s = self.STATE_C
                elif int(self.moves[0]) == -1 and int(self.moves[1]) == -1: s = self.STATE_D
                else: 
                    s = None
                    print('wrong state codes: ', s, a, self.moves, int(self.moves[0]) == 1, int(self.moves[1]) == 1)
            else:
                if np.sum(self.moves) == len(self.moves): s = self.STATE_A
                elif np.sum(self.moves) == -len(self.moves): s = self.STATE_D
                elif 1 in self.moves and -1 in self.moves: s = self.STATE_B
                else: 
                    s = None
                    print('wrong state codes: ', s, a, self.moves, int(self.moves[0]) == 1, int(self.moves[1]) == 1)
            if s is not None: self.storeMemory(t)
            mvs = np.array(self.moves)
            self.moves = []
            return s,mvs

    def moveSingle(self,s,a,N,t):
        rs = np.ones((self.nP,2))
        if self.BC and self.trajs is not None:
            if s==self.STATE_E:
                a_cor = self.trajs[0,self.T-t-1]
                ac = self.ACTION_A if a_cor == 1 else self.ACTION_B
                sobj = self.move2state(ac,t)
                if sobj is not None:
                    s,mvs = sobj
                    rs = self.temp_rs
                else:
                    rs[0,:] = [1,0] if ac == a else [0,0]
                    self.temp_rs = rs
                return rs,s
        elif s==self.STATE_E:
            sobj = self.move2state(a,t)
            if sobj is not None:
                s,mvs = sobj
                if s == self.STATE_A:
                    rs[:,0] = self.reward_from_A(N)[0]
                    rs[:,1] = self.reward_from_A(N)[0]
                elif s in [self.STATE_B,self.STATE_C]:
                    rs[mvs == 1,0] = self.reward_from_B(N)[0]
                    rs[mvs == 1,1] = self.reward_from_B(N)[1]
                    rs[mvs == -1,0] = self.reward_from_C(N)[0]
                    rs[mvs == -1,1] = self.reward_from_C(N)[1]
                elif s == self.STATE_D:
                    rs[:,0] = self.reward_from_D(N)[0]
                    rs[:,1] = self.reward_from_D(N)[0]
                return rs,s
        return 0*rs,s

    def move(self,s,a,N,t=None):
        r = [[0,0],[0,0]]
        for at in a:
            r, nxt_s = self.moveSingle(s,at,N,t)
        return r, nxt_s
        
    def selectInitialActionBatch(self,startState,veryFirst=False,last_a=None,Ns=None,NSAs=None,t=None):
        a = []
        for i in np.arange(self.nP):
            a.append(self.selectInitialAction(startState,veryFirst,last_a[i],Ns[i],NSAs[i],self.algs[i],i,self.fQprimes[i],t=t))
        return a

    def randomActionBatch(self,s,a,Ns,eps=.1):
        actual_a = []
        for i,alg in enumerate(self.algs):
            if alg in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M']:
                actual_a.append(self.randomAction(s,a[i],eps))
            elif alg == 'eGreedy' and Ns[i][s] > len(self.actionsPerState[s]):
                actual_a.append(self.randomAction(s,a[i],eps))
            else:
                actual_a.append(a[i])
        return actual_a

    def updateAgentBatch(self, s, a, r, nxt_s, alpha, t, pr=None, nr=None, Ns=None, NSAs=None):
        nxt_a = []
        for i,alg in enumerate(self.algs):
            nxt_a.append(self.updateAgent(s,a[i],r[i],nxt_s,alpha[i],t,pr[i],nr[i],Ns[i],NSAs[i],alg,i,self.fQprimes[i]))
        return nxt_a

    def experiment(self):   
        episodes = {}
        t = 0
        self.reset()

        IC = [0] * self.nP #contains the number of times Cooperate action is chosen at initial state I
        
        Ns,NSAs = [],[]
        reward,pos_reward,neg_reward,actions,last_a = [],[],[],[],[]
        for p in np.arange(self.nP):
            N={}    # contains the number of visits for each state
            for s in self.stateSpace: N[s] = 0
            NSA = {}        # contains the number of visits for each state and action
            for s in self.stateSpace: 
                NSA[s] = {}
                for a in self.actionsPerState[s]: NSA[s][a] = 0
            Ns.append(N)
            NSAs.append(NSA)
            reward.append(None)
            pos_reward.append(None)
            neg_reward.append(None)
            actions.append(None)
            last_a.append(None)
        
        # loop for T episodes
        for i in range(self.T):

            s = self.initialState
            
            if i == 0: 
                a = self.selectInitialActionBatch(self.initialState,True,last_a,Ns,NSAs,t=i)
            else: 
                a = self.selectInitialActionBatch(self.initialState,False,last_a,Ns,NSAs,t=i)

            gameover = False

            #loop until game is over, this will be ONE episode
            while not gameover:
                actions = a
                # record learning steps
                t += 1
                                
                a = self.randomActionBatch(s, a, Ns, self.epsilon) # apply epsilon greedy selection (including for action chosen at STATE A)
                
                #update the number of visits for state s and the number of visits per state and action
                alpha = []
                for p in np.arange(self.nP):
                    Ns[p][s] += 1
                    if not s in NSAs[p]: NSAs[p][s] = {}
                    NSAs[p][s][a[p]] += 1
                    alpha_t = 1 / np.power(NSAs[p][s][a[p]], .8)
                    alpha.append(alpha_t)

                #move to the next state and get the reward
                fullr, nxt_s = self.move(s,a,N,i)
                for j,fr in enumerate(fullr):
                    pr,nr = fr
                    r = pr+nr
                    reward[j],pos_reward[j],neg_reward[j] = r,pr,nr

                # if coop action is chosen at state E, increment the counter
                for p in np.arange(self.nP):
                    if (s == self.STATE_E and a[p] == self.ACTION_A):
                        IC[p] += 1

                #update the Q values and get the best action for the next state
                nxt_a = self.updateAgentBatch(s, a, reward, nxt_s, alpha, t, pos_reward, neg_reward, Ns, NSAs)

                #if next state is terminal then mark as gameover (end of episode)
                gameover = self.isTerminal(nxt_s)

                if gameover: 
                    for p in np.arange(self.nP):
                        Ns[p][nxt_s] += 1
                last_a = a
                s = nxt_s
                a = nxt_a

            #update stats for each episode
            #for IPD, Qx(I)a and Qx(I)b correspond to action a and b for player 0 and
            #  Qx(I)c and Qx(I)d correspond to action a and b for player 1
            if not (i in episodes):
                episodes[i] = {}
                for p in np.arange(self.nP):
                    episodes[i]["count"+str(p)] = 0
                # episodes[i]["count1"] = 0
                # episodes[i]["Q1(I)a"] = episodes[i]["Q2(I)a"] = episodes[i]["Q1(I)b"] = episodes[i]["Q2(I)b"] = 0
                # episodes[i]["Q1(I)c"] = episodes[i]["Q2(I)c"] = episodes[i]["Q1(I)d"] = episodes[i]["Q2(I)d"] = 0
            for p in np.arange(self.nP):
                episodes[i]["count"+str(p)],episodes[i]["percent"+str(p)] = IC[p], IC[p] / (i+1)
                episodes[i]["reward"+str(p)],episodes[i]["pos_reward"+str(p)],episodes[i]["neg_reward"+str(p)] = reward[p],pos_reward[p],neg_reward[p]
                episodes[i]["actions"+str(p)] = actions[p]

            # episodes[i]["count0"],episodes[i]["percent0"] = IC[0], IC[0] / (i+1)
            # episodes[i]["count1"],episodes[i]["percent1"] = IC[1], IC[1] / (i+1)
            # episodes[i]["reward0"],episodes[i]["pos_reward0"],episodes[i]["neg_reward0"] = reward[0],pos_reward[0],neg_reward[0]
            # episodes[i]["reward1"],episodes[i]["pos_reward1"],episodes[i]["neg_reward1"] = reward[1],pos_reward[1],neg_reward[1]
            # episodes[i]["actions0"] = actions[0]
            # episodes[i]["actions1"] = actions[1]

#             episodes[i]["cumreward"] = sum(reward)
            # episodes[i]["Q1(I)a"] = ((episodes[i]["Q1(I)a"] * i) + self.Q1s[0][self.STATE_E][self.ACTION_A])/(i+1)
            # episodes[i]["Q2(I)a"] = ((episodes[i]["Q2(I)a"] * i) + self.Q2s[0][self.STATE_E][self.ACTION_A])/(i+1)
            # episodes[i]["Q1(I)b"] = ((episodes[i]["Q1(I)b"] * i) + self.Q1s[0][self.STATE_E][self.ACTION_B])/(i+1)
            # episodes[i]["Q2(I)b"] = ((episodes[i]["Q2(I)b"] * i) + self.Q2s[0][self.STATE_E][self.ACTION_B])/(i+1)
            # episodes[i]["Q1(I)c"] = ((episodes[i]["Q1(I)c"] * i) + self.Q1s[1][self.STATE_E][self.ACTION_A])/(i+1)
            # episodes[i]["Q2(I)c"] = ((episodes[i]["Q2(I)c"] * i) + self.Q2s[1][self.STATE_E][self.ACTION_A])/(i+1)
            # episodes[i]["Q1(I)d"] = ((episodes[i]["Q1(I)d"] * i) + self.Q1s[1][self.STATE_E][self.ACTION_B])/(i+1)
            # episodes[i]["Q2(I)d"] = ((episodes[i]["Q2(I)d"] * i) + self.Q2s[1][self.STATE_E][self.ACTION_B])/(i+1)
        
        return episodes 
        
    def run(self):
        report = {}

        # count0 = np.ndarray((self.nTrials,self.T))
        # percent0 = np.ndarray((self.nTrials,self.T))
        # count1 = np.ndarray((self.nTrials,self.T))
        # percent1 = np.ndarray((self.nTrials,self.T))
        # # Q1Ia = np.ndarray((self.nTrials,self.T))
        # # Q2Ia = np.ndarray((self.nTrials,self.T))
        # # Q1Ib = np.ndarray((self.nTrials,self.T))
        # # Q2Ib = np.ndarray((self.nTrials,self.T))
        # # Q1Ic = np.ndarray((self.nTrials,self.T))
        # # Q2Ic = np.ndarray((self.nTrials,self.T))
        # # Q1Id = np.ndarray((self.nTrials,self.T))
        # # Q2Id = np.ndarray((self.nTrials,self.T))
        # cumreward0 = np.ndarray((self.nTrials,self.T))
        # reward0 = np.ndarray((self.nTrials,self.T))
        # pos_reward0 = np.ndarray((self.nTrials,self.T))
        # neg_reward0 = np.ndarray((self.nTrials,self.T))
        # actions0 = np.ndarray((self.nTrials,self.T))

        # cumreward1 = np.ndarray((self.nTrials,self.T))
        # reward1 = np.ndarray((self.nTrials,self.T))
        # pos_reward1 = np.ndarray((self.nTrials,self.T))
        # neg_reward1 = np.ndarray((self.nTrials,self.T))
        # actions1 = np.ndarray((self.nTrials,self.T))
    
#         #run batch of experiments
#         for k in range(self.nTrials):
#             tmp = self.experiment()
#             #aggregate every experiment result into the final report
#             for i in range(self.T):
#                 count0[k,i] = tmp[i]["count0"]
#                 percent0[k,i] = 100*tmp[i]["count0"] / (i+1)
#                 count1[k,i] = tmp[i]["count1"]
#                 percent1[k,i] = 100*tmp[i]["count1"] / (i+1)
#                 # Q1Id[k,i] = tmp[i]["Q1(I)a"]
#                 # Q2Ia[k,i] = tmp[i]["Q2(I)a"]
#                 # Q1Ib[k,i] = tmp[i]["Q1(I)b"]
#                 # Q2Ib[k,i] = tmp[i]["Q2(I)b"]
#                 # Q1Ic[k,i] = tmp[i]["Q1(I)c"]
#                 # Q2Ic[k,i] = tmp[i]["Q2(I)c"]
#                 # Q1Id[k,i] = tmp[i]["Q1(I)d"]
#                 # Q2Id[k,i] = tmp[i]["Q2(I)d"]
# #                 cumreward[k,i] = tmp[i]["cumreward"]
#                 reward0[k,i] = tmp[i]["reward0"]
#                 pos_reward0[k,i] = tmp[i]["pos_reward0"]
#                 neg_reward0[k,i] = tmp[i]["neg_reward0"]
#                 actions0[k,i] = tmp[i]["actions0"]
#                 reward1[k,i] = tmp[i]["reward1"]
#                 pos_reward1[k,i] = tmp[i]["pos_reward1"]
#                 neg_reward1[k,i] = tmp[i]["neg_reward1"]
#                 actions1[k,i] = tmp[i]["actions1"]

        for p in range(self.nP):
            count = np.ndarray((self.nTrials,self.T))
            percent = np.ndarray((self.nTrials,self.T))
            cumreward = np.ndarray((self.nTrials,self.T))
            reward = np.ndarray((self.nTrials,self.T))
            pos_reward = np.ndarray((self.nTrials,self.T))
            neg_reward = np.ndarray((self.nTrials,self.T))
            actions = np.ndarray((self.nTrials,self.T))
            for k in range(self.nTrials):
                tmp = self.experiment()
                for i in range(self.T):
                    count[k,i] = tmp[i]["count"+str(p)]
                    percent[k,i] = 100*tmp[i]["count"+str(p)] / (i+1)
                    reward[k,i] = tmp[i]["reward"+str(p)]
                    pos_reward[k,i] = tmp[i]["pos_reward"+str(p)]
                    neg_reward[k,i] = tmp[i]["neg_reward"+str(p)]
                    actions[k,i] = tmp[i]["actions"+str(p)]
            report["count"+str(p)],report["percent"+str(p)] = count,percent
            report["cumreward"+str(p)],report["reward"+str(p)],report["pos_reward"+str(p)],report["neg_reward"+str(p)] = cumreward,reward,pos_reward,neg_reward
            report["actions"+str(p)] = actions
       
        # report["count0"],report["percent0"] = count0,percent0
        # report["count1"],report["percent1"] = count1,percent1
        # # report["Q1(I)a"],report["Q2(I)a"],report["Q1(I)b"],report["Q2(I)b"] = Q1Ia,Q2Ia,Q1Ib,Q2Ib
        # # report["Q1(I)c"],report["Q2(I)c"],report["Q1(I)d"],report["Q2(I)d"] = Q1Ic,Q2Ic,Q1Id,Q2Id
        # report["cumreward0"],report["reward0"],report["pos_reward0"],report["neg_reward0"] = cumreward0,reward0,pos_reward0,neg_reward0
        # report["actions0"] = actions0
        # report["cumreward1"],report["reward1"],report["pos_reward1"],report["neg_reward1"] = cumreward1,reward1,pos_reward1,neg_reward1
        # report["actions1"] = actions1
        return report

