#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import random


# In[ ]:


# Directions
#0 is left
#1 is down
#2 is right
#3 is Up

# 8.8 displays your position


# In[126]:




class myGridWorld:
    
    size=5
    RewardGrid=np.zeros([5,5])
    RewardGrid[0][4]=1
    PositionGrid=np.zeros([5,5])
    PositionGrid[4][0]=1.1
    action_space=4
    noisyMoveChance=0.3
    currI=4
    currJ=0
    DoneStatus=False
    EnableNoise=True
    observation_spaces=size*size
    
    def __init__(self,size=5,noisyMoveChance=0.3,EnableNoise=True):
        self.basicReset()
        self.EnableNoise=EnableNoise
        if(0<size):
            self.size=int(size)
            self.RewardGrid=np.zeros([size,size])
            self.RewardGrid[0][size-1]=1
            self.PositionGrid=np.zeros([size,size])
            self.PositionGrid[size-1][0]=1.1
            self.observation_spaces=self.size*self.size
            self.currI=size-1
            self.currJ=0
            self.observation_spaces=self.size*self.size
        if(0<noisyMoveChance and noisyMoveChance<1):
            self.noisyMoveChance=noisyMoveChance
            
    def basicReset(self):
        self.size=5
        self.RewardGrid=np.zeros([5,5])
        self.RewardGrid[0][4]=1
        self.PositionGrid=np.zeros([5,5])
        self.PositionGrid[4][0]=1.1
        self.action_space=4
        self.noisyMoveChance=0.3
        self.currI=4
        self.currJ=0
        self.DoneStatus=False
        self.EnableNoise=True
        self.observation_spaces=self.size*self.size
            
    def reset(self,size=5,noisyMoveChance=0.3,EnableNoise=True):
        self.__init__(size,noisyMoveChance,EnableNoise)
        return self.currI*self.size+self.currJ
    
    def printRewardGrid(self):
        for i in range(len(self.RewardGrid)):
            for j in range(len(self.RewardGrid[0])):
                print(self.RewardGrid[i][j],end=' ')
            print()
            
    def printPositionGrid(self):
        for i in range(len(self.PositionGrid)):
            for j in range(len(self.PositionGrid[0])):
                print(self.PositionGrid[i][j],end=' ')
            print()
            
    def getPositionGrid(self):
        return self.PositionGrid
            
    def render(self):
        self.printPositionGrid()
        
    def getAvailableMoves(self):
        return self.action_space
    
    def getSize(self):
        return self.size
            
    def move(self,action):
        randNum=random.random()
        if(self.EnableNoise and randNum<=self.noisyMoveChance):
            self.makeNoisyMove(action)
        else:
            self.makeProperMove(action)
        return self.currI,self.currJ,self.currI*self.size+self.currJ,self.RewardGrid[self.currI][self.currJ],self.DoneStatus
        
    def makeNoisyMove(self,action):
        randNum=random.randint(0,3)
        self.makeProperMove(randNum)
        
    def makeProperMove(self,action):
        if(action==0):#Left
            if(0<self.currJ):
                self.PositionGrid[self.currI][self.currJ]=0
                self.currJ-=1
                self.PositionGrid[self.currI][self.currJ]=1.1
        elif(action==1):#1 is down
            if(self.currI<self.size-1):
                self.PositionGrid[self.currI][self.currJ]=0
                self.currI+=1
                self.PositionGrid[self.currI][self.currJ]=1.1
        elif(action==2):#2 is right
            if(self.currJ<self.size-1):
                self.PositionGrid[self.currI][self.currJ]=0
                self.currJ+=1
                self.PositionGrid[self.currI][self.currJ]=1.1
        elif(action==3):#3 is Up
            if(0<self.currI):
                self.PositionGrid[self.currI][self.currJ]=0
                self.currI-=1
                self.PositionGrid[self.currI][self.currJ]=1.1
                
        if(self.currI==0 and self.currJ==self.size-1):
            self.DoneStatus=True
            
    def step(self,action):
        return self.move(action)


# In[182]:


class myGridWorldTrainer:
    
    Q=[]
    matrix=[]
    Trajectories=[]
    DirectionalMatrix=[]
    
    def trainModel(self,model):
        env=model
        alpha = 0.6
        gamma = 0.9
        Q = np.zeros([env.observation_spaces, env.action_space])
        for episode in range(1,1001):
            done = False
            TotalReward = 0
            state = env.reset()
            while done != True:
                    if(episode<500):
                        action = random.randint(0,3)
                    else:
                        action=np.argmax(Q[state])
                    i,j,state2, reward, done = env.step(action)
                    Q[state,action] += alpha * (reward + gamma* np.max(Q[state2]) - Q[state,action])
                    TotalReward += reward
                    state = state2
        self.Q=Q
        return Q
    
    def getDirections(self,Q):
        matrix=[]
        for i in range(0,25):
            matrix.append(np.argmax(Q[i]))
        matrix=np.reshape(matrix,(5,5))
        DirectionalMatrix=[]
        for i in range(5):
            row=[]
            for j in range(5):
                if(matrix[i][j]==0):
                    row.append('\u2190')
                elif(matrix[i][j]==1):
                    row.append('\u2193')
                elif(matrix[i][j]==2):
                    row.append('\u2192')
                elif(matrix[i][j]==3):
                    row.append('\u2191')
            DirectionalMatrix.append(row)
#         for row in DirectionalMatrix:
#             print(row)
        self.DirectionalMatrix=DirectionalMatrix
        self.matrix=matrix
        return matrix
            
    def getTrajectories(self,matrix,numTrajectories):
        Trajectories=[]

        for iters in range(numTrajectories):
            path=[]
            done=False
            state = env.reset()
            TotalReward = 0
            path.append(state)
            i=int(state/env.size)
            j=state%env.size
            while done != True:
                action=matrix[i][j]
                i,j,state2, reward, done = env.step(action)
                TotalReward += reward
                state = state2
                path.append(state)

            Trajectories.append(path)
#         for i in Trajectories:
#             print(i)
        self.Trajectories=Trajectories
        return Trajectories

    def allInOne(self,model,numTrajectories):
        Q=self.trainModel(model)
        matrix=self.getDirections(Q)
        return self.getTrajectories(matrix,numTrajectories)


# In[184]:


sampleGrid=myGridWorld()
sampleGridTrainer=myGridWorldTrainer()
sampleTrajectories=sampleGridTrainer.allInOne(sampleGrid,20)
for i in sampleTrajectories:
    print(i)


# In[ ]:


#=========================================================================================================================

#Testing data below


# In[181]:


# for i in sampleGridTrainer.DirectionalMatrix:
#     print (i)


# In[153]:


#print(matrix)


# In[154]:


# for row in DirectionalMatrix:
#     print(row)


# In[156]:


# import pickle
# mydata = [Q,matrix,DirectionalMatrix]
# outputFile = 'model.data'
# fw = open(outputFile, 'wb')
# pickle.dump(mydata, fw)
# fw.close()


# In[159]:


# import pickle
# inputFile = 'model.data'
# fd = open(inputFile, 'rb')
# dataset = pickle.load(fd)
# print (dataset)


# In[ ]:


# \u2190 ←
# \u2191 ↑
# \u2192 →
# \u2193 ↓


# In[ ]:


#0 is left
#1 is down
#2 is right
#3 is Up


# In[168]:





# In[169]:


# for i in Trajectories:
#     print(i)


# In[ ]:




