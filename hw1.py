
# coding: utf-8

# In[1]:

random

kmeans
    euclidean:
        k=1 0.82
        k=100 
        k=1000 0.957
        k=5000 0.9669
        k=10000 0.9693
    twosided tangent:
        k=10 0.8439
        k=100 0.9667
    onesided tangent: 
        k=100 0.9426


# In[1]:

import numpy as np
from PIL import Image
from numpy import linalg as LA
from sklearn.datasets import fetch_mldata
from kmeans import kmeans


# In[2]:

mnist = fetch_mldata('MNIST original', data_home='./data')


# In[3]:

def imageshow(img, name='', showflag=True):
    im = Image.fromarray(np.reshape(img,(28,28)).astype(np.uint8))
    if showflag:
        im.show()
    if name!='':
        im.save(name)
    return im


# In[8]:

def score(test_dataset, test_labels, prototypes, prototype_labels, metric):
    res = np.empty((len(test_dataset)), dtype=np.uint8)
    D = metric(test_dataset, prototypes)
    res = np.argmin(D, axis=1)
    s=np.mean(res==test_labels)
    return res,s


# In[5]:

# Mean on each label
from sklearn.cluster import KMeans
def KMeanPrototypes(train_data, train_label, k):
    prototypes = []
    p_label = []
    for i in xrange(10):
        print 'clustering: ', i
        cur_train_dataset = train_data[train_label==i]
        kmeans = KMeans(n_clusters=k/10, random_state=0).fit(cur_train_dataset)
        for c in kmeans.cluster_centers_: prototypes.append(c)
        p_label += [i]*(k/10)
    return prototypes, p_label

def l2(a,b):
    return LA.norm(a-b)


# In[30]:

k = 10000
p,l = KMeanPrototypes(mnist.data[:60000], mnist.target[:60000], k)
pred, s = score(mnist.data[-10000:], mnist.target[-10000:], p, l, l2)
print s


# In[6]:

from tangentDistance import *
print oneSideTD(mnist.data[0],mnist.data[100])


# In[7]:

# TD
from mpdist import mpdist
from kmeans import kmeans

k=10
prototypes=[]
prototype_labels=[]
m1TD = lambda X,Y: mpdist(X, Y, oneSideTD, ("tangentDistance","ctypes","numpy"))
for i in xrange(10):
    print 'clustering %d-th label' %(i)
    train_dataset = mnist.data[mnist.target==i]
    test_dataset = mnist.target[mnist.target==i]
    init_center = np.array(train_dataset[:(k/10)])
    c,cid,dist=kmeans(train_dataset, init_center, metric=m1TD, pdist=True)
    for cc in c: prototypes.append(cc)
    prototype_labels += [i]*(k/10)


# In[10]:

cls, s=score(mnist.data[-10000:], mnist.target[-10000:], prototypes, prototype_labels, m1TD)


# In[11]:

# 10 prototypes using kmeans + one-sided tangent distance
s


# In[11]:

# 100 prototypes using kmeans + one-sided tangent distance
score


# In[35]:

# 100 prototypes using kmeans + two-sided tangent distance
pred


# In[73]:

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(mnist.data[mnist.target==0])


# In[87]:

p=np.array([[0 for i in xrange(28)] for j in xrange(28)], dtype=np.uint8)
for i in xrange(4):
    v=pca.components_[i,:]
    mx,mn = v.max(), v.min()
    img=imageshow((v-mn)/(mx-mn)*255, showflag=False)
    p=np.concatenate((p,img))


# In[94]:

p[0].shape


# In[13]:

for i in xrange(10):
    img=np.empty((28,28),dtype=np.uint8)
    for j in xrange(10):
        img=np.concatenate((img,np.reshape(prototypes[i*10+j],(28,28))))
        Image.fromarray(img[28:,:]).save(str(i)+'.jpg')


# In[18]:

import pp,time
ppservers=()
job_server = pp.Server(ppservers=ppservers)

print "Starting pp with", job_server.get_ncpus(), "workers"

start_time = time.time()

# The following submits 8 jobs and then retrieves the results
train_dataset = mnist.data[mnist.target==0]
cur = train_dataset[0]
jobs = [(i,job_server.submit(oneSideTD,([train_dataset[i]],[cur]), (), ("tangentDistance","ctypes","numpy"))) for i in xrange(len(train_dataset))]
dists=[0]*len(train_dataset)
for i,job in jobs:
    dists[i]=job()

print "Time elapsed: ", time.time() - start_time, "s"
job_server.print_stats()


# In[19]:

print dists[100]
oneSideTD([train_dataset[100]],[train_dataset[0]])


# In[8]:

from mpdist import mpdist
res = mpdist([mnist.data[0]],mnist.data[mnist.target==0], oneSideTD, ("tangentDistance","ctypes","numpy"))


# In[ ]:



