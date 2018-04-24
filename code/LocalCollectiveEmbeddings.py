from polara.recommender.models import RecommenderModel
from polara.recommender.data import RecommenderData
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from scipy import sparse
from polara.recommender.coldstart.data import ItemColdStartData
from polara.recommender.models import SVDModel
from IPython.display import clear_output
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

class LocalCollectiveEmbeddings(RecommenderModel):
    
    def __init__(self, *args, **kwargs):
        super(LocalCollectiveEmbeddings, self).__init__(*args, **kwargs)
        self.method = 'LCE'
    
    def get_train_content(self, content_data):
        self.train_content = content_data
    
    def get_content_shape(self):
        self.content_shape = {}
        for col in ['artistid', 'albumid']:
            self.content_shape[col] = self.train_content[col].max() + 1

    def get_training_content_matrix(self):
        self.get_content_shape()
        idx_userid = self.data.training[self.data.fields[0]].values
        val = np.ones(self.data.training.shape[0])
        
        i = 0
        features = []
        
        for col in ['artistid', 'albumid']:
            idx_feature = self.train_content[col].values
            shp = (idx_userid.max() + 1, 
                   self.content_shape[col])
        
            features_new = sparse.csr_matrix((val, (idx_userid, idx_feature)), 
                                             shape=shp)
            
            if i == 0:
                features = features_new
            else:
                features = sparse.hstack((features, features_new))
            
            i += 1
        
        return features
    
    def get_vectorizer(self, playlists):
        playlists['name'] = [str(playlists['name'].values[i]).lower() 
                             for i in range(playlists.shape[0])]
        
        names = playlists['name'].values
        
        for i in range(names.shape[0]):
            names[i] = re.compile('[^a-zA-Z0-9 ]').sub('', names[i].lower())
        playlists['name'] = names
            
        stop_words = stopwords.words('english') + ['music', 'song', 'songs', 'playlist', '']      
        
        vectorizer = CountVectorizer(binary=True)
        vectorizer.stop_words_ = stop_words
        vectorizer.fit(playlists.name.values)
        
        return vectorizer
    
    def get_train_name_matrix(self):
        playlists = self.train_content.drop_duplicates('pid')
        self.vectorizer = self.get_vectorizer(playlists)
        X = self.vectorizer.transform(playlists.name.values)
        return X
    
    def get_test_name_matrix(self):
        playlists = self.data.test.testset.drop_duplicates('pid')
        X = self.vectorizer.transform(playlists.name.values)
        return X
    
    def get_test_content_matrix(self):
        idx_userid = self.data.test.testset[self.data.fields[0]].values
        val = np.ones(self.data.test.testset.shape[0])
        
        i = 0
        features = []
        
        for col in ['artistid', 'albumid']:
            idx_feature = self.data.test.testset[col].values
            shp = (idx_userid.max() + 1, 
                   self.content_shape[col])
        
            features_new = sparse.csr_matrix((val, (idx_userid, idx_feature)), 
                                             shape=shp)
            
            if i == 0:
                features = features_new
            else:
                features = sparse.hstack((features, features_new))
            
            i += 1
        
        
        return features
        
        
    def construct_closeness_matrix(self, X, rank, binary=False):
        print ('Construct closeness matrix...')
        nbrs = NearestNeighbors(n_neighbors=1 + rank).fit(X)
        if binary:
            closeness_matrix = nbrs.kneighbors_graph(X)
        else:
            closeness_matrix = nbrs.kneighbors_graph(X, mode='distance')
        print ('Done.')
            
        return closeness_matrix
    
    def get_constant(self, R, X):
        trRtR = tr(R, R)
        trXtX = tr(X, X)
        return trRtR, trXtX
    
    def update_factors(self, R, X, U, V, H, A, D, 
                       alpha, beta, lamb):
        
        gamma = 1. - alpha
        
        UtU = U.T.dot(U)
        UtR = U.T.dot(R)
        UtX = U.T.dot(X)
        UtUV = UtU.dot(V)
        UtUH = UtU.dot(H)
        DU = D.dot(U)
        AU = A.dot(U)
        
        #update V
        V_1 = np.divide((alpha * UtR), 
                        (alpha * UtUV + lamb * V).maximum(1e-10))
        V = V.multiply(V_1)
            
        #update H
        H_1 = np.divide(
            (gamma * UtX), (gamma * UtUH + lamb * H).maximum(1e-10))
        H = H.multiply(H_1)
            
        # update U
        U_t1 = alpha * R.dot(V.T) + gamma * X.dot(H.T) + beta * AU
        U_t2 = alpha * U.dot(V.dot(V.T)) + gamma * \
        U.dot(H.dot(H.T)) + beta * DU + lamb * U
            
        U_t3 = np.divide(U_t1, (U_t2).maximum(1e-10))
        U = U.multiply(U_t3)
        
        #calculate oblective function without constant
        
        tr1 = alpha * ((-2.) * tr(V, UtR) + tr(V, UtUV))
        tr2 = gamma * ((-2.) * tr(H, UtX) + tr(H, UtUH))
        tr3 = beta * (tr(U, DU) - tr(U, AU))
        tr4 = lamb * (UtU.diagonal().sum() + tr(V, V) + tr(H, H))

        Obj = tr1 + tr2 + tr3 + tr4
        
        
        return U, V, H, Obj
        
        
        
    def build(self, content_data, 
              rank=10, alpha=0.1, beta=0.005, lamb=0.0001, 
              epsilon=0.01, seed=0,maxiter=150, verbose=True):
        
        self.get_train_content(content_data)
        
        R = self.get_training_matrix(dtype='float64')
        X = self.get_training_content_matrix()
        bag_of_names = self.get_train_name_matrix()
        X = sparse.hstack((X, bag_of_names))
        A = self.construct_closeness_matrix(X, rank, binary=True).tocsr()
        
        num_users = R.shape[0]
        num_items = R.shape[1]
        num_features = X.shape[1]
        
        U = np.abs(sparse.rand(num_users, rank, 0.99, 'csr', dtype=R.dtype, random_state=seed))
        V = np.abs(sparse.rand(rank, num_items, 0.99, 'csr', dtype=R.dtype, random_state=seed))
        H = np.abs(sparse.rand(rank, num_features, 0.99, 'csr', dtype=R.dtype, random_state=seed))
        
        
        #auxiliary constant   
        D = sparse.dia_matrix((A.sum(axis=0), 0), A.shape)
        trRtR, trXtX = self.get_constant(R, X)

        itNum = 1
        delta = 2.0 * epsilon

        ObjHist = []
        
        while True:

            U, V, H, Obj = self.update_factors(R, X, U, V, H, A, D, 
                                               alpha, beta, lamb)
            Obj += alpha*trRtR + (1. - alpha)*trXtX
            Obj = Obj / (num_users * num_features * num_items * rank)
            ObjHist.append(Obj)
            
            if itNum > 1:
                delta = abs(ObjHist[-1] - ObjHist[-2])
                if verbose:
                    print ("Iteration: ", itNum, "Objective: ", Obj, "Delta: ", delta)
                if itNum > maxiter or delta < epsilon:
                    break

            itNum += 1
            
        self.user_factors = U
        self.feature_factors = H 
        self.item_factors = V
        
        
    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        slice_data = self._slice_test_data(test_data, start, stop)
        features = self.get_test_content_matrix()
        names = self.get_test_name_matrix()
        features = sparse.hstack((features, names))
        Ut = np.linalg.lstsq(self.feature_factors.T.toarray(), 
                             features.T.toarray(), rcond=-1)[0]
        R = Ut.T.dot(self.item_factors.toarray())
        return R, slice_data
    
    
    
def reindex_content(content_data, col, sort=True, inplace=True):
    grouper = content_data.groupby(col, sort=sort).grouper
    new_val = grouper.group_info[1]
    old_val = grouper.levels[0]
    val_transform = pd.DataFrame({'old': old_val, 'new': new_val})
    new_data = grouper.group_info[0]

    if inplace:
        result = val_transform
        content_data.loc[:, col] = new_data
    else:
        result = (new_data, val_transform)
    return result
        
def reindex_content_columns(content_data, columns):
    index_content = {}
    for col in columns:
        index_content[col] = reindex_content(content_data, col)
    return index_content

def tr(A, B):
    x = A.multiply(B)
    return (x.sum(axis=0)).sum(axis=1)