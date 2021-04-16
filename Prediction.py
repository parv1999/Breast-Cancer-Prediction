
import pandas as pd

import numpy as np 

import random as rn

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import learning_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics 

from sklearn.metrics import classification_report 

from sklearn.metrics import roc_curve

import seaborn as sns 

#data set url :- https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

df = pd.read_csv("/home/parv/datasets/breast_cancer.csv")

df['class'][ df[ 'class' ] == 4 ] = 1
    
df['class'][ df[ 'class' ] == 2 ] = 0

print("Missing Values",np.sum(df==np.nan)+np.sum(df=="?"))

y = df['class'].values

x = df.drop(['class'], axis = 1)

x = (x - np.min(x))/(np.max(x) - np.min(x)).values 

x_train, x_test, y_train, y_test = train_test_split( x , y , test_size = 0.20, random_state = 42) 

plt.figure()

sns.heatmap(df.corr(),annot=True,cmap=sns.color_palette("flare", as_cmap=True))


class tools():

    def get_accuracy( self , correct_labels,predictions):
        
        return 100*np.sum(correct_labels==predictions)/correct_labels.shape[0]  
                
  
    def Learning_Curves( self , est , x_train , y_train , title ):
        
        train_sizes = [ 10 , 50 , 100 , 200 , 300 , 400 ]
        
        train_sizes , train_scores , validation_scores = learning_curve(estimator = est , X = x_train , y = y_train , cv = 5 , scoring = 'f1' ,train_sizes = train_sizes , shuffle=True )  
        
        train_scores_mean = np.mean( train_scores , axis = 1 )
        
        validation_scores_mean = np.mean( validation_scores , axis = 1  )
        
        plt.plot( train_sizes , train_scores_mean , label = 'Training F1 Score' )
        
        plt.plot( train_sizes , validation_scores_mean , label = 'Testing F1 Score' )
                
        plt.title(title)
        
        plt.legend()
        
class Knn( tools ):
        
        
    def get_distance( self , l1 , l2 ):
    
        dist = 0 
        
        for i in range( len( l1 ) ):
            
            dist = dist + ( l1[ i ] - l2[ i ] )**2 
            
        return dist**0.5 
    
    
    def get_k_nearest_labels( self , x_train , y_train , x_test , k ):
        
        distance = [ ]
        
        for i in range( len( x_train ) ) :
            
            distance.append( [ i , self.get_distance( x_test , x_train[ i ] ) ] )
            
        distance.sort( key = lambda x:x[ 1 ] )                     
        
        row_class = [ ]
        
        for i in range( 0 , k ):
            
            row_class.append( y_train[ distance[ i ][ 0 ]  ] ) 
            
        return row_class
    
    
    def get_mode( self , row_class ) :
        
        d = { }
        
        d[0] = [ ]
        
        d[1] = [ ]
        
        for i in row_class:
            
            d[ i ].append( i )
        
        if len( d[ 0 ] ) > len( d[ 1 ] ) :
            
            return d[ 0 ][ 0 ] 
    
        elif len( d[ 0 ] ) == len( d[ 1 ] ) :
            
            choices = [ 0 , 1 ]
            
            return rn.choice( choices )
        
        elif len( d[ 0 ] ) < len( d[ 1 ] ) :
            
            return d[ 1 ][ 0 ]     
    
       
    def classify_knn( self , x_test , x_train , y_train, k ) :
        
        predictions   = [ ]
        
        for mat in x_test:
        
            label = self.get_mode( self.get_k_nearest_labels( x_train , y_train , mat , k ) ) 
        
            predictions.append( label )
            
        return predictions

    
    def run_test( self , k , y_test , x_test , x_train , y_train ):

        test_pred_labels = { }
        
        train_pred_labels = { }
        
        for K in k:
            
            pred = self.classify_knn( x_test , x_train , y_train , K )
            
            acc_test = self.get_accuracy( y_test , pred )
            
            print(f"pred ( on testing data ): at K = { K } Acc = { acc_test } \n")
            
            test_pred_labels[ K ] = [ acc_test , pred ]
            
            pred_2 = self.classify_knn( x_train , x_train , y_train , K )
            
            acc_train = self.get_accuracy( y_train, pred_2 )
            
            print(f"pred ( on training data ): at K = { K } Acc = { acc_test } \n")
            
            train_pred_labels[ K ] = [ acc_train , pred_2 ]
            
        plt.figure()    
            
        self.Learning_Curves( KNeighborsClassifier(n_neighbors=5) , x_train, y_train, title = 'KNN' )
            
        pred = test_pred_labels[ max( test_pred_labels , key=test_pred_labels.get ) ][ 1 ]
        
        cnf_mat = metrics.confusion_matrix(y_test , pred)
            
        plt.figure()
        
        sns.heatmap(pd.DataFrame(cnf_mat),annot=True,cmap='YlGnBu')
        
        fpr, tpr, thresholds = roc_curve(y_test, pred)
        
        print("KNN classification report")
        
        print(classification_report(y_test,pred))
        
        plt.figure()
        
        plt.plot([0,1],[0,1],'k--')
        
        plt.plot(fpr,tpr, label='Knn')
        
        plt.xlabel('fpr')
        
        plt.ylabel('tpr')
        
        plt.title('Knn ROC curve')
        
        plt.show()
            
        return test_pred_labels , train_pred_labels 
    



# KNN Implementaion 
    
k = [ i for i in range(1,4) ]

X_train = x_train.to_numpy()

X_test = x_test.to_numpy()

Y_train = y_train

Y_test = y_test

test_pred_labels , train_pred_labels = Knn().run_test( k , Y_test , X_test , X_train , Y_train )


# Logistic Regression Implementation 

class Logistic__Regression(tools):
    
    def init_weight_bias( self , dim ):
        
        w = np.full((dim,1),0.01)
        b=0.0
        
        return w,b 
    
    def sigmoid( self , z ):
        
        y_hat = 1/(1 + np.exp(-z))
        
        return y_hat
    
    
    def cost_function( self , x_train , y_train , w  , b  ):         
        
        Z = np.dot( w.T , x_train ) + b
        
        yhat = self.sigmoid( Z )
        
        loss = -y_train*np.log(yhat) - (1-y_train)*np.log(1-yhat)  
        
        cost = ( np.sum( loss ) ) /x_train.shape[1]
        
        dw = ( np.dot(  x_train , ( yhat - y_train ).T ) )/x_train.shape[1]
        
        db = ( np.sum( ( yhat - y_train ).T ) )/x_train.shape[1]
            
        gradient = { 'db':db , 'dw':dw }
        
        return cost , gradient
    
    def optimization( self , max_iter , alpha , x_train , y_train , w , b  ):
        
        index =[]    
        cost_b_w = [ ]
        cost_ = []
        #w = np.array( w ).reshape( 1 , len(w) )
        
        for i in range( max_iter ):
                
                cost , grad = self.cost_function( x_train , y_train , w , b )  
                
                dw = grad['dw']
                
                db = grad['db']
                
                w = w - alpha * dw
            
                b = b - alpha * db
                
                if i % 100== 0:
                    
                    print(f'Cost after {i}th iteration :{cost}' )
                    
                    cost_.append(cost)
                    index.append(i)
                    
                cost_b_w.append( ( cost , b , w ) )
    
        plt.plot(index, cost_ )
        plt.xlabel("Number of Iterarion") 
        plt.ylabel("Cost") 
        plt.show() 
    
        cost_b_w.sort( key = lambda x:x[ 1 ] )
    
        return cost_b_w[ 0 ]
    
    
    def Classify( self , x_test , w , b ):
        
        Z = self.sigmoid( np.dot( w.T , x_test ) + b )
        
        predictions = np.zeros((1,x_test.shape[1]))
        
        for i in range( Z.shape[1] ):
            
            if Z[0,i] >= 0.50 :
                
                predictions[0,i] = 1
                
            else:
                
                predictions[0,i] = 0
                
        return predictions
    
    def Logistic_Regression( self , x_train , x_test , y_train , y_test , max_iter , alpha ):
        
        dim = x_train.shape[0]
        
        w,b = self.init_weight_bias(dim)
        
        cost_b_w = self.optimization( max_iter , alpha , x_train , y_train , w , b  )
        
        b = cost_b_w[ 1 ]
        
        w = cost_b_w[ 2 ]
        
        pred_train = self.Classify( x_train , w , b )
        
        pred_test  = self.Classify( x_test  , w , b )
        
        print("Train Accuracy : {} ".format( self.get_accuracy(y_train , pred_train ) ) )
        
        print("Test Accuracy : {} ".format( self.get_accuracy(y_test , pred_test ) ) )

        plt.figure()    
            
        self.Learning_Curves( LogisticRegression() , x_train.T, y_train.T, title = 'Logistic Regression' )
        
        cnf_mat = metrics.confusion_matrix(y_test.T , pred_test.T)
            
        plt.figure()
        
        sns.heatmap(pd.DataFrame(cnf_mat),annot=True,cmap='YlGnBu')
        
        fpr, tpr, thresholds = roc_curve(y_test.T, pred_test.T)
        
        print("Logistic Regression classification report")
        
        print(classification_report(y_test.T,pred_test.T))
        
        plt.figure()
        
        plt.plot([0,1],[0,1],'k--')
        
        plt.plot(fpr,tpr, label='Logisic Regression')
        
        plt.xlabel('fpr')
        
        plt.ylabel('tpr')
        
        plt.title('Logistic Regression ROC curve')
        
        plt.show()
                  
        return  pred_train , pred_test 
    
    
x_train = x_train.T 

x_test = x_test.T 

y_train = y_train.T 

y_test = y_test.T 

pred_train , pred_test= Logistic__Regression().Logistic_Regression( x_train , x_test , y_train , y_test , max_iter =1000 , alpha = 0.05) 



    
        


