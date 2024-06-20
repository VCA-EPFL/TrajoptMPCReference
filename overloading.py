import numpy as np
import inspect
# from exampleHelpers import *


class matrix_(np.ndarray):
    operation_history = []
    line_search_iteration = 0
    iteration =0 # QP solve iteration
    soft_constraint_iteration = 0 # outer loop

    

    def __new__(cls, w):
        obj = np.asarray(w).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        #self.info = getattr(obj, 'info', None)
 
    def __mul__(self, o):
        self.save_op_history('mul',o)
        if(isinstance(o,matrix_)):
            return matrix_(np.matmul(self.view(np.ndarray),o.view(np.ndarray)))
        if(isinstance(o,np.ndarray)):
            return matrix_(np.matmul(self.view(np.ndarray),o))
        if(isinstance(o,int) | isinstance(o,float)):
            return matrix_(self.view(np.ndarray)*o)
        else:
            raise TypeError("Not Implement")
    
    def __sub__(self,o):
        self.save_op_history('sub',o)
        
    
        if(isinstance(o,matrix_)):
            return matrix_(self.view(np.ndarray)-o.view(np.ndarray))
        if(isinstance(o,np.ndarray)):

            return matrix_(self.view(np.ndarray)-o)
        else:
            print("Type\n", type(o))
            raise TypeError("Not Implement")
    
    def __add__(self, o):
        self.save_op_history('add',o)
    
        if(isinstance(o,matrix_)):
            return matrix_(self.view(np.ndarray)+o.view(np.ndarray))
        if(isinstance(o,np.ndarray)):
            return matrix_(self.view(np.ndarray)+o)
        if(isinstance(o,int) | isinstance(o,float)):
            return matrix_(self.view(np.ndarray)+o)
        else:
            raise TypeError("Not Implement")
        
    def __rsub__(self, o):
        return matrix_(o)-self


    def __rmul__(self,o):
        if(isinstance(o,np.ndarray)):
            return matrix_(o)*self
        if(isinstance(o,int)|isinstance(o,float)):
            return matrix_(o*self.view(np.ndarray))
    
    def __radd__(self,o):
        return matrix_(o)+self
        


    def dot(self,o, out=None):
        if(isinstance(o,matrix_)):
            return self*o
        if(isinstance(o,int)|isinstance(o,float)):
            return matrix_(o*self.view(np.ndarray))
        if(isinstance(o,np.ndarray)):
            if (self*o).shape==(1,):
                return float(self*o)
            else:
                return self*o
            
    def transpose(self):
        self.save_op_history('transpose',None)
        if(isinstance(self,matrix_)):
            return matrix_(np.transpose(self))
        else:
            return np.transpose(self)
            
    
    def __rmatmul__(self, o, out=None):
        return matrix_(o)*self
    
    def __matmul__(self,o,out=None):
        return self*matrix_(o)
    
    def linalg_solve(self,A,b):
        singular=False
        try:
            result = matrix_(np.linalg.solve(A, b))
            matrix_.save_op_history(A,'linalg_solve',b)
            return result


        except:
            singular=True #Warning singular system -- solving with least squares.")
            result, _, _, _ = matrix_(np.linalg.lstsq(A, b, rcond=None))
            matrix_.save_op_history(A,'linalg_solve_lstsq',b)

            return result, singular

    def invert_matrix(self):
        try:
            result=matrix_(np.linalg.inv(self))
            matrix_.save_op_history(self,'invert_matrix',None)
        except:
            print("Warning singular matrix -- using Psuedo Inverse.")
            result= matrix_(np.linalg.pinv(self))
            matrix_.save_op_history(self,'pseudo_invert_matrix',None)
        return result
    
    def diag(self):
        matrix_.save_op_history(self,'diag',None)
        return matrix_(np.diag(self))

    def vstack(A,B):
        matrix_.save_op_history(A,'vstack',B)
        return matrix_(np.vstack((A,B)))
    
    def hstack(A,B,C=None):
        if C is not None:
            matrix_.save_op_history(A,'hstack',[B,C])
            return matrix_(np.hstack((A,B,C)))
        else:
            matrix_.save_op_history(A,'hstack',B)
            return matrix_(np.hstack((A,B)))
    
    def reshape(self,shape):
        matrix_.save_op_history(self,'reshape',shape)
        if(isinstance(self,matrix_)):
            return matrix_(np.reshape(self,shape))
        else:
            return np.reshape(self,shape)
        
    def save_op_history(self,type,o):

        current_frame = inspect.currentframe()
        outerframes=inspect.getouterframes(current_frame, 2)
        if(outerframes[2].filename.split('/')[-1]=='overloading.py'): # if matmul, or rsub or rmul or radd, don't want to record this function call
            frames=outerframes[3:-3] #-3 because first calls are not interesting (twolinks, SQPexamples, runSQPexamples)
        else:
            frames=outerframes[2:-3] # by default start at 2 (0: save_op_history, 1: operation function)
        if(len(frames)>30):
            raise ValueError("Horizon higher than 30, is: ", len(frames))
        
        padding=[np.nan] * (30 - len(frames))# Fill up to 10 so everyline has same number of columns => checked before if 10 columns if enough for every operation
        filenames=[frame.filename.split('/')[-1] for frame in frames]
        functions=[frame.function for frame in frames]
        lines=[frame.lineno for frame in frames]

        iter_1=matrix_.iteration
        iter_2=matrix_.soft_constraint_iteration
        matrix_.operation_history.append((type, self, o, *filenames,*padding, *functions, *padding,*lines,*padding, iter_1, iter_2))
        
        
