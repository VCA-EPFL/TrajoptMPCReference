import numpy as np
import inspect


class matrix_(np.ndarray):
    operation_history = []

    def __new__(cls, w):

        obj = np.asarray(w).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        #self.info = getattr(obj, 'info', None)
 
    def __mul__(self, o):
        # print("mul")
        current_frame = inspect.currentframe()
        matrix_.operation_history.append(("mul", self, o, 
                                  *[frame.filename for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                  *[frame.function for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                  *[frame.lineno for frame in inspect.getouterframes(current_frame, 2)[1:4]]))
        
        if(isinstance(o,matrix_)):
            return matrix_(np.matmul(self.view(np.ndarray),o.view(np.ndarray)))
        if(isinstance(o,np.ndarray)):
            return matrix_(np.matmul(self.view(np.ndarray),o))
        if(isinstance(o,int) | isinstance(o,float)):
            return matrix_(self.view(np.ndarray)*o)
        else:
            raise TypeError("Not Implement")
    
    def __sub__(self,o):
        # print("sub")
        current_frame = inspect.currentframe()
        matrix_.operation_history.append(("sub", self, o, 
                                  *[frame.filename for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                  *[frame.function for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                  *[frame.lineno for frame in inspect.getouterframes(current_frame, 2)[1:4]]))
    
        if(isinstance(o,matrix_)):
            print("matrix")
            return matrix_(self.view(np.ndarray)-o.view(np.ndarray))
        if(isinstance(o,np.ndarray)):

            return matrix_(self.view(np.ndarray)-o)
        else:
            raise TypeError("Not Implement")
    
    def __add__(self, o):
        # print("add")
        current_frame = inspect.currentframe()
        # frame_info=[(frame.filename, frame.function, frame.lineno) for frame in inspect.getouterframes(current_frame, 2)[1:4]]
        # matrix_.operation_history.append(("add", self,o, frame_info ))
        matrix_.operation_history.append(("add", self, o, 
                                  *[frame.filename for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                  *[frame.function for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                  *[frame.lineno for frame in inspect.getouterframes(current_frame, 2)[1:4]]))

        # matrix_.operation_history.append(("add", self,o ))
        # for frame in inspect.getouterframes(current_frame, 2)[1:4]:
        #     matrix_.operation_history.append((frame.filename, frame.function, frame.lineno))
        if(isinstance(o,matrix_)):
            return matrix_(self.view(np.ndarray)+o.view(np.ndarray))
        if(isinstance(o,np.ndarray)):
            return matrix_(self.view(np.ndarray)+o)
        if(isinstance(o,int) | isinstance(o,float)):
            return matrix_(self.view(np.ndarray)+o)
        
        else:
            raise TypeError("Not Implement")
        
    def __rsub__(self, o):
        # print("rsub")
        return matrix_(o)-self


    def __rmul__(self,o):
        # print("rmul")
        if(isinstance(o,np.ndarray)):
            return matrix_(o)*self
        if(isinstance(o,int)|isinstance(o,float)):
            return matrix_(o*self.view(np.ndarray))
    
    def __radd__(self,o):
        # print("radd")
        return matrix_(o)+self
        


    def dot(self,o, out=None):
        # print("dot")
        if(isinstance(o,matrix_)):
            return self*o
        if(isinstance(o,int)|isinstance(o,float)):
            return matrix_(o*self.view(np.ndarray))
        if(isinstance(o,np.ndarray)):
            if (self*o).shape==(1,):
                return float(self*o)
            else:
                return self*o
        

    
    def __rmatmul__(self, o, out=None):
        # print("rmatmul")
        return matrix_(o)*self
    
    def __matmul__(self,o,out=None):
        # print("matmul")
        return self*matrix_(o)
    


