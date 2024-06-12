import numpy as np
import inspect


class matrix_(np.ndarray):
    operation_history = []
    horizon=4

    def __new__(cls, w):

        obj = np.asarray(w).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        #self.info = getattr(obj, 'info', None)
 
    def __mul__(self, o):
        current_frame = inspect.currentframe()
        matrix_.operation_history.append(("mul", self, o, 
                                  *[frame.filename.split('/')[-1] for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                #   [frame.filename for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                          
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
        current_frame = inspect.currentframe()
        matrix_.operation_history.append(("sub", self, o, 
                                  *[frame.filename.split('/')[-1] for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                #   [frame.filename for frame in inspect.getouterframes(current_frame, 2)[1:4]],

                                  *[frame.function for frame in inspect.getouterframes(current_frame, 2)[1:4]],
                                  *[frame.lineno for frame in inspect.getouterframes(current_frame, 2)[1:4]]))
    
        if(isinstance(o,matrix_)):
            return matrix_(self.view(np.ndarray)-o.view(np.ndarray))
        if(isinstance(o,np.ndarray)):

            return matrix_(self.view(np.ndarray)-o)
        else:
            print("Type\n", type(o))
            raise TypeError("Not Implement")
    
    def __add__(self, o):
        current_frame = inspect.currentframe()
        # frame_info=[(frame.filename, frame.function, frame.lineno) for frame in inspect.getouterframes(current_frame, 2)[1:4]]
        # matrix_.operation_history.append(("add", self,o, frame_info ))
        frames=inspect.getouterframes(current_frame, 2)[1:self.horizon]
        filenames=[frame.filename.split('/')[-1] for frame in frames]
        functions=[frame.function for frame in frames]
        lines=[frame.lineno for frame in frames]
        matrix_.operation_history.append(("add", self, o, *filenames, *functions, *lines))

        # matrix_.operation_history.append(("add", self, o, 
        #                           [frame.filename.split('/')[-1] for frame in inspect.getouterframes(current_frame, 2)[1:4]],
        #                           [frame.function for frame in inspect.getouterframes(current_frame, 2)[1:4]],
        #                           [frame.lineno for frame in inspect.getouterframes(current_frame, 2)[1:4]]))

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
        n=self.shape       
        if(len(n)==1):
            n=n+(1,)
        return self.reshape(n[::-1])
        # print("transpose\n",self.reshape(n[::-1]).shape)
    
    def __rmatmul__(self, o, out=None):
        return matrix_(o)*self
    
    def __matmul__(self,o,out=None):
        return self*matrix_(o)
    


