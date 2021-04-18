class ExponentialMovingAverage():                                                                                       
    def __init__(self, decay=0.99 ):                                                                                    
        self._decay = 0.99                                                                                              
        self._bias=1.                                                                                                   
        self._biased_avg=0.                                                                                             
    def average(self):                                                                                                  
        return self._biased_avg/(1-self._bias)                                                                          
    def __iadd__(self, val):                                                                                            
        self._biased_avg = self._biased_avg*self._decay + val*(1-self._decay)                                           
        self._bias*=self._decay                                                                                         
        return self        
