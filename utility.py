import os
import matlab
import matlab.engine

class MatlabEngine():
    """
    Matlab Engine Class, used for python incorporate with matlab
    """
    def __init__(self):
        project_address = os.getcwd()
        self.matlab_engine = matlab.engine.start_matlab()
        # Add the search dir of matlab
        self.matlab_engine.addpath(os.path.join(project_address, 'evaluation_methods'))
        self.matlab_engine.addpath(os.path.join(project_address, 'evaluation_methods', "matlabPyrTools"))

    def start_matlab_engine(self):
        self.matlab_engine = matlab.engine.start_matlab()
    
    def quit_matlab_engine(self):
        self.matlab_engine.quit()
        
    def __del__(self):
        try:
            self.matlab_engine.quit()
        except:
            pass