import pandas as pd
import numpy as np


class Dataset:
    
    # Dependent quantities to be predicted
    _cols_y = ['Ca', 'P', 'pH', 'SOC', 'Sand']
    
    # Feature names not adhering to some prefix + number naming scheme
    _cols_other = ['CTI','ELEV','EVI','LSTD','LSTN','RELI','TMAP','TMFI','Depth']
    
    def __init__(self, f, sample_rate=.5):
        self.data = pd.read_csv(f)
        
        #Code the sole categorical feature as a binary numeric value
        self.data['Depth'] = self.data['Depth'].map({'Topsoil': 0, 'Subsoil': 1})
        self.cols = self.data.columns.tolist()
        #Remove features for CO2 spectra wavelengths -- competition hosts suggested doing this
        cols_co2_spectra = ['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11''m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54''m2356.61','m2354.68','m2352.76']
        self.data = self.data[[col for col in self.cols if col not in cols_co2_spectra]]
        # If sampling rate given, only use a fraction of 
        # the data for the rest of the analysis
        if sample_rate:
            n = len(self.data)
            self.data = self.data.iloc[np.random.randint(0,n,sample_rate*n)]
            
        # Set the unique identifier for each sample as well 
        # as other commonly used properties
        self.data.set_index('PIDN', inplace=True)    
        #redefine the column list after CO2 deletion        
        self.cols = self.data.columns.tolist()
        
    def N(self):
        """Return data set size"""
        return len(self.data)
    
    def X_IR(self):
        """Return data frame with IR absorption features"""
        return self.data[[col for col in self.cols if col.startswith('m')]]
    
    def X_nonPC(self):
        """Return data fraome for all features that have not yet been reduced via MDS of some kind"""
        return self.data[[col for col in self.X().columns.tolist() if not col.startswith('pc_') and col != 'Depth']]
    
    def X_BSA(self):
        """Return data frame for all features starting with 'BSA'"""
        return self.data[[col for col in self.cols if col.startswith('BSA')]]
    
    def X_REF(self):
        """Return data frame for all features starting with 'REF'"""
        return self.data[[col for col in self.cols if col.startswith('REF')]]
    
    def X(self):
        """Return data frame with data for all independent variables"""
        return self.data[[col for col in self.cols if not col in self._cols_y]]
    
    def Y(self):
        """Return data frame with data for all dependent variables"""
        return self.data[self._cols_y]
