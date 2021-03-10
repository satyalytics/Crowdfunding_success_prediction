

class CrowdFunding:
    
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
    
        
    def encode(self, df, encoder):
        self.df, self.enc = apply_encoder(df, encoder)
    
        
    def scale(self, df, scaler):
        self.df, self.sclaer = apply_scaling(df, scaler)
    
        
    def apply_model(self, df, clf):
        pass
    
    
    def apply_bagging(self, df):
        pass
    
    
    def apply_stacking(self, df):
        pass
    
    
    def apply_boosting(self, df):
        pass



def main():
    pass


if __name__ == "__main__":
    main()