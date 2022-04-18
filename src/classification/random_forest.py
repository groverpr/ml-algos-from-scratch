from src.libraries import *
from decision_tree import VanillaDecisionTreeClassifier


class VanillaRandomForest(VanillaDecisionTreeClassifier):
    def __init__(self, 
                 n_trees = 50, 
                 target='y',
                 max_depth=3,
                 min_leaf_size=1, 
                 frac_max_features = 0.8,
                 bootstrap = True,
                 random_state=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size 
        self.frac_max_features = frac_max_features  # for each node split - > to create variety b/w trees
        self.target = target # name of target feature. single dataframe with feature and target is passed during fit
        self.lowest_impurity = 1e3 # initialization of high number
        self.bootstrap = bootstrap  # Boolean
        self.random_state = random_state  # for reproducibilty
    
    def create_random_forest(self, df):
        """
        Create n_trees that define the ensemble. The variety b/w each tree is brought by two 
        important parameters: 
        1. max_features: that only passes fraction of total features to find best split each time. 
                         since this is implemented at node level, it needs to be implemented in tree
        2. bootstrap: that creates a random sample (with sampling) of rows to create a trees. 
                      different bootstrap for different trees for variety
        """
        dts = []
        for n in range(self.n_trees):
            print(f'Training tree number {n}')
            dt = VanillaDecisionTreeClassifier(  # can replace this by self.parent something later
                max_depth=self.max_depth, 
                min_leaf_size=self.min_leaf_size,
                frac_max_features = self.frac_max_features,
                target = self.target
            )
            if self.bootstrap:  # ensemble will be better with bootstrap
                df_sample = df.copy().sample(frac=1, replace=True, random_state=n*self.random_state) 
            else:
                df_sample = df.copy()
            dt.fit(df_sample)
            dts.append(dt)
        return dts
    
    
    def fit(self, df):
        self.dts = self.create_random_forest(df)
        
    
    def predict(self, X_test):
        """
        Just predict from each tree and take simple average of results
        """
        preds_all_dts = np.array([dt.predict(X_test) for dt in self.dts])
        return preds_all_dts.mean(0)
    
    