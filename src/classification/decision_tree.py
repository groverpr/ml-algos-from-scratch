from src.libraries import *


class VanillaDecisionTreeClassifier(ABC):
    def __init__(self, target='y', max_depth=3, min_leaf_size=1, frac_max_features=1):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size 
        self.target = target # name of target feature. single dataframe with feature and target is passed during fit
        self.lowest_impurity = 1e3 # initialization of high number
        self.frac_max_features = frac_max_features  # fraction of features to be used in each split (useful for random forest)
    
    def total_split_impurity(self, left_df, right_df):
        """
        Measurement used to find best feature-value pair for each split of classification decision trees.  
        
        This function gives total gini impurity value (scalar) given data that goes in left and right nodes
        
        left_df: (pd.DataFrame) Data going to left node after split
        right_df: (pd.DataFrame) Data going to right node after split
        
        Return: total gini impurity by node split weighted by samples going in left and right nodes
        """
        gini_left = self.gini_impurity(left_df)
        gini_right = self.gini_impurity(right_df)
        total_impurity = gini_left*(left_df.shape[0]/(left_df.shape[0] + right_df.shape[0])) + \
        gini_right*(right_df.shape[0]/(left_df.shape[0] + right_df.shape[0]))
        return total_impurity
    
    def gini_impurity(self, df):
        """
        Gini impurity of single node. For more info: https://www.learndatasci.com/glossary/gini-impurity/
        Ranges betwee 0 to 0.5 where 0 means most pure node 
        and a great candidate for split
        
        df: (pd.DataFrame) Data in a given node
        
        Return: gini impurity
        """
        ypos = (df[self.target] == 1).sum()
        yneg = (df[self.target] == 0).sum()
        ppos = ypos/(ypos+yneg)
        pneg = 1-ppos
        return 1 - ((ppos**2) + (pneg**2))
        
                          
    def find_best_split(self, df, current_depth=0):
        """
        Given data in current node (df), it finds feature name and it's value that would give best possible split.
        Uses total split impurity as measurement of best split. i.e. total split impurity should be lowest among all options
        This function is called recursively starting from root to leaf until stopping criteria is reached.
        
        df: (pd.DataFrame) Data in current node that we are trying to split
        current_depth: (int) Current depth of tree. 
                             Required to track the depth in order to stop when max depth is reached
                             
        Return: tree_dict (dict) Nested dictionary containing tree node attributes from root to leaves  
        """
        # only split if the new nodes give lower impurity than what we have in current node
        self.lowest_impurity = self.gini_impurity(df) 
        
        # Check all features to find feature value pair giving that would give best split 
        feats = df.drop(self.target,axis=1).columns
        n_feats = len(feats)
        if self.frac_max_features<1.:
            feats = np.random.permutation(feats)[: round(self.frac_max_features*n_feats)]
        for feat_name in feats:
            self.find_best_split_single_feat(df, feat_name)
        
        # if no better split found after all search, just return the y_mean value at this node
        if self.lowest_impurity >= self.gini_impurity(df) or current_depth > self.max_depth: 
            return df[self.target].mean()
        
        best_feat_name, y_bucket, best_feat_val = self.best_feat_name, self.y_bucket, self.best_feat_val
        # tree dict is used to save nested tree structure that will later be used to predict
        tree_dict = defaultdict() 
        tree_dict[best_feat_name]  = {}
        tree_dict[best_feat_name]['feat_value'] = best_feat_val
        tree_dict[best_feat_name]['ymean'] = y_bucket
        
        # simply sort by selected feature name and take left-right
        df_sort = df.copy().sort_values(by=self.best_feat_name, ascending=True)  # Not optimal to sort each time
        left_tree_df = df_sort[df_sort[self.best_feat_name] <= self.best_feat_val]
        right_tree_df = df_sort[df_sort[self.best_feat_name] > self.best_feat_val]

        if (len(left_tree_df)==0) or (len(right_tree_df)==0):  # Creating balanced tree (optional)
            return
                
        # Go down the tree if it finds better splits. If it does, create nested dicts and replace value at nodes
        current_depth+=1
        tree_dict[best_feat_name]['left'] = self.find_best_split(left_tree_df, current_depth)
        tree_dict[best_feat_name]['right'] = self.find_best_split(right_tree_df, current_depth)
        
        return tree_dict
    
    def find_best_split_single_feat(self, df, feat_name):
        
        """
        Given data in current node (df) and feature name to check, find the value that gives best possible 
        node split (of df) at this feature. Finding a feature value that gives lower impurity than 
        currently found impurity is not guaranteed. It will only select feature-value pair if it reduces the impurity.
        
        df: (pd.DataFrame) Data in current node we are trying to split
        feat_name: (str) Feature name we are trying to split on
        
        Return: None. Updates characterstics of best split value in class object. 
        """
        
        # It is not optimal to sort and reset index each time
        # But we are not really optimizing for now
        # This is a simple vanilla implementation for understanding
        df_sort= df.copy().sort_values(by=feat_name)
        df_sort.reset_index(drop=True, inplace=True)
        uniq_vals = df_sort[feat_name].unique()  # only check unique values to save some redundancy
        for i in uniq_vals: #range(self.min_leaf_size, df_sort.shape[0] - self.min_leaf_size): 
            
            # Left and Right datasets should be equal or larger than min_leaf_size. 
            left_df = df_sort[df_sort[feat_name] <= i] 
            right_df = df_sort[df_sort[feat_name] > i]
            
            if len(left_df)<self.min_leaf_size or len(right_df)<self.min_leaf_size:
                continue # skip this value if it gives a split node smaller than min accepted leaf size

            new_impurity = self.total_split_impurity(left_df, right_df)

            if new_impurity < self.lowest_impurity:
                # Update best split node attributes if new impurity is lower than lowest sofar
                self.lowest_impurity = new_impurity
                self.best_feat_name = feat_name
                self.best_feat_val = i
                self.y_bucket = df_sort[self.target].mean()

        
    def fit(self, df):
        """
        Calls find_best_split function and fits the tree with input dataframe
        
        df: (pd.DataFrame) Training data to fit the decision tree classfier
        
        Returns: (dict) Fitted tree 
        """
        self.fitted_tree = self.find_best_split(df)
        return self.fitted_tree
        
#     @property
#     def better_split_not_found(self): return self.lowest_impurity == 1e3  # impurity is not reduced 
    
    @staticmethod
    def _is_leaf_node(subtree):
        """
        subtree: (dict) Given a branch of tree, check whether it is leaf
        
        Return (bool) True if given tree branch is leaf node, False otherwise
        """
        if not subtree['left'] and not subtree['right']:  # if leaf, would not have any children
            return True  
        else: return False
    
    @staticmethod
    def _predict_row(fitted_tree, row):
        """
        fitted_tree: (dict) Output of self.fit
        row: (pd.DataFrame) Single row of test dataframe
        
        Return (Float) Prediction value for given input row. Between 0 to 1
        
        From top to bottom, traverse the whole tree and find which node does this data point lies in
        """
        
        if type(fitted_tree) == float:
            return fitted_tree # leaf has no further children
        
        key = list(fitted_tree.keys())[0] # feature index of current node
        fitted_key = fitted_tree[key]
        
        if VanillaDecisionTreeClassifier._is_leaf_node(fitted_key):
            return fitted_key['ymean']
        
        if fitted_key['feat_value'] >= row[key]:
            return VanillaDecisionTreeClassifier._predict_row(fitted_key['left'], row)
        else:
            return VanillaDecisionTreeClassifier._predict_row(fitted_key['right'], row)
        

        
    def predict(self, X_test):
        """
        Runs predict_row for each row of test dataframe
        
        fitted_tree: (dict) Output of self.fit
        X_test: (pd.DataFrame) Dataframe containing test data you want to predict using fitted decision tree
        
        Return: (List) Predictions from tree in given test data
        """
        
        preds = [VanillaDecisionTreeClassifier._predict_row(self.fitted_tree, X_test.iloc[i,:])\
                 for i in range(X_test.shape[0])]
        return preds
    
 