from src.libraries import *

def find_node_dir_val(subtree, dir_vals = []):
    if type(subtree) == float:
        return
    
    key = list(subtree.keys())[0] # would be "x" or "y"
    value = subtree[key]
    if key == 'x':
        direction = 'vline'
    elif key == 'y':
        direction = 'hline'
    else:
        raise ValueError('dataframe should have x or y columns')
        
    if type(value) == float:
        return
    else:
        dir_vals.append({'dir': direction, 'val': value['feat_value']})
        find_node_dir_val(value['left'], dir_vals)
        find_node_dir_val(value['right'], dir_vals)
    return dir_vals


def plot_2d_grids_from_fitted_tree(fitted_tree, df, preds, xrange=(0, 100), yrange=(0, 100), alpha=1):
    """
    For visualization purposes: 
        Tree should be fitted on 2D feaatures named "x" and "y"
    
    fitted_tree: (Dict) Output of fit from VanillaDecisionTreeClassifier object
    df:  
    preds:     
    xrange: (Tuple): min-max of x axis, for plotting
    yrange: (Tuple): min-max of x axis, for plotting
    
    Return: (None) A matplotlib plot with grids as given by tree
    """
    
    # Recursively create vertical or horizontal
    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Axis1: Adding data points from input df
    ax.plot(
        df[df['class'] == 0]['x'], 
        df[df['class'] == 0]['y'], 
        'o',
        color=green)
    ax.plot(
        df[df['class'] == 1]['x'], 
        df[df['class'] == 1]['y'], 
        'o',
        color=red)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Original Data')
    ax.grid()
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    
    # Axis 2: Creating decision tree boundarires
    dir_vals = find_node_dir_val(fitted_tree)
    for dir_val_dicts in dir_vals:
        direction, value = dir_val_dicts['dir'], dir_val_dicts['val']
        if direction == "vline": 
            ax2.vlines(value, yrange[0], yrange[1], color='k', alpha=alpha)
        elif direction == "hline": 
            ax2.hlines(value, xrange[0], xrange[1], color='k', alpha=alpha)
        else:
            raise ValueError('Direction should be either vline or hline. Check')

    # Adding prediction + colors based on predictions
    df['preds'] = preds
    df['color'] = df.preds.apply(lambda p: colavg(red, green, p))
    # Adding data points to tree      
    ax2.scatter(
        df['x'],
        df['y'],
        color=df['color'])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Decision Tree + Colors based on predictions')
    ax2.grid()
    ax2.set_xlim(xrange[0], xrange[1])
    ax2.set_ylim(yrange[0], yrange[1])
    
    plt.tight_layout()

# Colors 
red = (1, 0, 0)
green = (0, 1, 0)
colavg = lambda c1, c2, w1: tuple([np.sqrt((c1[i]**2)*w1 + (c2[i]**2)*(1-w1)) for i in range(3)])


def plot_2d_grids_from_fitted_rf(rf, df, preds, n_trees=25, xrange=(0, 100), yrange=(0, 100), alpha=0.5):
    """
    rf: (VanillaRandomForestClassifier) Fitted random forest classifier
    df: (pd.Dataframe) Data to plot on left side. Contains "x", "y" and "class"
    preds: (np.array|List) List/array of predictions to use for right plot. Same length as data
    """
    #### Axis 1 #### 
    # Recursively create vertical or horizontal
    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Axis1: Adding data points from input df
    ax.plot(
        df[df['class'] == 0]['x'], 
        df[df['class'] == 0]['y'], 
        'o',
        color=green)
    ax.plot(
        df[df['class'] == 1]['x'], 
        df[df['class'] == 1]['y'], 
        'o',
        color=red)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Original Data')
    ax.grid()
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    
    #### Axis 2 ####
    # Creating decision tree boundarires
    for n in range(n_trees):
        dir_vals = find_node_dir_val(rf.dts[n].fitted_tree)
        for dir_val_dicts in dir_vals:
            direction, value = dir_val_dicts['dir'], dir_val_dicts['val']
            if direction == "vline": 
                ax2.vlines(value, yrange[0], yrange[1], color='k', alpha=alpha)
            elif direction == "hline": 
                ax2.hlines(value, xrange[0], xrange[1], color='k', alpha=alpha)
            else:
                raise ValueError('Direction should be either vline or hline. Check')
            
    # Adding prediction + colors based on predictions
    df['preds'] = preds
    df['color'] = df.preds.apply(lambda p: colavg(red, green, p))
    # Adding data points to tree      
    ax2.scatter(
        df['x'],
        df['y'],
        color=df['color'])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Random Forest + Colors based on predictions')
    ax2.grid()
    ax2.set_xlim(xrange[0], xrange[1])
    ax2.set_ylim(yrange[0], yrange[1])
    
    plt.tight_layout()