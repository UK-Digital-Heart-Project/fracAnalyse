import matplotlib
import matplotlib.pyplot as plt
import warnings

def using_hg2(fig=None):
    global tf_cached
    
    try:
        tf_cached
    except NameError:
        tf_cached = None

    if tf_cached is None:
        try:
            if fig is None:
                fig = plt.figure()
                created_fig = True
            else:
                created_fig = False

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # Simulate the check for modern graphics (like HG2 in MATLAB)
                    tf = matplotlib.__version__ >= '1.4.0'  # Matplotlib version that represents modern features
                except:
                    tf = False

            if created_fig:
                plt.close(fig)
            
            tf_cached = tf
        except:
            tf_cached = False
    
    return tf_cached
