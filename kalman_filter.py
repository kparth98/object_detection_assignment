import numpy as np
from pykalman import KalmanFilter

def getKF(init_mean,init_cov=100*np.eye(4)):
    tr_mtx = np.array([[1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    a = 0.1
#    a = -0.04
    tr_offset = np.array([0, a/2 , 0, a])
    
    kf = KalmanFilter(n_dim_state=4,
                      n_dim_obs=4,
                      transition_matrices=tr_mtx,
                      transition_offsets=tr_offset,
                      observation_matrices=tr_mtx,
                      observation_offsets=tr_offset,
                      initial_state_mean=init_mean,
                      initial_state_covariance=np.eye(4),
                      transition_covariance=init_cov,
                      observation_covariance=init_cov)
    return kf

def updateKF(kf,mean,cov, new_meas=None):
    if new_meas is None:
        return kf.filter_update(mean,cov)
    else:
        return kf.filter_update(mean,cov,new_meas)