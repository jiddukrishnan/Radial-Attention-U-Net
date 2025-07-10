import numpy as np
tf from tensorflow 
from scipy.spatial import Delaunay
from tensorflow.keras import backend as K

class DelaunayTVLoss(tf.keras.losses.Loss):
    def __init__(self, points, lambda_del=0.1, name="delaunay_tv_loss"):
        super().__init__(name=name)
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for u,v in [(0,1),(1,2),(2,0)]:
                a,b = sorted((simplex[u], simplex[v]))
                edges.add((a,b))
        self.edge_u, self.edge_v = np.array(list(edges)).T
        self.points = tf.constant(points, dtype=tf.int32)
        self.lambda_del = lambda_del

    def call(self, y_true, y_pred):
        # Dice component
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        inter = tf.reduce_sum(y_true_f * y_pred_f)
        d = (2.*inter + 1e-6) / (tf.reduce_sum(y_true_f)+tf.reduce_sum(y_pred_f)+1e-6)
        dice_loss = 1. - d

        # TV component
        b,h,w,_ = tf.unstack(tf.shape(y_pred))
        flat = tf.reshape(y_pred, [b, h*w])
        idx = self.points[:,0]*w + self.points[:,1]
        vals = tf.gather(flat, idx, axis=1)
        u = tf.gather(vals, self.edge_u, axis=1)
        v = tf.gather(vals, self.edge_v, axis=1)
        tv_mesh = tf.reduce_mean(tf.abs(u - v))

        return dice_loss + self.lambda_del * tv_mesh
