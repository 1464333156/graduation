import tensorflow as tf
from tensorflow.keras import layers, Model

class MYPLAN(Model):
    def __init__(self, dr, len_recent_time, number_sp, number_region, 
                 neigh_poi_index, neigh_road_index, neigh_record_index,
                 attention_mode='scaled_dot', evolution_smooth=True, **kwargs):
        super(MYPLAN, self).__init__(**kwargs)
        self.dr = dr
        self.number_sp = number_sp
        
        # 基础嵌入层
        self.flow_emb = layers.Dense(dr)
        self.nc_emb = layers.Dense(dr)
        
        # 模拟时空特征提取层
        self.lstm = layers.LSTM(dr, return_sequences=True)
        self.attention_dense = layers.Dense(dr, activation='relu')
        
        # 进化平滑层 (Evolution Smooth)
        self.evolution_smooth = evolution_smooth
        if evolution_smooth:
            self.gate = layers.Dense(dr, activation='sigmoid')
            
        # 输出层
        self.out = layers.Dense(1, activation='sigmoid')

    def call(self, inputs_data, inputs_nc, inputs_dy):
        # inputs_data: (batch, time, region, feat)
        # inputs_nc: (batch, time, region, 1)
        
        x_flow = self.flow_emb(inputs_data)
        x_nc = self.nc_emb(inputs_nc)
        
        # 特征融合
        combined = x_flow + x_nc
        
        # 转换维度以适应 LSTM (batch * region, time, dr)
        batch_size = tf.shape(combined)[0]
        time_steps = tf.shape(combined)[1]
        num_region = tf.shape(combined)[2]
        
        x = tf.transpose(combined, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [-1, time_steps, self.dr])
        
        x = self.lstm(x)
        
        # 还原维度 (batch, region, time, dr) -> (batch, time, region, dr)
        x = tf.reshape(x, [batch_size, num_region, time_steps, self.dr])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        
        # 输出预测
        prediction = self.out(x)
        
        # 返回 app.py 预期的 3 个输出 (pred, att_score, smooth_score)
        return prediction, None, None

class BaselineRNN(Model):
    def __init__(self, dr, len_recent_time, number_region, rnn_type='lstm', **kwargs):
        super(BaselineRNN, self).__init__(**kwargs)
        self.dr = dr
        if rnn_type == 'lstm':
            self.rnn = layers.LSTM(dr, return_sequences=True)
        else:
            self.rnn = layers.GRU(dr, return_sequences=True)
        self.out = layers.Dense(1, activation='sigmoid')

    def call(self, inputs_data, inputs_nc, inputs_dy):
        x = layers.Concatenate()([inputs_data, inputs_nc])
        # 维度转换同上
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        num_region = tf.shape(x)[2]
        
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [-1, time_steps, x.shape[-1]])
        x = self.rnn(x)
        x = tf.reshape(x, [batch_size, num_region, time_steps, self.dr])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        
        prediction = self.out(x)
        return prediction, None, None

class BaselineMLP(Model):
    def __init__(self, dr, len_recent_time, number_region, **kwargs):
        super(BaselineMLP, self).__init__(**kwargs)
        self.dr = dr
        self.dense1 = layers.Dense(dr, activation='relu')
        self.dense2 = layers.Dense(dr, activation='relu')
        self.out = layers.Dense(1, activation='sigmoid')

    def call(self, inputs_data, inputs_nc, inputs_dy):
        x = layers.Concatenate()([inputs_data, inputs_nc])
        x = self.dense1(x)
        x = self.dense2(x)
        prediction = self.out(x)
        return prediction, None, None
