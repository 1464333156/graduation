import tensorflow as tf
from tensorflow.keras import layers, Model

def get_model(params, args_max_neigh=None):
    """
    MYPLAN 交通异常检测模型结构
    """
    # 基础参数
    num_region = params.number_region
    len_recent = params.len_recent_time
    d_model = params.dr
    
    # 1. 输入层
    # 动态流量输入 (Recent traffic flow)
    input_flow = layers.Input(shape=(len_recent, num_region, 1), name='flow_input')
    # 邻域上下文输入 (Neighbor context)
    input_nc = layers.Input(shape=(len_recent, num_region, 1), name='nc_input')
    # 动态特征输入 (Dynamic features)
    input_dy = layers.Input(shape=(len_recent, num_region, 2 * d_model), name='dy_input')

    # 2. 时空特征提取 (简化示意版，匹配参数结构)
    # 实际模型中这里会有复杂的 Attention 和 Evolution Smooth 逻辑
    x_flow = layers.Reshape((len_recent, num_region))(input_flow)
    x_nc = layers.Reshape((len_recent, num_region))(input_nc)
    
    # 融合层
    combined = layers.Concatenate()([x_flow, x_nc])
    
    # 时间维度处理 (LSTM/GRU)
    lstm_out = layers.LSTM(d_model, return_sequences=True)(combined)
    
    # 3. 输出层 - 预测每个区域在最后一个时间步的异常概率
    # 最终输出形状应为 (num_region, 1)
    # 注意：为了适配 app.py 中的 y_pred.numpy()[-1]，这里输出序列
    prediction = layers.Dense(1, activation='sigmoid')(lstm_out)
    
    # 构建模型
    model = Model(inputs=[input_flow, input_nc, input_dy], outputs=[prediction, None, None])
    
    return model