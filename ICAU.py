import tensorflow as tf
import keras

class InpaintContextAttentionUnit(keras.layers.Layer):
    def __init__(self, n=8, fil=16, **kwargs):
        super(InpaintContextAttentionUnit,self).__init__(**kwargs)
        self.n = n
        self.fil = fil
        #filters need to be set as the input channels dim.

    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(filters=self.fil,kernel_size=[3,3],padding="valid",activation="relu")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n':self.n,
            'fil': self.fil})
        return config

    def call(self, feature_map, training=None):
        def inpaint_rows(inputs):
            """
            The function for mimicking inpainting and then stacking t,he individual rows.
            """
            input_b = inputs.shape[0]
            input_h = inputs.shape[1]
            input_w = inputs.shape[2]
            input_c = inputs.shape[3]
        
            inpainted_tensors = []
            #adding padding to the tensor on all four sides.
            paddings = [[0,0],[1,1],[1,1],[0,0]]
            feature_map_padded = tf.pad(inputs,paddings=paddings,mode="CONSTANT")
            
            for row in range(feature_map_padded.shape[1]-2):
                tensor_top = feature_map_padded[:,row:row+1,:,:]
                tensor_mid = tf.zeros(shape=[tf.shape(inputs)[0],1, input_w+2, input_c])
                tensor_bottom = feature_map_padded[:,row+2:row+3,:,:]
                sub_tensor = tf.concat([tensor_top,tensor_mid,tensor_bottom],axis=1)
                sub_tensorCONV = self.conv2d(sub_tensor)
                inpainted_tensors.append(sub_tensorCONV)
            
            res = tf.concat(inpainted_tensors,axis=1)
            return res
        
        def inpaint_cols(inputs):
            transposed_input = tf.transpose(inputs,[0,2,1,3])
            inpainted_transposed_input = inpaint_rows(transposed_input)
            return tf.transpose(inpainted_transposed_input,[0,2,1,3])    

        h = feature_map.shape[1]
        w = feature_map.shape[2]
        n = self.n

        #Average Pooling the Input Feature.
        out = keras.layers.AveragePooling2D(pool_size=(h/n,2), strides=(h/n,2),padding="SAME")(feature_map)
 
        #Applying Conv-Inpainting on the input features.
        row_op = inpaint_rows(out)
        col_op = inpaint_cols(out)

        #Upsampling the inpainted feature map
        row_op_upsampled = tf.image.resize(row_op,[h, w])
        col_op_upsampled = tf.image.resize(col_op,[h, w])
        
        #Stacking the original Feature map and the Inpainted feature maps respectively.
        stacked_op = tf.concat([row_op_upsampled,col_op_upsampled],axis=3)        
        stacked_orig = tf.concat([feature_map, feature_map], axis=3)       

        #Subtracting the inpainted features from the original feature map.
        diff_feature = tf.math.subtract(stacked_orig,stacked_op)
        res = tf.concat([feature_map,diff_feature],axis=3)
        return res

#Testing the subclassed layer.
#inputs = tf.keras.Input(shape=(8,16,32),batch_size=4)
#outputs = InpaintContextAttentionUnit(fil=inputs.shape[3],n=2)(inputs)
#outputs = tf.keras.layers.Conv2D(filters = outputs.shape[3]/2,kernel_size=1, activation='relu', padding='SAME',data_format='channels_last')(outputs)
#model = tf.keras.Model(inputs=inputs, outputs=outputs, name='test')
#model.compile(loss=tf.keras.losses.MeanSquaredError())
#model.summary()

