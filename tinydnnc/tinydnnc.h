
enum DNN_NetworkType {
  DNN_NETWORK_SEQUENTIAL
};

typedef struct DNN_Network {
  enum DNN_NetworkType type;
  void *ptr;
} DNN_Network;

enum DNN_LayerType {
  DNN_LAYER_EADD,
  DNN_LAYER_AVGPOOL,
  DNN_LAYER_AVGUNPOOL,
  DNN_LAYER_BATCHNORM,
  DNN_LAYER_CONCAT,
  DNN_LAYER_CONV,
  DNN_LAYER_DECONV,
  DNN_LAYER_DROPOUT,
  DNN_LAYER_FC,
  DNN_LAYER_INPUT,
  DNN_LAYER_LINEAR,
  DNN_LAYER_LRN,
  DNN_LAYER_MAXPOOL,
  DNN_LAYER_MAXUNPOOL,
  DNN_LAYER_PARTIALCONN,
  DNN_LAYER_POWER,
  DNN_LAYER_QCONV,
  DNN_LAYER_QDECONV,
  DNN_LAYER_QFC,
  DNN_LAYER_SLICE
};

enum DNN_ActivationType {
  DNN_ACTIVATION_NONE,
  DNN_ACTIVATION_IDENTITY,
  DNN_ACTIVATION_SIGMOID,
  DNN_ACTIVATION_RELU,
  DNN_ACTIVATION_LEAKYRELU,
  DNN_ACTIVATION_ELU,
  DNN_ACTIVATION_SOFTMAX,
  DNN_ACTIVATION_TANH,
  DNN_ACTIVATION_TANHP1M2
};

enum DNN_BackendType {
  DNN_BACKEND_TINYDNN,
  DNN_BACKEND_NNPACK,
  DNN_BACKEND_LIBDNN,
  DNN_BACKEND_AVX,
  DNN_BACKEND_OPENCL
};

typedef struct DNN_Layer {
  enum DNN_LayerType type;
  enum DNN_ActivationType acttype;
  void *ptr;
  void (*addfn)(DNN_Network *net, struct DNN_Layer *layer);
  void (*deletefn)(struct DNN_Layer *layer);
} DNN_Layer;

enum DNN_OptimizerType {
  DNN_OPTIMIZER_ADAGRAD,
  DNN_OPTIMIZER_RMSPROP,
  DNN_OPTIMIZER_ADAM,
  DNN_OPTIMIZER_SGD,
  DNN_OPTIMIZER_MOMENTUM
};

typedef struct DNN_Optimizer {
  enum DNN_OptimizerType type;
  void *ptr;
  void (*deletefn)(struct DNN_Optimizer *opt);
} DNN_Optimizer;

enum DNN_LossType {
  DNN_LOSS_MSE,
  DNN_LOSS_ABSOLUTE,
  DNN_LOSS_CROSSENTROPY,
  DNN_LOSS_CROSSENTROPY_MULTICLASS
};

enum DNN_Padding {
  DNN_PADDING_VALID,
  DNN_PADDING_SAME
};

enum DNN_Phase {
  DNN_PHASE_TRAIN,
  DNN_PHASE_TEST
};

enum DNN_Region {
  DNN_REGION_ACROSS,
  DNN_REGION_WITHIN
};

enum DNN_Slice {
  DNN_SLICE_SAMPLES,
  DNN_SLICE_CHANNELS
};



#ifdef __cplusplus
extern "C" {
#endif

  DNN_Network *DNN_SequentialNetwork();

  void DNN_SequentialAdd(DNN_Network *net, DNN_Layer *layer);

  DNN_Network *DNN_NetworkClone(DNN_Network *net);

  void DNN_NetworkDelete(DNN_Network *net);

  void DNN_LayerDelete(DNN_Layer *layer);

  void DNN_OptimizerDelete(DNN_Optimizer *opt);

  DNN_Layer *DNN_ElwiseAddLayer(long num_args,
                                long dim);

  DNN_Layer *DNN_AveragePoolLayer(enum DNN_ActivationType acttype,
                                  long in_width, long in_height,
                                  long in_channels, long pool_size,
                                  long stride);

  DNN_Layer *DNN_AverageUnpoolLayer(enum DNN_ActivationType acttype,
                                    long in_width, long in_height,
                                    long in_channels, long pool_size,
                                    long stride);

  DNN_Layer *DNN_BatchNormalizationLayer(long in_spatial_size,
                                         long in_channels,
                                         float epsilon, // 1e-5
                                         float momentum, // 0.999
                                         enum DNN_Phase phase); // DNN_PHASE_TRAIN
  DNN_Layer *DNN_ConcatLayer(long num_args,
                             long ndim);

  DNN_Layer *DNN_ConvolutionalLayer(enum DNN_ActivationType activation,
                                    long in_width, long in_height,
                                    long window_width, long window_height,
                                    long in_channels, long out_channels,
                                    enum DNN_Padding pad_type,
                                    long has_bias,
                                    long w_stride, long h_stride,
                                    enum DNN_BackendType backend_type);

  DNN_Layer *DNN_DeconvolutionalLayer(enum DNN_ActivationType acttype,
                                      long in_width, long in_height,
                                      long window_width, long window_height,
                                      long in_channels, long out_channels,
                                      enum DNN_Padding padtype,
                                      long has_bias,
                                      long w_stride, long h_stride,
                                      enum DNN_BackendType backend_type);

  DNN_Layer *DNN_FullyConnectedLayer(enum DNN_ActivationType acttype,
                                     long in_dim, long out_dim,
                                     long has_bias,
                                     enum DNN_BackendType backend_type);

  DNN_Layer *DNN_InputLayer(long dim0, long dim1, long dim2);

  DNN_Layer *DNN_LinearLayer(enum DNN_ActivationType acttype,
                             long dim, float scale, float bias);

  DNN_Layer *DNN_LRNLayer(enum DNN_ActivationType acttype,
                          long in_width, long in_height,
                          long local_size, long in_channels,
                          float alpha, float beta,
                          enum DNN_Region region);

  DNN_Layer *DNN_MaxPoolLayer(enum DNN_ActivationType acttype,
                              long in_width, long in_height,
                              long in_channels, long pool_size,
                              long stride,
                              enum DNN_BackendType backend_type);

  DNN_Layer *DNN_PowerLayer(long dim0, long dim1, long dim2, float factor);

  DNN_Layer *DNN_QuantizedConvolutionalLayer(enum DNN_ActivationType acttype,
                                             long in_width, long in_height,
                                             long window_width, long window_height,
                                             long in_channels, long out_channels,
                                             enum DNN_Padding padtype,
                                             long has_bias,
                                             long w_stride, long h_stride,
                                             enum DNN_BackendType backend_type);

  DNN_Layer *DNN_QuantizedDeconvolutionalLayer(enum DNN_ActivationType acttype,
                                               long in_width, long in_height,
                                               long window_width, long window_height,
                                               long in_channels, long out_channels,
                                               enum DNN_Padding padtype,
                                               long has_bias,
                                               long w_stride, long h_stride,
                                               enum DNN_BackendType backend_type);

  DNN_Layer *DNN_QuantizedFullyConnectedLayer(enum DNN_ActivationType acttype,
                                              long in_dim, long out_dim,
                                              long has_bias,
                                              enum DNN_BackendType backend_type);

  DNN_Layer *DNN_SliceLayer(long dim0, long dim1, long dim2,
                            enum DNN_Slice slice_type,
                            long num_outputs);




  DNN_Optimizer *DNN_AdaGradOptimizer(float alpha); // 0.01

  DNN_Optimizer *DNN_RMSPropOptimizer(float alpha, // 0.0001
                                      float mu); // 0.99

  DNN_Optimizer *DNN_AdamOptimizer(float alpha, // 0.001
                                   float b1, // 0.9
                                   float b2, // 0.999
                                   float b1_t, // 0.9
                                   float b2_t); // 0.999

  DNN_Optimizer *DNN_SGDOptimizer(float alpha, // 0.01
                                  float lambda); // 0.0

  DNN_Optimizer *DNN_MomentumOptimizer(float alpha, // 0.01
                                       float lambda, // 0.0
                                       float mu); // 0.9

  void DNN_Train(DNN_Network *net,
                 DNN_Optimizer *opt,
                 enum DNN_LossType losstype,
                 float *inputs,
                 long *outputs,
                 long n_samples,
                 long sample_size,
                 long batch_size,
                 long epochs,
                 void (*batch_cb)(DNN_Network *net, void *data), // on_batch_callback
                 void (*epoch_cb)(DNN_Network *net, void *data), // on_epoch_callback
                 void *cb_data,
                 long reset_weights, // false
                 long n_threads,
                 float *cost); // relative target costs, can be NULL

  void DNN_Fit(DNN_Network *net,
               DNN_Optimizer *opt,
               enum DNN_LossType losstype,
               float *inputs,
               float *outputs,
               long n_samples,
               long sample_size,
               long output_size,
               long batch_size,
               long epochs,
               void (*batch_cb)(DNN_Network *net, void *data), // on_batch_callback
               void (*epoch_cb)(DNN_Network *net, void *data), // on_epoch_callback
               void *cb_data,
               long reset_weights, // false
               long n_threads,
               float *cost); // relative target costs, can be NULL

  void DNN_Predict(DNN_Network *net,
                   float *input,
                   float *output,
                   long input_size,
                   long output_size);

  long DNN_PredictLabel(DNN_Network *net,
                        float *input,
                        long input_size);

  float DNN_GetError(DNN_Network *net,
                     float *inputs, // std::vector<vec_t>&, aka tensor_t&
                     long *outputs, // std::vector<label_t>&
                     long n_samples,
                     long sample_size);

  float DNN_GetError2(DNN_Network *net,
                     float *inputs,
                     float *outputs,
                     long n_samples,
                     long sample_size,
                     long output_size);

  float DNN_GetLoss(DNN_Network *net,
                    enum DNN_LossType losstype,
                    float *inputs, // std::vector<vec_t>&, aka tensor_t&
                    float *outputs, // std::vector<vec_t>&
                    long n_samples,
                    long sample_size,
                    long output_size);

  #ifdef __cplusplus
}
#endif

