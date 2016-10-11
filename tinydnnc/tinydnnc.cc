
#include "tinydnnc.h"

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

// Network is assumed to always be network<sequential>
// Extension to graph is planned

#define DNN_NewLayerMacro(LAYER,LAYERNAME,LAYERCLASS,...) \
  LAYER->ptr = new LAYERCLASS(__VA_ARGS__);               \
  LAYER->type = LAYERNAME;                                \
  LAYER->acttype = DNN_ACTIVATION_NONE;                   \
  LAYER->addfn = DNN_Add_##LAYERNAME;                     \
  LAYER->deletefn = DNN_Delete_##LAYERNAME;

#define DNN_NewFFLayerMacro(LAYER,LAYERNAME,LAYERCLASS,ACTTYPE,...) \
  switch (ACTTYPE) {                                                \
  case DNN_ACTIVATION_NONE:                                         \
    assert(0);                                                      \
    break;                                                          \
  case DNN_ACTIVATION_IDENTITY:                                     \
    LAYER->ptr = new LAYERCLASS<identity>(__VA_ARGS__);             \
    break;                                                          \
  case DNN_ACTIVATION_SIGMOID:                                      \
    LAYER->ptr = new LAYERCLASS<sigmoid>(__VA_ARGS__);              \
    break;                                                          \
  case DNN_ACTIVATION_RELU:                                         \
    LAYER->ptr = new LAYERCLASS<relu>(__VA_ARGS__);                 \
    break;                                                          \
  case DNN_ACTIVATION_LEAKYRELU:                                    \
    LAYER->ptr = new LAYERCLASS<leaky_relu>(__VA_ARGS__);           \
    break;                                                          \
  case DNN_ACTIVATION_ELU:                                          \
    LAYER->ptr = new LAYERCLASS<elu>(__VA_ARGS__);                  \
    break;                                                          \
  case DNN_ACTIVATION_SOFTMAX:                                      \
    LAYER->ptr = new LAYERCLASS<softmax>(__VA_ARGS__);              \
    break;                                                          \
  case DNN_ACTIVATION_TANH:                                         \
    LAYER->ptr = new LAYERCLASS<tan_h>(__VA_ARGS__);                \
    break;                                                          \
  case DNN_ACTIVATION_TANHP1M2:                                     \
    LAYER->ptr = new LAYERCLASS<tan_hp1m2>(__VA_ARGS__);            \
    break;                                                          \
  }                                                                 \
  LAYER->type = LAYERNAME;                                          \
  LAYER->acttype = acttype;                                         \
  LAYER->addfn = DNN_Add_##LAYERNAME;                               \
  LAYER->deletefn = DNN_Delete_##LAYERNAME;

#define DNN_AddLayerMacro(NETWORK,LAYER,LAYERCLASS)                     \
  *((network<sequential> *)NETWORK->ptr) << *((LAYERCLASS *)LAYER->ptr);

#define DNN_AddFFLayerMacro(NETWORK,LAYER,LAYERCLASS) \
  switch (LAYER->acttype) {                           \
  case DNN_ACTIVATION_NONE:                           \
    assert(0);                                        \
    break;                                            \
  case DNN_ACTIVATION_IDENTITY:                       \
    *((network<sequential> *)NETWORK->ptr) <<         \
      *((LAYERCLASS<identity> *)LAYER->ptr);          \
    break;                                            \
  case DNN_ACTIVATION_SIGMOID:                        \
    *((network<sequential> *)NETWORK->ptr) <<         \
      *((LAYERCLASS<sigmoid> *)LAYER->ptr);           \
    break;                                            \
  case DNN_ACTIVATION_RELU:                           \
    *((network<sequential> *)NETWORK->ptr) <<         \
      *((LAYERCLASS<relu> *)LAYER->ptr);              \
    break;                                            \
  case DNN_ACTIVATION_LEAKYRELU:                      \
    *((network<sequential> *)NETWORK->ptr) <<         \
      *((LAYERCLASS<leaky_relu> *)LAYER->ptr);        \
    break;                                            \
  case DNN_ACTIVATION_ELU:                            \
    *((network<sequential> *)NETWORK->ptr) <<         \
      *((LAYERCLASS<elu> *)LAYER->ptr);               \
    break;                                            \
  case DNN_ACTIVATION_SOFTMAX:                        \
    *((network<sequential> *)NETWORK->ptr) <<         \
      *((LAYERCLASS<softmax> *)LAYER->ptr);           \
    break;                                            \
  case DNN_ACTIVATION_TANH:                           \
    *((network<sequential> *)NETWORK->ptr) <<         \
      *((LAYERCLASS<tan_h> *)LAYER->ptr);             \
    break;                                            \
  case DNN_ACTIVATION_TANHP1M2:                       \
    *((network<sequential> *)NETWORK->ptr) <<         \
      *((LAYERCLASS<tan_hp1m2> *)LAYER->ptr);         \
    break;                                            \
  }

#define DNN_DeleteLayerMacro(LAYER,LAYERCLASS)  \
  delete (LAYERCLASS *)LAYER->ptr;

#define DNN_DeleteFFLayerMacro(LAYER,LAYERCLASS)  \
  switch (LAYER->acttype) {                       \
  case DNN_ACTIVATION_NONE:                       \
    assert(0);                                    \
    break;                                        \
  case DNN_ACTIVATION_IDENTITY:                   \
    delete (LAYERCLASS<identity> *)LAYER->ptr;    \
    break;                                        \
  case DNN_ACTIVATION_SIGMOID:                    \
    delete (LAYERCLASS<sigmoid> *)LAYER->ptr;     \
    break;                                        \
  case DNN_ACTIVATION_RELU:                       \
    delete (LAYERCLASS<relu> *)LAYER->ptr;        \
    break;                                        \
  case DNN_ACTIVATION_LEAKYRELU:                  \
    delete (LAYERCLASS<leaky_relu> *)LAYER->ptr;  \
    break;                                        \
  case DNN_ACTIVATION_ELU:                        \
    delete (LAYERCLASS<elu> *)LAYER->ptr;         \
    break;                                        \
  case DNN_ACTIVATION_SOFTMAX:                    \
    delete (LAYERCLASS<softmax> *)LAYER->ptr;     \
    break;                                        \
  case DNN_ACTIVATION_TANH:                       \
    delete (LAYERCLASS<tan_h> *)LAYER->ptr;       \
    break;                                        \
  case DNN_ACTIVATION_TANHP1M2:                   \
    delete (LAYERCLASS<tan_hp1m2> *)LAYER->ptr;   \
    break;                                        \
  }

#define DNN_DefineLayerMacro(LAYERNAME,LAYERCLASS)              \
  void DNN_Add_##LAYERNAME(DNN_Network *net, DNN_Layer *layer)  \
  {                                                             \
    DNN_AddLayerMacro(net,layer,LAYERCLASS);                    \
  }                                                             \
  void DNN_Delete_##LAYERNAME(DNN_Layer *layer)                 \
  {                                                             \
    DNN_DeleteLayerMacro(layer,LAYERCLASS);                     \
  }

#define DNN_DefineFFLayerMacro(LAYERNAME,LAYERCLASS)            \
  void DNN_Add_##LAYERNAME(DNN_Network *net, DNN_Layer *layer)  \
  {                                                             \
    DNN_AddFFLayerMacro(net,layer,LAYERCLASS);                  \
  }                                                             \
  void DNN_Delete_##LAYERNAME(DNN_Layer *layer)                 \
  {                                                             \
    DNN_DeleteFFLayerMacro(layer,LAYERCLASS);                   \
  }

#define DNN_DefineOptimizerMacro(OPTNAME,OPTCLASS)  \
  void DNN_Delete_##OPTNAME(DNN_Optimizer *opt)     \
  {                                                 \
    delete (OPTCLASS *)opt->ptr;                    \
  }

#define DNN_SetupOptimizerMacro(OPT,OPTNAME,OPTPTR) \
  OPT->type = OPTNAME;                              \
  OPT->ptr = OPTPTR;                                \
  OPT->deletefn = DNN_Delete_##OPTNAME;

#define DNN_SetupOptimizerMacro(OPT,OPTNAME,OPTPTR) \
  OPT->type = OPTNAME;                              \
  OPT->ptr = OPTPTR;                                \
  OPT->deletefn = DNN_Delete_##OPTNAME;

#define DNN_SetBackendMacro(BACKEND,BACKENDTYPE)  \
  switch (BACKENDTYPE) {                          \
  case DNN_BACKEND_TINYDNN:                       \
    BACKEND = backend_t::tiny_dnn;                \
    break;                                        \
  case DNN_BACKEND_NNPACK:                        \
    BACKEND = backend_t::nnpack;                  \
    break;                                        \
  case DNN_BACKEND_LIBDNN:                        \
    BACKEND = backend_t::libdnn;                  \
    break;                                        \
  case DNN_BACKEND_AVX:                           \
    BACKEND = backend_t::avx;                     \
    break;                                        \
  case DNN_BACKEND_OPENCL:                        \
    BACKEND = backend_t::opencl;                  \
    break;                                        \
  }

void DNN_LayerDelete(DNN_Layer *layer)
{
  (*layer->deletefn)(layer);
  delete layer;
}

DNN_Network *DNN_SequentialNetwork()
{
  DNN_Network *net = new DNN_Network;
  net->type = DNN_NETWORK_SEQUENTIAL;
  net->ptr = new network<sequential>;
  return net;
}

void DNN_SequentialAdd(DNN_Network *net, DNN_Layer *layer)
{
  layer->addfn(net,layer);
}

DNN_Network *DNN_NetworkClone(DNN_Network *fromnet)
{
  DNN_Network *net = new DNN_Network;
  net->type = DNN_NETWORK_SEQUENTIAL;
  net->ptr = new network<sequential>;
  *(network<sequential> *)net->ptr = *(network<sequential> *)fromnet->ptr;
  return net;
}

void DNN_NetworkDelete(DNN_Network *net)
{
  delete (network<sequential> *)net->ptr;
  delete net;
}

void DNN_OptimizerDelete(DNN_Optimizer *opt)
{
  opt->deletefn(opt);
  delete opt;
}

//////////////////////////////////////////////
// Layers
//////////////////////////////////////////////

DNN_DefineLayerMacro(DNN_LAYER_EADD,elementwise_add_layer);
DNN_Layer *DNN_ElwiseAddLayer(long num_args,
                              long dim)
{
  DNN_Layer *layer = new DNN_Layer;

  DNN_NewLayerMacro(layer, DNN_LAYER_EADD, elementwise_add_layer,
                    num_args, dim);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_AVGPOOL,average_pooling_layer);
DNN_Layer *DNN_AveragePoolLayer(enum DNN_ActivationType acttype,
                                long in_width, long in_height,
                                long in_channels, long pool_size,
                                long stride)
{
  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_AVGPOOL, average_pooling_layer,
                      acttype,
                      in_width, in_height, in_channels, pool_size, stride);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_AVGUNPOOL,average_unpooling_layer);
DNN_Layer *DNN_AverageUnpoolLayer(enum DNN_ActivationType acttype,
                                  long in_width, long in_height,
                                  long in_channels, long pool_size,
                                  long stride)
{
  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_AVGUNPOOL, average_unpooling_layer,
                      acttype,
                      in_width, in_height, in_channels, pool_size, stride);

  return layer;
}

DNN_DefineLayerMacro(DNN_LAYER_BATCHNORM,batch_normalization_layer);
DNN_Layer *DNN_BatchNormalizationLayer(long in_spatial_size,
                                       long in_channels,
                                       float epsilon, // 1e-5
                                       float momentum, // 0.999
                                       enum DNN_Phase phase) // DNN_PHASE_TRAIN
{
  DNN_Layer *layer = new DNN_Layer;

  net_phase nphase;
  switch (phase) {
  case DNN_PHASE_TRAIN:
    nphase = net_phase::train;
    break;
  case DNN_PHASE_TEST:
    nphase = net_phase::test;
    break;
  }

  DNN_NewLayerMacro(layer, DNN_LAYER_BATCHNORM, batch_normalization_layer,
                    in_spatial_size, in_channels, epsilon, momentum,
                    nphase);

  return layer;
}

DNN_DefineLayerMacro(DNN_LAYER_CONCAT,concat_layer);
DNN_Layer *DNN_ConcatLayer(long num_args,
                           long ndim)
{
  DNN_Layer *layer = new DNN_Layer;

  DNN_NewLayerMacro(layer, DNN_LAYER_CONCAT, concat_layer,
                    num_args, ndim);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_CONV,convolutional_layer);
DNN_Layer *DNN_ConvolutionalLayer(enum DNN_ActivationType acttype,
                                  long in_width, long in_height,
                                  long window_width, long window_height,
                                  long in_channels, long out_channels,
                                  enum DNN_Padding padtype,
                                  long has_bias,
                                  long w_stride, long h_stride,
                                  enum DNN_BackendType backend_type)
{
  enum padding pad;
  switch (padtype) {
  case DNN_PADDING_SAME:
    pad = padding::same;
    break;
  case DNN_PADDING_VALID:
    pad = padding::valid;
    break;
  }

  enum backend_t backend;
  DNN_SetBackendMacro(backend,backend_type);

  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_CONV, convolutional_layer,
                      acttype,
                      in_width, in_height,
                      window_width, window_height,
                      in_channels, out_channels,
                      pad, has_bias,
                      w_stride, h_stride,
                      backend);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_DECONV,deconvolutional_layer);
DNN_Layer *DNN_DeconvolutionalLayer(enum DNN_ActivationType acttype,
                                    long in_width, long in_height,
                                    long window_width, long window_height,
                                    long in_channels, long out_channels,
                                    enum DNN_Padding padtype,
                                    long has_bias,
                                    long w_stride, long h_stride,
                                    enum DNN_BackendType backend_type)
{
  enum padding pad;
  switch (padtype) {
  case DNN_PADDING_SAME:
    pad = padding::same;
    break;
  case DNN_PADDING_VALID:
    pad = padding::valid;
    break;
  }

  enum backend_t backend;
  DNN_SetBackendMacro(backend,backend_type);

  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_DECONV, deconvolutional_layer,
                      acttype,
                      in_width, in_height,
                      window_width, window_height,
                      in_channels, out_channels,
                      pad, has_bias,
                      w_stride, h_stride,
                      backend);

  return layer;
}

DNN_DefineLayerMacro(DNN_LAYER_DROPOUT,dropout_layer);
DNN_Layer *DNN_ConcatLayer(long in_dim,
                           float dropout_rate,
                           enum DNN_Phase phase)
{
  DNN_Layer *layer = new DNN_Layer;

  net_phase nphase;
  switch (phase) {
  case DNN_PHASE_TRAIN:
    nphase = net_phase::train;
    break;
  case DNN_PHASE_TEST:
    nphase = net_phase::test;
    break;
  }

  DNN_NewLayerMacro(layer, DNN_LAYER_DROPOUT, dropout_layer,
                    in_dim, dropout_rate, nphase);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_FC,fully_connected_layer);
DNN_Layer *DNN_FullyConnectedLayer(enum DNN_ActivationType acttype,
                                   long in_dim, long out_dim,
                                   long has_bias,
                                   enum DNN_BackendType backend_type)
{
  enum backend_t backend;
  DNN_SetBackendMacro(backend,backend_type);

  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_FC, fully_connected_layer,
                      acttype,
                      in_dim, out_dim, has_bias,
                      backend);

  return layer;
}

DNN_DefineLayerMacro(DNN_LAYER_INPUT,input_layer);
DNN_Layer *DNN_InputLayer(long dim0, long dim1, long dim2)
{
  DNN_Layer *layer = new DNN_Layer;

  DNN_NewLayerMacro(layer, DNN_LAYER_INPUT, input_layer,
                    shape3d(dim0,dim1,dim2));

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_LINEAR,linear_layer);
DNN_Layer *DNN_LinearLayer(enum DNN_ActivationType acttype,
                           long dim, float scale, float bias)
{
  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_LINEAR, linear_layer,
                      acttype,
                      dim, scale, bias);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_LRN,lrn_layer);
DNN_Layer *DNN_LRNLayer(enum DNN_ActivationType acttype,
                        long in_width, long in_height,
                        long local_size, long in_channels,
                        float alpha, float beta,
                        enum DNN_Region region)
{
  DNN_Layer *layer = new DNN_Layer;

  norm_region nregion;
  switch (region) {
  case DNN_REGION_ACROSS:
    nregion = norm_region::across_channels;
    break;
  case DNN_REGION_WITHIN:
    nregion = norm_region::within_channels;
    break;
  }

  DNN_NewFFLayerMacro(layer, DNN_LAYER_LRN, lrn_layer,
                      acttype,
                      in_width, in_height,
                      local_size, in_channels,
                      alpha, beta, nregion);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_MAXPOOL,max_pooling_layer);
DNN_Layer *DNN_MaxPoolLayer(enum DNN_ActivationType acttype,
                            long in_width, long in_height,
                            long in_channels, long pool_size,
                            long stride,
                            enum DNN_BackendType backend_type)
{
  enum backend_t backend;
  DNN_SetBackendMacro(backend,backend_type);

  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_MAXPOOL, max_pooling_layer,
                      acttype,
                      in_width, in_height, in_channels,
                      pool_size, stride,
                      backend);

  return layer;
}

// FIXME COMPILER ERROR
//DNN_DefineFFLayerMacro(DNN_LAYER_MAXUNPOOL,max_unpooling_layer);
//DNN_Layer *DNN_MaxUnpoolLayer(enum DNN_ActivationType acttype,
//                              long in_width, long in_height,
//                              long in_channels, long pool_size,
//                              long stride)
//{
//  DNN_Layer *layer = new DNN_Layer;
//
//  DNN_NewFFLayerMacro(layer, DNN_LAYER_MAXUNPOOL, max_unpooling_layer,
//                      acttype,
//                      in_width, in_height, in_channels,
//                      pool_size, stride);
//
//  return layer;
//}

// TODO partial_connected_layer (requires connecting manually)

DNN_DefineLayerMacro(DNN_LAYER_POWER,power_layer);
DNN_Layer *DNN_PowerLayer(long dim0, long dim1, long dim2, float factor)
{
  DNN_Layer *layer = new DNN_Layer;

  DNN_NewLayerMacro(layer, DNN_LAYER_POWER, power_layer,
                    shape3d(dim0,dim1,dim2), factor);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_QCONV,quantized_convolutional_layer);
DNN_Layer *DNN_QuantizedConvolutionalLayer(enum DNN_ActivationType acttype,
                                           long in_width, long in_height,
                                           long window_width, long window_height,
                                           long in_channels, long out_channels,
                                           enum DNN_Padding padtype,
                                           long has_bias,
                                           long w_stride, long h_stride,
                                           enum DNN_BackendType backend_type)
{
  enum padding pad;
  switch (padtype) {
  case DNN_PADDING_SAME:
    pad = padding::same;
    break;
  case DNN_PADDING_VALID:
    pad = padding::valid;
    break;
  }

  enum backend_t backend;
  DNN_SetBackendMacro(backend,backend_type);

  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_QCONV, quantized_convolutional_layer,
                      acttype,
                      in_width, in_height,
                      window_width, window_height,
                      in_channels, out_channels,
                      pad, has_bias,
                      w_stride, h_stride,
                      backend);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_QDECONV,quantized_deconvolutional_layer);
DNN_Layer *DNN_QuantizedDeconvolutionalLayer(enum DNN_ActivationType acttype,
                                             long in_width, long in_height,
                                             long window_width, long window_height,
                                             long in_channels, long out_channels,
                                             enum DNN_Padding padtype,
                                             long has_bias,
                                             long w_stride, long h_stride,
                                             enum DNN_BackendType backend_type)
{
  enum padding pad;
  switch (padtype) {
  case DNN_PADDING_SAME:
    pad = padding::same;
    break;
  case DNN_PADDING_VALID:
    pad = padding::valid;
    break;
  }

  enum backend_t backend;
  DNN_SetBackendMacro(backend,backend_type);

  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_QDECONV, quantized_deconvolutional_layer,
                      acttype,
                      in_width, in_height,
                      window_width, window_height,
                      in_channels, out_channels,
                      pad, has_bias,
                      w_stride, h_stride,
                      backend);

  return layer;
}

DNN_DefineFFLayerMacro(DNN_LAYER_QFC,quantized_fully_connected_layer);
DNN_Layer *DNN_QuantizedFullyConnectedLayer(enum DNN_ActivationType acttype,
                                            long in_dim, long out_dim,
                                            long has_bias,
                                            enum DNN_BackendType backend_type)
{
  enum backend_t backend;
  DNN_SetBackendMacro(backend,backend_type);

  DNN_Layer *layer = new DNN_Layer;

  DNN_NewFFLayerMacro(layer, DNN_LAYER_QFC, quantized_fully_connected_layer,
                      acttype,
                      in_dim, out_dim, has_bias,
                      backend);

  return layer;
}

DNN_DefineLayerMacro(DNN_LAYER_SLICE,slice_layer);
DNN_Layer *DNN_SliceLayer(long dim0, long dim1, long dim2,
                          enum DNN_Slice slice_type,
                          long num_outputs)
{
  DNN_Layer *layer = new DNN_Layer;

  enum slice_type stype;
  switch (slice_type) {
  case DNN_SLICE_SAMPLES:
    stype = slice_type::slice_samples;
    break;
  case DNN_SLICE_CHANNELS:
    stype = slice_type::slice_channels;
    break;
  }

  DNN_NewLayerMacro(layer, DNN_LAYER_SLICE, slice_layer,
                    shape3d(dim0,dim1,dim2), stype,
                    num_outputs);

  return layer;
}



//////////////////////////////////////////////
// Optimizers
//////////////////////////////////////////////

DNN_DefineOptimizerMacro(DNN_OPTIMIZER_ADAGRAD,adagrad);
DNN_Optimizer *DNN_AdaGradOptimizer(float alpha) // 0.01
{
  adagrad *optptr = new adagrad;
  optptr->alpha = alpha;

  DNN_Optimizer *opt = new DNN_Optimizer;
  DNN_SetupOptimizerMacro(opt,DNN_OPTIMIZER_ADAGRAD,optptr);

  return opt;
}

DNN_DefineOptimizerMacro(DNN_OPTIMIZER_RMSPROP,RMSprop);
DNN_Optimizer *DNN_RMSPropOptimizer(float alpha, // 0.0001
                                    float mu) // 0.99
{
  RMSprop *optptr = new RMSprop;
  optptr->alpha = alpha;
  optptr->mu = mu;

  DNN_Optimizer *opt = new DNN_Optimizer;
  DNN_SetupOptimizerMacro(opt,DNN_OPTIMIZER_RMSPROP,optptr);

  return opt;
}

DNN_DefineOptimizerMacro(DNN_OPTIMIZER_ADAM,adam);
DNN_Optimizer *DNN_AdamOptimizer(float alpha, // 0.001
                                 float b1, // 0.9
                                 float b2, // 0.999
                                 float b1_t, // 0.9
                                 float b2_t) // 0.999
{
  adam *optptr = new adam;
  optptr->alpha = alpha;
  optptr->b1 = b1;
  optptr->b2 = b2;
  optptr->b1_t = b1_t;
  optptr->b2_t = b2_t;

  DNN_Optimizer *opt = new DNN_Optimizer;
  DNN_SetupOptimizerMacro(opt,DNN_OPTIMIZER_ADAM,optptr);

  return opt;
}

DNN_DefineOptimizerMacro(DNN_OPTIMIZER_SGD,gradient_descent);
DNN_Optimizer *DNN_SGDOptimizer(float alpha, // 0.01
                               float lambda) // 0.0
{
  gradient_descent *optptr = new gradient_descent;
  optptr->alpha = alpha;
  optptr->lambda = lambda;

  DNN_Optimizer *opt = new DNN_Optimizer;
  DNN_SetupOptimizerMacro(opt,DNN_OPTIMIZER_SGD,optptr);

  return opt;
}

DNN_DefineOptimizerMacro(DNN_OPTIMIZER_MOMENTUM,momentum);
DNN_Optimizer *DNN_MomentumOptimizer(float alpha, // 0.01
                                     float lambda, // 0.0
                                     float mu) // 0.9
{
  momentum *optptr = new momentum;
  optptr->alpha = alpha;
  optptr->lambda = lambda;
  optptr->mu = mu;

  DNN_Optimizer *opt = new DNN_Optimizer;
  DNN_SetupOptimizerMacro(opt,DNN_OPTIMIZER_MOMENTUM,optptr);

  return opt;
}


//////////////////////////////////////////////
// Training
//////////////////////////////////////////////

#define DNN_TrainCallMacro(NET,METHOD,LOSSCLASS,OPT,OPTCLASS,...)         \
  (*(network<sequential> *)NET->ptr).METHOD<LOSSCLASS>(*(OPTCLASS *)OPT->ptr, __VA_ARGS__);

#define DNN_DispatchTrainOnLoss(NET,METHOD,LOSSTYPE,OPT,OPTCLASS,...)     \
  switch (LOSSTYPE) {                                                   \
  case DNN_LOSS_MSE:                                                    \
    DNN_TrainCallMacro(NET,METHOD,mse,OPT,OPTCLASS,__VA_ARGS__);          \
    break;                                                              \
  case DNN_LOSS_ABSOLUTE:                                               \
    DNN_TrainCallMacro(NET,METHOD,absolute,OPT,OPTCLASS,__VA_ARGS__);     \
    break;                                                              \
  case DNN_LOSS_CROSSENTROPY:                                           \
    DNN_TrainCallMacro(NET,METHOD,cross_entropy,OPT,OPTCLASS,__VA_ARGS__); \
    break;                                                              \
  case DNN_LOSS_CROSSENTROPY_MULTICLASS:                                \
    DNN_TrainCallMacro(NET,METHOD,cross_entropy_multiclass,OPT,OPTCLASS,__VA_ARGS__); \
    break;                                                              \
  }

#define DNN_DispatchTrain(NET,METHOD,LOSSTYPE,OPT,...)                    \
  switch(OPT->type) {                                                   \
  case DNN_OPTIMIZER_ADAGRAD:                                           \
    DNN_DispatchTrainOnLoss(NET,METHOD,LOSSTYPE,OPT,adagrad,__VA_ARGS__); \
    break;                                                              \
  case DNN_OPTIMIZER_RMSPROP:                                           \
    DNN_DispatchTrainOnLoss(NET,METHOD,LOSSTYPE,OPT,RMSprop,__VA_ARGS__); \
    break;                                                              \
  case DNN_OPTIMIZER_ADAM:                                              \
    DNN_DispatchTrainOnLoss(NET,METHOD,LOSSTYPE,OPT,adam,__VA_ARGS__);    \
    break;                                                              \
  case DNN_OPTIMIZER_SGD:                                               \
    DNN_DispatchTrainOnLoss(NET,METHOD,LOSSTYPE,OPT,gradient_descent,__VA_ARGS__); \
    break;                                                              \
  case DNN_OPTIMIZER_MOMENTUM:                                          \
    DNN_DispatchTrainOnLoss(NET,METHOD,LOSSTYPE,OPT,momentum,__VA_ARGS__); \
    break;                                                              \
  }

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
               float *cost) // relative target costs, can be NULL
{

  std::vector<vec_t> dnn_inputs(n_samples);
  std::vector<label_t> dnn_outputs(n_samples);
  std::vector<vec_t> dnn_cost;

  if (cost) {
    dnn_cost.resize(n_samples);
  }

  for (int n=0; n<n_samples; n++) {
    dnn_inputs[n].assign(inputs+n*sample_size,
                         inputs+(n+1)*sample_size);
    dnn_outputs[n] = outputs[n];
    if (cost) {
      dnn_cost[n].assign(cost+n, cost+n+1);
    }
  }

  DNN_DispatchTrain(net, train, losstype, opt,
                    dnn_inputs, dnn_outputs,
                    batch_size, epochs,
                    [&](){ if (batch_cb == NULL) return;
                      batch_cb(net, cb_data);
                      network<sequential> &dnn_net = *(network<sequential> *)net->ptr;
                      dnn_net.set_netphase(net_phase::train);
                    },
                    [&](){ if (epoch_cb == NULL) return;
                      epoch_cb(net, cb_data);
                      network<sequential> &dnn_net = *(network<sequential> *)net->ptr;
                      dnn_net.set_netphase(net_phase::train);
                    },
                    reset_weights, n_threads,
                    dnn_cost);

}

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
             float *cost) // relative target costs, can be NULL
{
  std::vector<vec_t> dnn_inputs(n_samples);
  std::vector<vec_t> dnn_outputs(n_samples);
  std::vector<vec_t> dnn_cost;

  if (cost) {
    dnn_cost.resize(n_samples);
  }

  for (int n=0; n<n_samples; n++) {
    dnn_inputs[n].assign(inputs+n*sample_size,
                         inputs+(n+1)*sample_size);
    dnn_outputs[n].assign(outputs+n*output_size,
                          outputs+(n+1)*output_size);
    if (cost) {
      dnn_cost[n].assign(cost+n*output_size,
                         cost+(n+1)*output_size);
    }
  }

  DNN_DispatchTrain(net, fit, losstype, opt,
                    dnn_inputs, dnn_outputs,
                    batch_size, epochs,
                    [&](){ if (batch_cb == NULL) return;
                      network<sequential> &dnn_net = *(network<sequential> *)net->ptr;
                      batch_cb(net, cb_data);
                      dnn_net.set_netphase(net_phase::train);
                    },
                    [&](){ if (epoch_cb == NULL) return;
                      network<sequential> &dnn_net = *(network<sequential> *)net->ptr;
                      epoch_cb(net, cb_data);
                      dnn_net.set_netphase(net_phase::train);
                    },
                    reset_weights, n_threads,
                    dnn_cost);
}

void DNN_Predict(DNN_Network *net,
                 float *input,
                 float *output,
                 long input_size,
                 long output_size)
{
  vec_t dnn_input;
  dnn_input.assign(input, input+input_size);

  network<sequential>& dnn_net = (*(network<sequential> *)net->ptr);
  dnn_net.set_netphase(net_phase::test);
  vec_t dnn_output = dnn_net.predict(dnn_input);

  for (int i=0; i<output_size; i++) {
    output[i] = dnn_output[i];
  }
}

long DNN_PredictLabel(DNN_Network *net,
                      float *input,
                      long input_size)
{
  vec_t dnn_input;
  dnn_input.assign(input, input+input_size);

  network<sequential>& dnn_net = (*(network<sequential> *)net->ptr);
  dnn_net.set_netphase(net_phase::test);
  label_t label = dnn_net.predict_label(dnn_input);

  return label;
}

float DNN_GetError(DNN_Network *net,
                   float *inputs,
                   long *outputs,
                   long n_samples,
                   long sample_size)
{
  std::vector<vec_t> dnn_inputs(n_samples);
  std::vector<label_t> dnn_outputs(n_samples);

  // FIXME: copying every time is a waste.
  // This issue applies to Train, Fit, GetError and GetLoss.
  // The issue is that apparently the only safe way to
  // initialize a std::vector from a C array is through
  // a copy, because the std::vector can't know what the
  // allocator was.
  // The alternative here is to introduce a DNN_Data struct
  // that holds an opaque pointer to a std::vector allocated
  // on the C++ side. Passing this struct to GetLoss, GetError,
  // Train and Fit would avoid the copy at every call.
  for (int n=0; n<n_samples; n++) {
    dnn_inputs[n].assign(inputs+n*sample_size,
                         inputs+(n+1)*sample_size);
    dnn_outputs[n] = outputs[n];
  }

  network<sequential> &dnn_net = *(network<sequential> *)net->ptr;

  result res = dnn_net.test(dnn_inputs,dnn_outputs);

  float error = 1.0 - 0.01 * res.accuracy();

  return error;
}

/* Like GetError() but instead of getting labels it gets the raw
 * outputs that are translated to labels according to the max
 * value of the output. */
float DNN_GetError2(DNN_Network *net,
                   float *inputs,
                   float *outputs,
                   long n_samples,
                   long sample_size,
                   long output_size)
{
  std::vector<vec_t> dnn_inputs(n_samples);
  std::vector<label_t> dnn_outputs(n_samples);

  for (int n=0; n<n_samples; n++) {
    dnn_inputs[n].assign(inputs+n*sample_size,
                         inputs+(n+1)*sample_size);
    int label = 0;
    float label_max = outputs[0];
    for (int i=1; i<output_size; i++) {
        if (outputs[i] > label_max) {
            label = i;
            label_max = outputs[i];
        }
    }
    dnn_outputs[n] = label;
    outputs += output_size;
  }

  network<sequential> &dnn_net = *(network<sequential> *)net->ptr;

  result res = dnn_net.test(dnn_inputs,dnn_outputs);

  float error = 1.0 - 0.01 * res.accuracy();

  return error;
}

float DNN_GetLoss(DNN_Network *net,
                  enum DNN_LossType losstype,
                  float *inputs,
                  float *outputs,
                  long n_samples,
                  long sample_size,
                  long output_size)
{
  std::vector<vec_t> dnn_inputs(n_samples);
  std::vector<vec_t> dnn_outputs(n_samples);

  // FIXME: copying every time is a waste. See above.
  for (int n=0; n<n_samples; n++) {
    dnn_inputs[n].assign(inputs+n*sample_size,
                         inputs+(n+1)*sample_size);
    dnn_outputs[n].assign(outputs+n*output_size,
                          outputs+(n+1)*output_size);
  }

  network<sequential> &dnn_net = *(network<sequential> *)net->ptr;

  float loss_value;

  switch (losstype) {
  case DNN_LOSS_MSE:
    loss_value = dnn_net.get_loss<mse>(dnn_inputs,dnn_outputs);
    break;
  case DNN_LOSS_ABSOLUTE:
    loss_value = dnn_net.get_loss<absolute>(dnn_inputs,dnn_outputs);
    break;
  case DNN_LOSS_CROSSENTROPY:
    loss_value = dnn_net.get_loss<cross_entropy>(dnn_inputs,dnn_outputs);
    break;
  case DNN_LOSS_CROSSENTROPY_MULTICLASS:
    loss_value = dnn_net.get_loss<cross_entropy_multiclass>(dnn_inputs,dnn_outputs);
    break;
  }

  return loss_value;
}

