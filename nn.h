/* RPROP Neural Networks implementation
 * See: http://deeplearning.cs.cmu.edu/pdfs/Rprop.pdf
 *
 * Copyright (c) 2003-2016, Salvatore Sanfilippo <antirez at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Disque nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __NN_H
#define __NN_H

//#include <assert.h>

typedef float ann_float_t;
typedef ann_float_t (*AnnDerivativeFunc)(ann_float_t v);
/* Data structures.
 * Nets are not so 'dynamic', but enough to support
 * an arbitrary number of layers, with arbitrary units for layer.
 * Only fully connected feed-forward networks are supported. */
typedef struct {
	ann_float_t *output;		/* output[i], output of i-th unit */
	ann_float_t *error;		/* error[i], output error of i-th unit*/
	ann_float_t *weight;		/* weight[(i*units)+j] */
				/* weight between unit i-th and next j-th */
	ann_float_t *gradient;	/* gradient[(i*units)+j] gradient */
	ann_float_t *sgradient;	/* gradient for the full training set */
				/* only used for RPROP */
	ann_float_t *pgradient;	/* pastgradient[(i*units)+j] t-1 gradient */
				/* (t-1 sgradient for resilient BP) */
	ann_float_t *delta;		/* delta[(i*units)+j] cumulative update */
				/* (per-weight delta for RPROP) */
	int units;	/*moved to last position for alignment purposes*/
	int units_aligned; /*units rounded up for alignment*/
} AnnLayer;

/* Feed forward network structure */
typedef struct {
	AnnLayer *layer;
	int flags;
	int layers;
	AnnDerivativeFunc node_transf_func;
	AnnDerivativeFunc derivative_func;
	ann_float_t rprop_nminus;
	ann_float_t rprop_nplus;
	ann_float_t rprop_maxupdate;
	ann_float_t rprop_minupdate;
	ann_float_t learn_rate; /* Used for GD training. */
} AnnRprop;

typedef ann_float_t (*AnnTrainAlgoFunc)(AnnRprop *net, ann_float_t *input, ann_float_t *desired, int setlen);

/* Raw interface to data structures */
#define ANN_LAYERS(net) (net)->layers
#define ANN_LAYER(net, l) (net)->layer[/*assert(l >= 0),*/l]
#define ANN_OUTPUT(net,l,i) ANN_LAYER(net, l).output[i]
#define ANN_ERROR(net,l,i) ANN_LAYER(net, l).error[i]
#define ANN_LAYER_IDX(net,l,i,j) (((j)*ANN_LAYER(net, l).units_aligned)+(i))
#define ANN_WEIGHT(net,l,i,j) ANN_LAYER(net, l).weight[ANN_LAYER_IDX(net,l,i,j)]
#define ANN_GRADIENT(net,l,i,j) ANN_LAYER(net, l).gradient[ANN_LAYER_IDX(net,l,i,j)]
#define ANN_SGRADIENT(net,l,i,j) ANN_LAYER(net, l).sgradient[ANN_LAYER_IDX(net,l,i,j)]
#define ANN_PGRADIENT(net,l,i,j) ANN_LAYER(net, l).pgradient[ANN_LAYER_IDX(net,l,i,j)]
#define ANN_DELTA(net,l,i,j) ANN_LAYER(net, l).delta[ANN_LAYER_IDX(net,l,i,j)]
#define ANN_UNITS(net,l) ANN_LAYER(net, l).units
#define ANN_UNITS_ALLOCATED(net,l) ANN_LAYER(net, l).units_aligned
#define ANN_WEIGHTS(net,l) (ANN_UNITS(net,l)*ANN_UNITS(net,l-1))
#define ANN_OUTPUT_NODE(net,i) ANN_OUTPUT(net,0,i)
#define ANN_INPUT_NODE(net,i) ANN_OUTPUT(net,(ANN_LAYERS(net))-1,i)
#define ANN_OUTPUT_UNITS(net) ANN_UNITS(net,0)
#define ANN_INPUT_UNITS(net) (ANN_UNITS(net,(ANN_LAYERS(net))-1)-1)
#define ANN_RPROP_NMINUS(net) (net)->rprop_nminus
#define ANN_RPROP_NPLUS(net) (net)->rprop_nplus
#define ANN_RPROP_MAXUPDATE(net) (net)->rprop_maxupdate
#define ANN_RPROP_MINUPDATE(net) (net)->rprop_minupdate
#define ANN_LEARN_RATE(net) (net)->learn_rate

/* Constants */
#define ANN_DEFAULT_RPROP_NMINUS 0.5
#define ANN_DEFAULT_RPROP_NPLUS 1.2
#define ANN_DEFAULT_RPROP_MAXUPDATE 50
#define ANN_DEFAULT_RPROP_MINUPDATE 0.000001
#define ANN_RPROP_INITIAL_DELTA 0.1
#define ANN_DEFAULT_LEARN_RATE 0.1
#define ANN_ALGO_BPROP 0
#define ANN_ALGO_GD 1

/* Misc */
#define ANN_MAX(a,b) (((a)>(b))?(a):(b))
#define ANN_MIN(a,b) (((a)<(b))?(a):(b))

/* Prototypes */
ann_float_t AnnTransferFunctionSigmoid(ann_float_t x);
ann_float_t AnnTransferFunctionRelu(ann_float_t x);
ann_float_t AnnTransferFunctionTanh(ann_float_t x);
//ann_float_t AnnDerivativeIdentity(ann_float_t x);
ann_float_t AnnDerivativeSigmoid(ann_float_t x);
ann_float_t AnnDerivativeTanh(ann_float_t x);
ann_float_t AnnDerivativeRelu(ann_float_t x);

void AnnResetLayer(AnnLayer *layer);
AnnRprop *AnnAlloc(int layers);
void AnnFreeLayer(AnnLayer *layer);
void AnnFree(AnnRprop *net);
int AnnInitLayer(AnnRprop *net, int i, int units, int bias);
AnnRprop *AnnCreateNet(int layers, int *units);
AnnRprop *AnnCreateNet2(int iunits, int ounits);
AnnRprop *AnnCreateNet3(int iunits, int hunits, int ounits);
AnnRprop *AnnCreateNet4(int iunits, int hunits, int hunits2, int ounits);
AnnRprop *AnnClone(const AnnRprop* net);
size_t AnnCountWeights(AnnRprop *net);
void AnnSimulate(AnnRprop *net);
void Ann2Tcl(const AnnRprop *net);
void Ann2Js(const AnnRprop *net);
void AnnPrint(const AnnRprop *net);
ann_float_t AnnGlobalError(AnnRprop *net, ann_float_t *desidered);
void AnnSetInput(AnnRprop *net, ann_float_t *input);
ann_float_t AnnSimulateError(AnnRprop *net, ann_float_t *input, ann_float_t *desidered);
void AnnCalculateGradientsTrivial(AnnRprop *net, ann_float_t *desidered);
void AnnCalculateGradients(AnnRprop *net, ann_float_t *desidered);
void AnnSetDeltas(AnnRprop *net, ann_float_t val);
void AnnResetDeltas(AnnRprop *net);
void AnnResetSgradient(AnnRprop *net);
void AnnSetRandomWeights(AnnRprop *net);
void AnnScaleWeights(AnnRprop *net, ann_float_t factor);
void AnnUpdateDeltasGD(AnnRprop *net);
void AnnUpdateDeltasGDM(AnnRprop *net);
void AnnUpdateSgradient(AnnRprop *net);
void AnnAdjustWeights(AnnRprop *net, int setlen);
ann_float_t AnnBatchGDEpoch(AnnRprop *net, ann_float_t *input, ann_float_t *desidered, int setlen);
ann_float_t AnnBatchGDMEpoch(AnnRprop *net, ann_float_t *input, ann_float_t *desidered, int setlen);
void AnnAdjustWeightsResilientBP(AnnRprop *net);
ann_float_t AnnResilientBPEpoch(AnnRprop *net, ann_float_t *input, ann_float_t *desidered, int setlen);
ann_float_t AnnTrainWithAlgoFunc(AnnRprop *net, ann_float_t *input, ann_float_t *desidered, ann_float_t maxerr, int maxepochs, int setlen, AnnTrainAlgoFunc algo_func);
ann_float_t AnnTrain(AnnRprop *net, ann_float_t *input, ann_float_t *desidered, ann_float_t maxerr, int maxepochs, int setlen, int algo);
void AnnTestError(AnnRprop *net, ann_float_t *input, ann_float_t *desired, int setlen, ann_float_t *avgerr, ann_float_t *classerr);

#endif /* __NN_H */
