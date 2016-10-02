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

/* Data structures.
 * Nets are not so 'dynamic', but enough to support
 * an arbitrary number of layers, with arbitrary units for layer.
 * Only fully connected feed-forward networks are supported. */
struct AnnLayer {
	int units;
	float *output;		/* output[i], output of i-th unit */
	float *error;		/* error[i], output error of i-th unit*/
	float *weight;		/* weight[(i*units)+j] */
				/* weight between unit i-th and next j-th */
	float *gradient;	/* gradient[(i*units)+j] gradient */
	float *sgradient;	/* gradient for the full training set */
				/* only used for RPROP */
	float *pgradient;	/* pastgradient[(i*units)+j] t-1 gradient */
				/* (t-1 sgradient for resilient BP) */
	float *delta;		/* delta[(i*units)+j] cumulative update */
				/* (per-weight delta for RPROP) */
};

/* Feed forward network structure */
struct Ann {
	int flags;
	int layers;
	float rprop_nminus;
	float rprop_nplus;
	float rprop_maxupdate;
	float rprop_minupdate;
	struct AnnLayer *layer;
};

/* Raw interface to data structures */
#define OUTPUT(net,l,i) (net)->layer[l].output[i]
#define ERROR(net,l,i) (net)->layer[l].error[i]
#define WEIGHT(net,l,i,j) (net)->layer[l].weight[((j)*(net)->layer[l].units)+(i)]
#define GRADIENT(net,l,i,j) (net)->layer[l].gradient[((j)*(net)->layer[l].units)+(i)]
#define SGRADIENT(net,l,i,j) (net)->layer[l].sgradient[((j)*(net)->layer[l].units)+(i)]
#define PGRADIENT(net,l,i,j) (net)->layer[l].pgradient[((j)*(net)->layer[l].units)+(i)]
#define DELTA(net,l,i,j) (net)->layer[l].delta[((j)*(net)->layer[l].units)+(i)]
#define LAYERS(net) (net)->layers
#define UNITS(net,l) (net)->layer[l].units
#define WEIGHTS(net,l) (UNITS(net,l)*UNITS(net,l-1))
#define OUTPUT_NODE(net,i) OUTPUT(net,0,i)
#define INPUT_NODE(net,i) OUTPUT(net,((net)->layers)-1,i)
#define OUTPUT_UNITS(net) UNITS(net,0)
#define INPUT_UNITS(net) (UNITS(net,((net)->layers)-1)-(LAYERS(net)>2))
#define RPROP_NMINUS(net) (net)->rprop_nminus
#define RPROP_NPLUS(net) (net)->rprop_nplus
#define RPROP_MAXUPDATE(net) (net)->rprop_maxupdate
#define RPROP_MINUPDATE(net) (net)->rprop_minupdate

/* Constants */
#define DEFAULT_RPROP_NMINUS 0.5
#define DEFAULT_RPROP_NPLUS 1.2
#define DEFAULT_RPROP_MAXUPDATE 50
#define DEFAULT_RPROP_MINUPDATE 0.000001
#define RPROP_INITIAL_DELTA 0.1

/* Misc */
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

/* Prototypes */
void AnnResetLayer(struct AnnLayer *layer);
struct Ann *AnnAlloc(int layers);
void AnnFreeLayer(struct AnnLayer *layer);
void AnnFree(struct Ann *net);
int AnnInitLayer(struct Ann *net, int i, int units, int bias);
struct Ann *AnnCreateNet(int layers, int *units);
struct Ann *AnnCreateNet2(int iunits, int ounits);
struct Ann *AnnCreateNet3(int iunits, int hunits, int ounits);
struct Ann *AnnCreateNet4(int iunits, int hunits, int hunits2, int ounits);
struct Ann *AnnClone(struct Ann* net);
size_t AnnCountWeights(struct Ann *net);
void AnnSimulate(struct Ann *net);
void Ann2Tcl(struct Ann *net);
void AnnPrint(struct Ann *net);
float AnnGlobalError(struct Ann *net, float *desidered);
void AnnSetInput(struct Ann *net, float *input);
float AnnSimulateError(struct Ann *net, float *input, float *desidered);
void AnnCalculateGradientsTrivial(struct Ann *net, float *desidered);
void AnnCalculateGradients(struct Ann *net, float *desidered);
void AnnSetDeltas(struct Ann *net, float val);
void AnnResetDeltas(struct Ann *net);
void AnnResetSgradient(struct Ann *net);
void AnnSetRandomWeights(struct Ann *net);
void AnnScaleWeights(struct Ann *net, float factor);
void AnnUpdateDeltasGD(struct Ann *net);
void AnnUpdateDeltasGDM(struct Ann *net);
void AnnUpdateSgradient(struct Ann *net);
void AnnAdjustWeights(struct Ann *net);
float AnnBatchGDEpoch(struct Ann *net, float *input, float *desidered, int setlen);
float AnnBatchGDMEpoch(struct Ann *net, float *input, float *desidered, int setlen);
void AnnAdjustWeightsResilientBP(struct Ann *net);
float AnnResilientBPEpoch(struct Ann *net, float *input, float *desidered, int setlen);
float AnnTrain(struct Ann *net, float *input, float *desidered, float maxerr, int maxepochs, int setlen);
float AnnTestError(struct Ann *net, float *input, float *desidered, int setlen);
float AnnTestClassError(struct Ann *net, float *input, float *desidered, int setlen);

#endif /* __NN_H */
