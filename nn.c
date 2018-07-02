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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "nn.h"

/*
There is a problem with memory alignment when using avx and avx512
on some machines avx aligned works fine on others don't
*/
#if defined(USE_AVX512)
#define USING_SIMD
#include <immintrin.h>

typedef __m512 simdf_t;
#define  SIMDF_SIZE 16

#define simdf_zero() _mm512_setzero_ps()
#define simdf_set1f(x) _mm512_set1_ps(x)
#define simdf_loadu(x) _mm512_loadu_ps(x)
#define simdf_load(x) _mm512_loadu_ps(x) //we are still using unaligned here
#define simdf_mul(a,b) _mm512_mul_ps(a,b)
#define simdf_add(a,b) _mm512_add_ps(a,b)
#define simdf_storeu(a,b) _mm512_storeu_ps(a,b)
#define simdf_store(a,b) _mm512_storeu_ps(a,b) //we are still using unaligned here

//let the compiler optmize this
#define simdf_sum(x) (x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + \
                                x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15])

#define simdf_show(x) printf("%d : %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", \
                                __LINE__, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], \
                                x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
#endif

#if defined(USE_AVX)
#define USING_SIMD
#include <immintrin.h>

typedef __m256 simdf_t;
#define  SIMDF_SIZE 8

#define simdf_zero() _mm256_setzero_ps()
#define simdf_set1f(x) _mm256_set1_ps(x)
#define simdf_loadu(x) _mm256_loadu_ps(x)
#define simdf_load(x) _mm256_loadu_ps(x) //we are still using unaligned here
#define simdf_mul(a,b) _mm256_mul_ps(a,b)
#define simdf_add(a,b) _mm256_add_ps(a,b)
#define simdf_storeu(a,b) _mm256_storeu_ps(a,b)
#define simdf_store(a,b) _mm256_storeu_ps(a,b) //we are still using unaligned here

//let the compiler optmize this
#define simdf_sum(x) (x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7])

#define simdf_show(x) printf("%d : %f, %f, %f, %f, %f, %f, %f, %f\n", \
                                __LINE__, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
#endif

#if defined(USE_SSE)
#define USING_SIMD
#include <xmmintrin.h>
#include <pmmintrin.h>
typedef __m128 simdf_t;
#define  SIMDF_SIZE 4

#define simdf_zero() _mm_setzero_ps()
#define simdf_set1f(x) _mm_set1_ps(x)
#define simdf_loadu(x) _mm_loadu_ps(x)
#define simdf_load(x) _mm_load_ps(x)
#define simdf_mul(a,b) _mm_mul_ps(a,b)
#define simdf_add(a,b) _mm_add_ps(a,b)
#define simdf_storeu(a,b) _mm_storeu_ps(a,b)
#define simdf_store(a,b) _mm_store_ps(a,b)

//let the compiler optmize this
#define simdf_sum(x) (x[0] + x[1] + x[2] + x[3])

#define simdf_show(x) printf("%d : %f, %f, %f, %f\n", __LINE__, x[0], x[1], x[2], x[3]);
#endif

#if defined(USE_NEON)
#define USING_SIMD
#include <arm_neon.h>

typedef ann_float_t32x4_t simdf_t;
#define  SIMDF_SIZE 4

#define simdf_zero() vdupq_n_f32(0.0f)
#define simdf_set1f(x) vdupq_n_f32(x);
#define simdf_loadu(x) vld1q_f32(x)
#define simdf_load(x) vld1q_f32(x)
#define simdf_mul(a,b) vmulq_f32(a,b)
#define simdf_add(a,b) vaddq_f32(a,b)
#define simdf_storeu(a,b) vst1q_f32((ann_float_t32_t*)a,b)
#define simdf_store(a,b) vst1q_f32((ann_float_t32_t*)a,b)

//let the compiler optmize this
#define simdf_sum(x) (x[0] + x[1] + x[2] + x[3])

#define simdf_show(x) printf("%d : %f, %f, %f, %f\n", __LINE__, x[0], x[1], x[2], x[3]);
#endif

#ifndef SIMDF_SIZE
#define SIMDF_SIZE 1
#endif // SIMDF_SIZE

#define ANN_SIZEOF_ann_float_t sizeof(ann_float_t)
#define ANN_ALIGN_BASE (SIMDF_SIZE * ANN_SIZEOF_ann_float_t)
#define ANN_ALIGN_ROUND(x) ((x%ANN_ALIGN_BASE) ? (((x/ANN_ALIGN_BASE)+1)*ANN_ALIGN_BASE) : (size_t)x)

#ifndef HAS_ANN_MALLOC
#define ann_malloc(x) malloc(x)
#define ann_free(x) free(x)
#else
extern void *ann_malloc(size_t sz);
extern void ann_free(void *ptr);
#endif
/*
void *nnpmalloc(int line, size_t sz) {
	printf("%d : %zu : %zu\n", line, sz, ANN_ALIGN_ROUND(sz));
	return malloc(sz);
}
#define ann_malloc(x) nnpmalloc(__LINE__, x)
*/

/* Node Transfer Function */
ann_float_t AnnTransferFunctionSigmoid(ann_float_t x) {
    return ((ann_float_t)1)/(1+exp(-x));
}

ann_float_t AnnTransferFunctionRelu(ann_float_t x) {
    return (x > 0) ? x : 0;
}

ann_float_t AnnTransferFunctionTanh(ann_float_t x) {
    return tanh(x);
}

/*
ann_float_t AnnDerivativeIdentity(ann_float_t x) {
    return 1;
}
*/

ann_float_t AnnDerivativeSigmoid(ann_float_t x) {
    return x*(1-x);
}

ann_float_t AnnDerivativeTanh(ann_float_t x) {
    return (1-x)*(1+x);
}

ann_float_t AnnDerivativeRelu(ann_float_t x) {
    return (x > 0) ? 1 : 0;
}

/* Reset layer data to zero-units */
void AnnResetLayer(AnnLayer *layer) {
    layer->units = 0;
    layer->units_aligned = 0;
    layer->output = NULL;
    layer->error = NULL;
    layer->weight = NULL;
    layer->gradient = NULL;
    layer->pgradient = NULL;
    layer->delta = NULL;
    layer->sgradient = NULL;
}

/* Allocate and return an initialized N-layers network */
AnnRprop *AnnAlloc(int layers) {
    AnnRprop *net;
    int i;

    /* Alloc the net structure */
    if ((net = ann_malloc(sizeof(*net))) == NULL)
        return NULL;
    /* Alloc layers */
    if ((net->layer = ann_malloc(sizeof(AnnLayer)*layers)) == NULL) {
        ann_free(net);
        return NULL;
    }
    net->layers = layers;
    net->flags = 0;
    net->rprop_nminus = ANN_DEFAULT_RPROP_NMINUS;
    net->rprop_nplus = ANN_DEFAULT_RPROP_NPLUS;
    net->rprop_maxupdate = ANN_DEFAULT_RPROP_MAXUPDATE;
    net->rprop_minupdate = ANN_DEFAULT_RPROP_MINUPDATE;
    net->node_transf_func = AnnTransferFunctionSigmoid;
    net->derivative_func = AnnDerivativeSigmoid;
    /* Init layers */
    for (i = 0; i < layers; i++)
        AnnResetLayer(&net->layer[i]);
    return net;
}

/* Free a single layer */
void AnnFreeLayer(AnnLayer *layer)
{
    ann_free(layer->output);
    ann_free(layer->error);
    ann_free(layer->weight);
    ann_free(layer->gradient);
    ann_free(layer->pgradient);
    ann_free(layer->delta);
    ann_free(layer->sgradient);
    AnnResetLayer(layer);
}

/* Free the target net */
void AnnFree(AnnRprop *net)
{
    int i;

    /* Free layer data */
    for (i = 0; i < net->layers; i++) AnnFreeLayer(&net->layer[i]);
    /* Free allocated layers structures */
    ann_free(net->layer);
    /* And the main structure itself */
    ann_free(net);
}

/* Init a layer of the net with the specified number of units.
 * Return non-zero on out of memory. */
int AnnInitLayer(AnnRprop *net, int i, int units, int bias) {
    if (bias) units++; /* Take count of the bias unit */
    int ann_float_t_units = ANN_ALIGN_ROUND(units*ANN_SIZEOF_ann_float_t);
    int units_aligned = ann_float_t_units/ANN_SIZEOF_ann_float_t;
    int ann_float_t_units_units = 0;
    AnnLayer *layer = &ANN_LAYER(net, i);
    layer->units = units;
    layer->units_aligned = units_aligned;
    layer->output = ann_malloc(ann_float_t_units);
    layer->error = ann_malloc(ann_float_t_units);
    if (i) { /* not for output layer */
        ann_float_t_units_units = ann_float_t_units*ANN_LAYER(net, i-1).units;
        layer->weight = ann_malloc(ann_float_t_units_units);
        layer->gradient = ann_malloc(ann_float_t_units_units);
        layer->pgradient = ann_malloc(ann_float_t_units_units);
        layer->delta = ann_malloc(ann_float_t_units_units);
        layer->sgradient = ann_malloc(ann_float_t_units_units);
    }
    /* Check for out of memory conditions */
    if (layer->output == NULL ||
        layer->error == NULL ||
        (i && layer->weight == NULL) ||
        (i && layer->gradient == NULL) ||
        (i && layer->pgradient == NULL) ||
        (i && layer->sgradient == NULL) ||
        (i && layer->delta == NULL))
    {
        AnnFreeLayer(layer);
        AnnResetLayer(layer);
        return 1;
    }
    /* Set all the values to zero */
    memset(layer->output, 0, ann_float_t_units);
    memset(layer->error, 0, ann_float_t_units);
    if (i) {
        memset(layer->weight, 0, ann_float_t_units_units);
        memset(layer->gradient, 0, ann_float_t_units_units);
        memset(layer->pgradient, 0, ann_float_t_units_units);
        memset(layer->delta, 0, ann_float_t_units_units);
        memset(layer->sgradient, 0, ann_float_t_units_units);
    }
    /* Set the bias unit output to 1 */
    if (bias) layer->output[units-1] = 1;
    return 0;
}

/* Clone a network. On out of memory NULL is returned. */
AnnRprop *AnnClone(const AnnRprop* net) {
    AnnRprop* copy;
    int j;

    if ((copy = AnnAlloc(ANN_LAYERS(net))) == NULL) return NULL;
    for (j = 0; j < ANN_LAYERS(net); j++) {
        AnnLayer *ldst;
        const AnnLayer *lsrc;
        int units = ANN_UNITS(net,j);
        int bias = j > 0;
        if (AnnInitLayer(copy, j, units-bias, bias)) {
            AnnFree(copy);
            return NULL;
        }
        int ann_float_t_units = units*ANN_SIZEOF_ann_float_t;
        lsrc = &net->layer[j];
        ldst = &copy->layer[j];
        if (lsrc->output)
            memcpy(ldst->output, lsrc->output, ann_float_t_units);
        if (lsrc->error)
            memcpy(ldst->error, lsrc->error, ann_float_t_units);
        if (j) {
            int weights = ANN_WEIGHTS(net,j);
            ann_float_t_units = weights*ANN_SIZEOF_ann_float_t;
            if (lsrc->weight)
                memcpy(ldst->weight, lsrc->weight, ann_float_t_units);
            if (lsrc->gradient)
                memcpy(ldst->gradient, lsrc->gradient, ann_float_t_units);
            if (lsrc->pgradient)
                memcpy(ldst->pgradient, lsrc->pgradient, ann_float_t_units);
            if (lsrc->delta)
                memcpy(ldst->delta, lsrc->delta, ann_float_t_units);
            if (lsrc->sgradient)
                memcpy(ldst->sgradient, lsrc->sgradient, ann_float_t_units);
        }
    }
    copy->rprop_nminus = net->rprop_nminus;
    copy->rprop_nplus = net->rprop_nplus;
    copy->rprop_maxupdate = net->rprop_maxupdate;
    copy->rprop_minupdate = net->rprop_minupdate;
    copy->flags = net->flags;
    copy->node_transf_func = net->node_transf_func;
    copy->derivative_func = net->derivative_func;
    return copy;
}

/* Create a N-layer input/hidden/output net.
 * The units array should specify the number of
 * units in every layer from the output to the input layer. */
AnnRprop *AnnCreateNet(int layers, int *units) {
    AnnRprop *net;
    int i;

    if ((net = AnnAlloc(layers)) == NULL) return NULL;
    for (i = 0; i < layers; i++) {
        if (AnnInitLayer(net, i, units[i], i > 0)) {
            AnnFree(net);
            return NULL;
        }
    }
    AnnSetRandomWeights(net);
    AnnSetDeltas(net, ANN_RPROP_INITIAL_DELTA);
    ANN_LEARN_RATE(net) = ANN_DEFAULT_LEARN_RATE;
    return net;
}

/* Return the total number of weights this NN has. */
size_t AnnCountWeights(AnnRprop *net) {
    size_t weights = 0;
    for (int i = ANN_LAYERS(net)-1; i > 0; i--) {
        int nextunits = ANN_UNITS(net, i-1);
        int units = ANN_UNITS(net, i);
        if (i > 1) nextunits--; /* we don't output on bias units */
        weights += units*nextunits;
    }
    return weights;
}

/* Create a 4-layer input/hidden/output net */
AnnRprop *AnnCreateNet4(int iunits, int hunits, int hunits2, int ounits) {
    int units[4];

    units[0] = ounits;
    units[1] = hunits2;
    units[2] = hunits;
    units[3] = iunits;
    return AnnCreateNet(4, units);
}

/* Create a 3-layer input/hidden/output net */
AnnRprop *AnnCreateNet3(int iunits, int hunits, int ounits) {
    int units[3];

    units[0] = ounits;
    units[1] = hunits;
    units[2] = iunits;
    return AnnCreateNet(3, units);
}


/* Create a 2-layer "linear" network. */
AnnRprop *AnnCreateNet2(int iunits, int ounits) {
    int units[2];

    units[0] = ounits;
    units[1] = iunits;
    return AnnCreateNet(2, units);
}


void AnnSimulate(AnnRprop *net) {
    int i, j, k;

    for (i = ANN_LAYERS(net)-1; i > 0; i--) {
        AnnLayer *layer = &ANN_LAYER(net, i);
        int nextunits = ANN_UNITS(net, i-1);
        int units_aligned = layer->units_aligned;
        int units = layer->units;
        if (i > 1) nextunits--; /* dont output on bias units */
#ifdef USING_SIMD
        int xps, psteps = units/SIMDF_SIZE;
#endif // USING_SIMD
        for (j = 0; j < nextunits; j++) {
            ann_float_t A = 0; /* Activation final value. */
            ann_float_t *w = layer->weight + j*units_aligned;
            ann_float_t *o = layer->output;

            k = 0;

#ifdef USING_SIMD
            if(psteps)
            {
                simdf_t sumA = simdf_zero();
                for (xps = 0; xps < psteps; xps++) {
                    simdf_t weights = simdf_load(w);
                    simdf_t outputs = simdf_load(o);
                    simdf_t prod = simdf_mul(weights,outputs);
                    sumA = simdf_add(sumA, prod);
                    w += SIMDF_SIZE;
                    o += SIMDF_SIZE;
                }
                A += simdf_sum(sumA);
                k += psteps*SIMDF_SIZE;
            }
#endif

            /* Handle final piece shorter than SIMDF_SIZE . */
            for (; k < units; k++) {
                A += (*w++) * (*o++);
            }
            //ANN_OUTPUT(net, i-1, j) = (*net->node_transf_func)(A); //sigmoid(A);
	    	ANN_OUTPUT(net, i-1, j) = 1.0/(1.0+exp(-A));
        }
    }
}

/* Create a Tcl procedure that simulates the neural network */
void Ann2Tcl(const AnnRprop *net) {
    int i, j, k;

    printf("proc ann input {\n");
    printf("    set output {");
    for (i = 0; i < ANN_OUTPUT_UNITS(net); i++) {
        printf("0 ");
    }
    printf("}\n");
    printf("    proc sigmoid x {return [expr {1/(1+exp(-$x))}]}\n");
    for(i=0, k=ANN_INPUT_UNITS(net); i < k; ++i) {
      printf("    set input_%d [lindex $input %d]\n", i, i);
    }
    for (i = ANN_LAYERS(net)-1; i > 0; i--) {
        int nextunits = ANN_UNITS(net, i-1);
        int units = ANN_UNITS(net, i);
        //if (i > 1) nextunits--; /* dont output on bias units */
        for (j = 0; j < nextunits; j++) {
            ann_float_t W;
            if (i == 1) {
                printf("    lset output %d ", j);
            } else {
                printf("    set O_%d_%d", i-1, j);
            }
            printf(" [sigmoid [expr { \\\n");
            for (k = 0; k < units; k++) {
                W = ANN_WEIGHT(net, i, k, j);
                if (i > 1 && k == units-1) {
                    printf("        (%.9f)", W);
                } else if (i == ANN_LAYERS(net)-1) {
                    printf("        (%.9f*$input_%d)", W, k);
                } else {
                    printf("        (%.9f*$O_%d_%d)", W, i, k);
                }
                if ((k+1) < units) printf("+ \\\n");
            }
            printf("}]]\n");
        }
    }
    printf("    return $output\n");
    printf("}\n");
}

/* Create a Javascript procedure that simulates the neural network */
void Ann2Js(const AnnRprop *net) {
    int i, j, k;

    printf("function ann( input ) {\n");
    printf("    var output = [");
    for (i = 0; i < ANN_OUTPUT_UNITS(net); i++) {
	if(i) printf(", ");
        printf("0");
    }
    printf("];\n");
    printf("    var sigmoid = function(x) {return 1.0/(1.0+Math.exp(-x));};\n");
    for(i=0, k=ANN_INPUT_UNITS(net); i < k; ++i) {
      printf("    var input_%d = input[%d];\n", i, i);
    }
    for (i = ANN_LAYERS(net)-1; i > 0; i--) {
        int nextunits = ANN_UNITS(net, i-1);
        int units = ANN_UNITS(net, i);
        //if (i > 1) nextunits--; /* dont output on bias units */
        for (j = 0; j < nextunits; j++) {
            ann_float_t W;
            if (i == 1) {
                printf("    output[%d]", j);
            } else {
                printf("    var O_%d_%d", i-1, j);
            }
            printf(" = sigmoid(\n");
            for (k = 0; k < units; k++) {
                W = ANN_WEIGHT(net, i, k, j);
                if (i > 1 && k == units-1) {
                    printf("        (%.9f)", W);
                } else if (i == ANN_LAYERS(net)-1) {
                    printf("        (%.9f*input_%d)", W, k);
                } else {
                    printf("        (%.9f*O_%d_%d)", W, i, k);
                }
                if ((k+1) < units) printf("+\n");
            }
            printf(");\n");
        }
    }
    printf("    return output;\n");
    printf("}\n");
}

/* Print a network representation */
void AnnPrint(const AnnRprop *net) {
    int i, j, k;

    for (i = 0; i < ANN_LAYERS(net); i++) {
        char *layertype = "Hidden";
        if (i == 0) layertype = "Output";
        if (i == ANN_LAYERS(net)-1) layertype = "Input";
        printf("%s layer %d, units %d\n", layertype, i, ANN_UNITS(net,i));
        if (i) {
            /* Don't compute the bias unit as a target. */
            int targets = ANN_UNITS(net,i-1) - (i-1>0);
            /* Weights */
            printf("\tW");
            for (j = 0; j < ANN_UNITS(net, i); j++) {
                printf("(");
                for (k = 0; k < targets; k++) {
                    printf("%f", ANN_WEIGHT(net,i,j,k));
                    if (k != targets-1) printf(" ");
                }
                printf(") ");
            }
            printf("\n");
            /* Gradients */
            printf("\tg");
            for (j = 0; j < ANN_UNITS(net, i); j++) {
                printf("[");
                for (k = 0; k < targets; k++) {
                    printf("%f", ANN_GRADIENT(net,i,j,k));
                    if (k != targets-1) printf(" ");
                }
                printf("] ");
            }
            printf("\n");
            /* SGradients */
            printf("\tG");
            for (j = 0; j < ANN_UNITS(net, i); j++) {
                printf("[");
                for (k = 0; k < targets; k++) {
                    printf("%f", ANN_SGRADIENT(net,i,j,k));
                    if (k != targets-1) printf(" ");
                }
                printf("] ");
            }
            printf("\n");
            /* Gradients at t-1 */
            printf("\tP");
            for (j = 0; j < ANN_UNITS(net, i); j++) {
                printf("[");
                for (k = 0; k < targets; k++) {
                    printf("%f", ANN_PGRADIENT(net,i,j,k));
                    if (k != targets-1) printf(" ");
                }
                printf("] ");
            }
            printf("\n");
            /* Delta */
            printf("\tD");
            for (j = 0; j < ANN_UNITS(net, i); j++) {
                printf("|");
                for (k = 0; k < targets; k++) {
                    printf("%f", ANN_DELTA(net,i,j,k));
                    if (k != targets-1) printf(" ");
                }
                printf("| ");
            }
            printf("\n");
        }
        for (j = 0; j < ANN_UNITS(net,i); j++) {
            printf("\tO: %f ", ANN_OUTPUT(net,i,j));
        }
        printf("\n");
        printf("\tE /");
        for (j = 0; j < ANN_UNITS(net,i); j++) {
            printf("%f ", ANN_ERROR(net,i,j));
        }
        printf("/\n");
    }
}

/* Calcuate the global error of the net. This is just the
 * Root Mean Square (RMS) error, which is half the sum of the squared
 * errors. */
ann_float_t AnnGlobalError(AnnRprop *net, ann_float_t *desired) {
    ann_float_t e, t;
    int i, outputs = ANN_OUTPUT_UNITS(net);

    e = 0;
    for (i = 0; i < outputs; i++) {
        t = desired[i] - ANN_OUTPUT_NODE(net,i);
        e += t*t; /* No need for fabs(t), t*t will always be positive. */
    }
    return .5*e;
}

/* Set the network input */
void AnnSetInput(AnnRprop *net, ann_float_t *input)
{
    int i, inputs = ANN_INPUT_UNITS(net);

    for (i = 0; i < inputs; i++) ANN_INPUT_NODE(net,i) = input[i];
}

/* Simulate the net, and return the global error */
ann_float_t AnnSimulateError(AnnRprop *net, ann_float_t *input, ann_float_t *desired) {
    AnnSetInput(net, input);
    AnnSimulate(net);
    return AnnGlobalError(net, desired);
}

/* Compute the error vector y-t in the output unit. This error depends
 * on the loss function we use. */
void AnnCalculateOutputError(AnnRprop *net, ann_float_t *desired) {
    int units = ANN_OUTPUT_UNITS(net);
    ann_float_t factor = (ann_float_t)2/units;
    AnnLayer *layer = &ANN_LAYER(net, 0);
    for (int j = 0; j < units; j++) {
        layer->error[j] = factor * (layer->output[j] - desired[j]);
    }
}

/* Calculate gradients with a trivial and slow algorithm, this
 * is useful to check that the real implementation is working
 * well, comparing the results.
 *
 * The algorithm used is: to compute the error function in two
 * points (E1, with the real weight, and E2 with the weight W = W + 0.1),
 * than the approximation of the gradient is G = (E2-E1)/0.1. */
#define GTRIVIAL_DELTA 0.001
void AnnCalculateGradientsTrivial(AnnRprop *net, ann_float_t *desired) {
    int j, i, layers = ANN_LAYERS(net);

    for (j = 1; j < layers; j++) {
        int weights = ANN_WEIGHTS(net,j);
        for (i = 0; i < weights; i++) {
            ann_float_t t, e1, e2;
            AnnLayer *layer = &ANN_LAYER(net,j);

            /* Calculate the value of the error function
             * in this point. */
            AnnSimulate(net);
            e1 = AnnGlobalError(net, desired);
            t = layer->weight[i];
            /* Calculate the error a bit on the right */
            layer->weight[i] += GTRIVIAL_DELTA;
            AnnSimulate(net);
            e2 = AnnGlobalError(net, desired);
            /* Restore the original weight */
            layer->weight[i] = t;
            /* Calculate the gradient */
            layer->gradient[i] = (e2-e1)/GTRIVIAL_DELTA;
        }
    }
}

/* Calculate gradients using the back propagation algorithm */
void AnnCalculateGradients(AnnRprop *net, ann_float_t *desired) {
    int j, layers = ANN_LAYERS(net)-1;

    /* Populate the error vector net->layer[0]->error according
     * to the loss function. */
    AnnCalculateOutputError(net,desired);

    /* Back-propagate the error and compute the gradient
     * for every weight in the net. */
    for (j = 0; j < layers; j++) {
        AnnLayer *layer = &ANN_LAYER(net, j);
        AnnLayer *prev_layer = &ANN_LAYER(net, j+1);
        int i, units = layer->units;
        int prevunits = prev_layer->units;

        int prevunits_aligned = prev_layer->units_aligned;
#ifdef USING_SIMD
        int xps, psteps = prevunits/SIMDF_SIZE;
        simdf_t es;
#endif // USING_SIMD
        /* Skip bias units, they have no connections with the previous
         * layers. */
        if (j > 1) units--;
        /* Reset the next layer errors array */
        //for (i = 0; i < prevunits; i++) prev_layer->error[i] = 0;
        memset(prev_layer->error, 0, ANN_SIZEOF_ann_float_t*prevunits);
        /* For every node in this layer ... */
        for (i = 0; i < units; i++) {
            ann_float_t error_signal, ei, oi, derivative;
            int k;

            /* Compute gradient. */
            ei = layer->error[i];
            oi = layer->output[i];

            /* Common derivatives:
             *
             * identity: 1
             * sigmoid: oi*(1-oi)
             * softmax: oi*(1-oi)
             * tanh:    (1-oi)*(1+oi), that's 1-(oi*oi)
             * relu:    (oi > 0) ? 1 : 0
             */
            //derivative = oi*(1-oi);
            derivative = (*net->derivative_func)(oi);
            error_signal = ei*derivative;

            /* For every weight between this node and
             * the previous layer's nodes: */
            ann_float_t *g = prev_layer->gradient + i*prevunits_aligned;
            ann_float_t *w = prev_layer->weight + i*prevunits_aligned;
            ann_float_t *o = prev_layer->output;
            ann_float_t *e = prev_layer->error;

            /* 1. Calculate the gradient */
            k = 0;

#ifdef USING_SIMD
            if(psteps)
            {
                es = simdf_set1f(error_signal);
//printf("%d : %ld\n", __LINE__, ((long)o & 15));
                for (xps = 0; xps < psteps; xps++) {
                    simdf_t outputs = simdf_load(o);
                    simdf_t gradients = simdf_mul(es,outputs);
                    simdf_store(g, gradients);
                    o += SIMDF_SIZE;
                    g += SIMDF_SIZE;
                }
                k += psteps*SIMDF_SIZE;
            }
#endif
            /* Handle final piece shorter than SIMDF_SIZE . */
            for (; k < prevunits; k++) *g++ = error_signal*(*o++);

            /* 2. And back-propagate the error to the previous layer */
            k = 0;
#ifdef USING_SIMD
            if(psteps)
            {
                for (xps = 0; xps < psteps; xps++) {
                    simdf_t weights = simdf_load(w);
                    simdf_t errors = simdf_load(e);
                    simdf_t prod = simdf_mul(es, weights);
                    simdf_store(e, simdf_add(prod , errors));
                    e += SIMDF_SIZE;
                    w += SIMDF_SIZE;
                }
                k += psteps*SIMDF_SIZE;
            }
#endif
            /* Handle final piece shorter than SIMDF_SIZE . */
            for (; k < prevunits; k++) {
                (*e++) += error_signal * (*w++);
            }
        }
    }
}

/* Set the delta values of the net to a given value */
void AnnSetDeltas(AnnRprop *net, ann_float_t val) {
    int j, layers = ANN_LAYERS(net);

    for (j = 1; j < layers; j++) {
        int weights = ANN_WEIGHTS(net,j);
        int i;

        AnnLayer *layer = &ANN_LAYER(net, j);
        for (i = 0; i < weights; i++) layer->delta[i] = val;
    }
}

/* Set the sgradient values to zero */
void AnnResetSgradient(AnnRprop *net) {
    int j, layers = ANN_LAYERS(net);

    for (j = 1; j < layers; j++) {
        int weights = ANN_WEIGHTS(net, j);
        memset(ANN_LAYER(net, j).sgradient, 0, ANN_SIZEOF_ann_float_t*weights);
    }
}

/* Set random weights in the range -0.05,+0.05 */
void AnnSetRandomWeights(AnnRprop *net) {
    int i, j, k;

    for (i = 1; i < ANN_LAYERS(net); i++) {
        for (k = 0; k < ANN_UNITS(net, i-1); k++) {
            for (j = 0; j < ANN_UNITS(net, i); j++) {
                ANN_WEIGHT(net,i,j,k) = -0.05+.1*(rand()/(RAND_MAX+1.0));
            }
        }
    }
}

/* Scale the net weights of the given factor */
void AnnScaleWeights(AnnRprop *net, ann_float_t factor) {
    int j, layers = ANN_LAYERS(net);

    for (j = 1; j < layers; j++) {
        int weights = ANN_WEIGHTS(net,j);
        int i;

        AnnLayer *layer = &ANN_LAYER(net, j);
        for (i = 0; i < weights; i++)
            layer->weight[i] *= factor;
    }
}

/* Update the sgradient, that's the sum of the weight's gradient for every
 * element of the training set. This is used for the RPROP algorithm
 * that works with the sign of the derivative for the whole set. */
void AnnUpdateSgradient(AnnRprop *net) {
    int j, i, layers = ANN_LAYERS(net);

    for (j = 1; j < layers; j++) {
        int weights = ANN_WEIGHTS(net,j);
        ann_float_t *sg = net->layer[j].sgradient;
        ann_float_t *g = net->layer[j].gradient;
        i = 0;
#ifdef USING_SIMD
            int psteps = weights/SIMDF_SIZE;
            if(psteps)
            {
                int xps;
                for (xps = 0; xps < psteps; xps++) {
                    simdf_t sgradient = simdf_load(sg);
                    simdf_t gradient = simdf_load(g);
                    simdf_store(sg, simdf_add( sgradient, gradient));
                    sg += SIMDF_SIZE;
                    g += SIMDF_SIZE;
                }
                i += psteps*SIMDF_SIZE;
            }
#endif
        /* Handle final piece shorter than SIMDF_SIZE . */
        for (; i < weights; i++)
            (*sg++) += (*g++);
    }
}

/* Helper function for RPROP, returns -1 if n < 0, +1 if n > 0, 0 if n == 0 */
static inline ann_float_t sign(ann_float_t n) {
    if (n > 0) return +1.0;
    if (n < 0) return -1.0;
    return 0.0;
}

/* The core of the RPROP algorithm.
 *
 * Note that:
 * sgradient is the set-wise gradient.
 * delta is the per-weight update value. */
void AnnAdjustWeightsResilientBP(AnnRprop *net) {
    int j, i, layers = ANN_LAYERS(net);

    for (j = 1; j < layers; j++) {
        int weights = ANN_WEIGHTS(net,j) - (j-1>0);
        AnnLayer *layer = &ANN_LAYER(net, j);
        for (i = 0; i < weights; i++) {
            ann_float_t sgradient = layer->sgradient[i];
            ann_float_t t = layer->pgradient[i] * sgradient;
            ann_float_t delta = layer->delta[i];

            if (t > 0) {
                delta = ANN_MIN(delta*ANN_RPROP_NPLUS(net),ANN_RPROP_MAXUPDATE(net));
                ann_float_t wdelta = -sign(sgradient) * delta;
                layer->weight[i] += wdelta;
                layer->delta[i] = delta;
                layer->pgradient[i] = sgradient;
            } else if (t < 0) {
                ann_float_t past_wdelta = -sign(layer->pgradient[i]) * delta;
                delta = ANN_MAX(delta*ANN_RPROP_NMINUS(net),ANN_RPROP_MINUPDATE(net));
                layer->weight[i] -= past_wdelta;
                layer->delta[i] = delta;
                layer->pgradient[i] = 0;
            } else { /* t == 0 */
                ann_float_t wdelta = -sign(sgradient) * delta;
                layer->weight[i] += wdelta;
                layer->pgradient[i] = sgradient;
            }
        }
    }
}

/* Resilient Backpropagation Epoch */
ann_float_t AnnResilientBPEpoch(AnnRprop *net, ann_float_t *input, ann_float_t *desired, int setlen) {
    ann_float_t error = 0;
    int j, inputs = ANN_INPUT_UNITS(net), outputs = ANN_OUTPUT_UNITS(net);

    AnnResetSgradient(net);
    for (j = 0; j < setlen; j++) {
        error += AnnSimulateError(net, input, desired);
        AnnCalculateGradients(net, desired);
        AnnUpdateSgradient(net);
        input += inputs;
        desired += outputs;
    }
    AnnAdjustWeightsResilientBP(net);
    return error / setlen;
}

/* Update the deltas using the gradient descend algorithm.
 * Gradients should be already computed with AnnCalculateGraidents(). */
void AnnUpdateDeltasGD(AnnRprop *net) {
    int j, i, layers = ANN_LAYERS(net);

    for (j = 1; j < layers; j++) {
        int weights = ANN_WEIGHTS(net,j);
        AnnLayer *layer = &ANN_LAYER(net, j);
        for (i = 0; i < weights; i++)
            layer->delta[i] += layer->gradient[i];
    }
}

/* Adjust net weights using the (already) calculated deltas. */
void AnnAdjustWeights(AnnRprop *net, int setlen) {
    int j, i, layers = ANN_LAYERS(net);

    for (j = 1; j < layers; j++) {
        int weights = ANN_WEIGHTS(net,j);
        AnnLayer *layer = &ANN_LAYER(net, j);
        for (i = 0; i < weights; i++) {
            layer->weight[i] -= ANN_LEARN_RATE(net)/setlen*layer->delta[i];
        }
    }
}

/* Gradient Descend training */
ann_float_t AnnGDEpoch(AnnRprop *net, ann_float_t *input, ann_float_t *desidered, int setlen) {
    ann_float_t error = 0;
    int j, inputs = ANN_INPUT_UNITS(net), outputs = ANN_OUTPUT_UNITS(net);

    for (j = 0; j < setlen; j++) {
        AnnSetDeltas(net, 0);
        error += AnnSimulateError(net, input, desidered);
        AnnCalculateGradients(net, desidered);
        AnnUpdateDeltasGD(net);
        input += inputs;
        desidered += outputs;
        AnnAdjustWeights(net,setlen);
    }
    return error / setlen;
}

/* This function, called after AnnSimulate(), will return 1 if there is
 * an error in the detected class (compared to the desired output),
 * othewise 0 is returned. */
int AnnTestClassError(AnnRprop *net, ann_float_t *desired) {
    int i, outputs = ANN_OUTPUT_UNITS(net);
    int classid, outid;
    ann_float_t max = 0;

    /* Get the class ID from the test dataset output. */
    classid = 0;
    for (i = 0; i < outputs; i++)
        if (desired[i] == 1) break;
    classid = i;

    /* Get the network classification. */
    max = ANN_OUTPUT_NODE(net,0);
    outid = 0;
    for (i = 1; i < outputs; i++) {
        ann_float_t o = ANN_OUTPUT_NODE(net,i);
        if (o > max) {
            outid = i;
            max = o;
        }
    }
    return outid != classid;
}

/* Simulate the entire test dataset with the neural network and returns the
 * average error of all the entries tested. */
void AnnTestError(AnnRprop *net, ann_float_t *input, ann_float_t *desired, int setlen, ann_float_t *avgerr, ann_float_t *classerr) {
    ann_float_t error = 0;
    int j, inputs = ANN_INPUT_UNITS(net), outputs = ANN_OUTPUT_UNITS(net);
    int class_errors = 0;

    for (j = 0; j < setlen; j++) {
        error += AnnSimulateError(net, input, desired);
        if (classerr)
            class_errors += AnnTestClassError(net, desired);
        input += inputs;
        desired += outputs;
    }
    if (avgerr) *avgerr = error/setlen;
    if (classerr) *classerr = (ann_float_t)class_errors*100/setlen;
}

/* Train the net */
ann_float_t AnnTrainWithAlgoFunc(AnnRprop *net, ann_float_t *input, ann_float_t *desired, ann_float_t maxerr,
                                        int maxepochs, int setlen, AnnTrainAlgoFunc algo_func) {
    int i = 0;
    ann_float_t e = maxerr+1;

    while (i++ < maxepochs && e >= maxerr) {
        e = (*algo_func)(net, input, desired, setlen);
    }
    return e;
}


ann_float_t AnnTrain(AnnRprop *net, ann_float_t *input, ann_float_t *desired, ann_float_t maxerr, int maxepochs,
                                                                                int setlen, int algo) {
    AnnTrainAlgoFunc algo_func;
    if(algo == ANN_ALGO_BPROP) algo_func = AnnResilientBPEpoch;
    else if(algo == ANN_ALGO_GD) algo_func = AnnGDEpoch;
    else return -1;

    return AnnTrainWithAlgoFunc(net, input, desired, maxerr, maxepochs, setlen, algo_func);
}
