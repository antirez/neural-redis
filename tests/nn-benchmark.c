/* This benchmark generates a random set of inputs and outputs
 * on a large network and tests the backpropagation speed in order
 * for the net to arrive at a certain level of error. */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../nn.h"

#define NUM_INPUTS 300
#define NUM_OUTPUTS 100

long long mstime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec)*1000000;
    ust += tv.tv_usec;
    return ust/1000;
}

int math_random(int low, int up) {
  ann_float_t r = rand() * (1.0 / (RAND_MAX + 1.0));
  r *= (up - low) + 1.0;
  return (int)r+low;
}

void gen_dataset(AnnRprop *nn, ann_float_t **inputs, ann_float_t **outputs, int setsize) {
    *inputs = calloc(1, sizeof(ann_float_t)*setsize*NUM_INPUTS);
    *outputs = calloc(1, sizeof(ann_float_t)*setsize*NUM_OUTPUTS);
    int ilen = ANN_INPUT_UNITS(nn);
    int olen = ANN_OUTPUT_UNITS(nn);
    int olen_1 = olen - 1;

    ann_float_t *in = *inputs;
    ann_float_t *out = *outputs;
    for (int j = 0; j < setsize; j++) {
        for (int k = 0; k < ilen; k++) in[k] = rand() & 1;
        //int r = rand() & olen_1;
        int r = math_random(0, olen_1);
		out[r] = 1; 
		//printf("%d : %d\n", j, r);
        //for (int k = 0; k < olen; k++) {
        //    out[k] = (k == r) ? 1 : 0;
        //}
        in+= ilen;
        out+= olen;
    }
}

int main(void) {
    AnnRprop *nn = AnnCreateNet3(NUM_INPUTS, NUM_INPUTS*2, NUM_OUTPUTS);
    ann_float_t *inputs, *outputs;
    int setsize = 1000;

    nn->learn_rate = 0.5;
    gen_dataset(nn, &inputs, &outputs, setsize);

    int j;
    ann_float_t classerr = 100;
    long long totaltime = 0;
    int benchmark_milestone = 0;
    for (j = 0; j < 1000000; j++) {
        printf("[%d] Error: %f%% -- %lld ms per cycle\n", j, classerr,
            j ? totaltime/j : 0);
        if (classerr <= 50 && !benchmark_milestone) {
            printf("*** Error <= 50%% reached in %lld ms\n", totaltime);
            benchmark_milestone = 1;
        }
        long long start = mstime();
        AnnTrain(nn,inputs,outputs,0,1,setsize,ANN_ALGO_BPROP);
        long long elapsed = mstime() - start;
        totaltime += elapsed;

        AnnTestError(nn,inputs,outputs,setsize,NULL,&classerr);
    }
    AnnFree(nn);
    return 0;
}
