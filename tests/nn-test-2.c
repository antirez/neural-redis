/* This test was taken from:
 *
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 *
 * It is interesting since allows to cross-validate the implementation
 * with manually obtained values. */

#include <stdio.h>

#include "../nn.h"

int main(void) {
    AnnRprop *nn = AnnCreateNet3(2, 3, 1);
    ann_float_t inputs[8] = {0,0, 1,0, 0,1, 1,1};
    ann_float_t desired[4] = {0, 1, 1, 0};

    nn->learn_rate = 0.5;

    int j;
    for (j = 0; j < 100000; j++) {
        ann_float_t error = AnnTrain(nn, inputs, desired, 0, 1, 4, ANN_ALGO_GD);
        printf("Error: %f\n", error);
    }
    printf("\nAfter training:\n\n");
    for (j = 0; j < 4; j++) {
        AnnSetInput(nn,inputs+j*2);
        AnnSimulate(nn);
        printf("%f\n", ANN_OUTPUT_NODE(nn,0));
    }
    printf("\nTCL simulation:\n\n");
    Ann2Tcl(nn);
    Ann2Js(nn);
    AnnFree(nn);
    return 0;
}
