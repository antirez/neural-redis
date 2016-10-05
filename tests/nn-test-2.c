/* This test was taken from:
 *
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 *
 * It is interesting since allows to cross-validate the implementation
 * with manually obtained values. */

#include <stdio.h>

#include "../nn.h"

int main(void) {
    struct Ann *nn = AnnCreateNet3(2, 3, 1);
    float inputs[8] = {0,0, 1,0, 0,1, 1,1};
    float desired[4] = {0, 1, 1, 0};

    nn->learn_rate = 0.5;

    int j;
    for (j = 0; j < 100000; j++) {
        float error = AnnTrain(nn, inputs, desired, 0, 1, 4, NN_ALGO_GD);
        printf("Error: %f\n", error);
    }
    printf("\nAfter training:\n\n");
    for (j = 0; j < 4; j++) {
        AnnSetInput(nn,inputs+j*2);
        AnnSimulate(nn);
        printf("%f\n", OUTPUT_NODE(nn,0));
    }
    return 0;
}
