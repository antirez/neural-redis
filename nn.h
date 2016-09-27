#ifndef __NN_H
#define __NN_H

/* Data structures.
 * Nets are not so 'dynamic', but enough to support
 * an arbitrary number of layers, with arbitrary units for layer.
 * Only fully connected feed-forward networks are supported. */
struct AnnLayer {
	int units;
	double *output;		/* output[i], output of i-th unit */
	double *error;		/* error[i], output error of i-th unit*/
	double *weight;		/* weight[(i*units)+j] */
				/* weight between unit i-th and next j-th */
	double *gradient;	/* gradient[(i*units)+j] gradient */
	double *sgradient;	/* gradient for the full training set */
				/* only used for RPROP */
	double *pgradient;	/* pastgradient[(i*units)+j] t-1 gradient */
				/* (t-1 sgradient for resilient BP) */
	double *delta;		/* delta[(i*units)+j] cumulative update */
				/* (per-weight delta for RPROP) */
};

/* Feed forward network structure */
struct Ann {
	int flags;
	int layers;
	double rprop_nminus;
	double rprop_nplus;
	double rprop_maxupdate;
	double rprop_minupdate;
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
double AnnGlobalError(struct Ann *net, double *desidered);
void AnnSetInput(struct Ann *net, double *input);
double AnnSimulateError(struct Ann *net, double *input, double *desidered);
void AnnCalculateGradientsTrivial(struct Ann *net, double *desidered);
void AnnCalculateGradients(struct Ann *net, double *desidered);
void AnnSetDeltas(struct Ann *net, double val);
void AnnResetDeltas(struct Ann *net);
void AnnResetSgradient(struct Ann *net);
void AnnSetRandomWeights(struct Ann *net);
void AnnScaleWeights(struct Ann *net, double factor);
void AnnUpdateDeltasGD(struct Ann *net);
void AnnUpdateDeltasGDM(struct Ann *net);
void AnnUpdateSgradient(struct Ann *net);
void AnnAdjustWeights(struct Ann *net);
double AnnBatchGDEpoch(struct Ann *net, double *input, double *desidered, int setlen);
double AnnBatchGDMEpoch(struct Ann *net, double *input, double *desidered, int setlen);
void AnnAdjustWeightsResilientBP(struct Ann *net);
double AnnResilientBPEpoch(struct Ann *net, double *input, double *desidered, int setlen);
double AnnTrain(struct Ann *net, double *input, double *desidered, double maxerr, int maxepochs, int setlen);
double AnnTestError(struct Ann *net, double *input, double *desidered, int setlen);

#endif /* __NN_H */
