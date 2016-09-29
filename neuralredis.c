/* This file implements a neural network datatype with training capabilities
 * as a Redis module.
 *
 * Check https://github.com/antirez/neural-redis/ for more information
 *
 * -----------------------------------------------------------------------------
 *
 * Copyright (c) 2016, Salvatore Sanfilippo <antirez at gmail dot com>
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
 *   * Neither the name of Redis nor the names of its contributors may be used
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

#include "redismodule.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/time.h>

#include "nn.h"

static RedisModuleType *NRType;
uint64_t NRNextId = 1; /* Next neural network unique ID. */

/* ========================== Internal data structure  ====================== */

#define NR_FLAG_NONE 0
#define NR_FLAG_TRAINING (1<<0)         /* NN is training in a thread. */
#define NR_FLAG_REGRESSOR (1<<1)        /* NN will be used for regression. */
#define NR_FLAG_CLASSIFIER (1<<2)       /* NN will be used for classification.*/
#define NR_FLAG_NORMALIZE (1<<3)        /* Perform input/output normalization.*/
#define NR_FLAG_AUTO_STOP (1<<4)        /* Auto stop on training. */
#define NR_FLAG_OF_DETECTED (1<<5)      /* Auto stopped on overfitting. */

/* Flags to persist when saving the NN. */
#define NR_FLAG_TO_PRESIST (NR_FLAG_REGRESSOR| \
                            NR_FLAG_CLASSIFIER| \
                            NR_FLAG_NORMALIZE)

/* Flags to transfer after training. */
#define NR_FLAG_TO_TRANSFER (NR_FLAG_OF_DETECTED)

#define NR_MAX_LAYERS 32

typedef struct NRDataset {
    uint32_t len, maxlen;
    double *inputs, *outputs;
} NRDataset;

typedef struct {
    uint64_t id;        /* Neural network unique ID. */
    uint64_t training_total_steps; /* How many steps of trainig the network
                                      received. A step is a single input/output
                                      pattern presented to the net (counting
                                      the same pattern multiple times) */
    uint64_t training_total_ms;   /* Total milliseconds time of training. */
    uint64_t training_max_cycles; /* Max cycles of a single training. */
    uint64_t training_max_ms; /* Max time of a single training. */
    uint32_t flags;     /* NR_FLAG_... */
    uint32_t epochs;    /* Number of training epochs so far. */
    struct Ann *nn;     /* Neural network structure. */
    NRDataset dataset;  /* Training dataset. */
    NRDataset test;     /* Testing dataset. */
    double dataset_error;   /* Average error in the training dataset. */
    double test_error;      /* Average error in the test dataset. */
} NRTypeObject;

struct {
    RedisModuleString *key; /* Key name of the NN we are training.
                               Set to NULL for unused slots. */
    int db_id;          /* DB ID where the key is. */
    pthread_t tid;      /* Thread ID of the trainer. */
    int in_progress;    /* 0 if training terminated. */
    NRTypeObject *nr;   /* A copy of the NN we are training. */
} typedef NRPendingTraining;

/* We take an array with NNs currently training in other threads.
 * Every time an NN command is called, we try to see if there are
 * finished trainings, in order to udpate weights of the original
 * NN stored into the key (we work on a copy on the other thread).*/
#define NR_PENDING_TRAINING_MAX_LEN 32

static pthread_mutex_t NRPendingTrainingMutex = PTHREAD_MUTEX_INITIALIZER;
/* All the followings must be accessed after acquiring the mutex. */
static NRPendingTraining NRTrainings[NR_PENDING_TRAINING_MAX_LEN];
static int NRPendingTrainingCount = 0; /* Number of pending trainings. */

/* ========================== Low level object API ========================== */

long long NRMilliseconds(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec)*1000000;
    ust += tv.tv_usec;
    return ust/1000;
}

/* Create a network with the specified parameters. Note that the layers
 * must be specified from the output layer[0] to the input
 * layer[N]. Each element in the integer array 'layer' specify how many
 * units there are in the corresponding layer. */
NRTypeObject *createNRTypeObject(int flags, int *layers, int numlayers, int dset_len, int test_len) {
    NRTypeObject *o;
    o = RedisModule_Calloc(1,sizeof(*o));
    o->id = NRNextId++;
    o->flags = flags;
    o->nn = AnnCreateNet(numlayers,layers);
    o->dataset.maxlen = dset_len;
    o->test.maxlen = test_len;
    return o;
}

/* Insert data (observations needed to train and test the NN) into the
 * NN object. While the learning and testing datasets are yet not full
 * the observed pattern is inserted evenly in one or the other side in
 * order to make sure the two datasets are populated evenly. When both
 * are already full, a random elmenet from one or the other (doing
 * a random weighted choice depending on the length) is substituted with
 * the new item. */
void NRTypeInsertData(NRTypeObject *o, double *inputs, double *outputs) {
    NRDataset *target;

    /* Check if there is no dataset at all. This may be a valid setup
     * with online learning, sample by sample. */
    if (o->dataset.maxlen == 0 && o->test.maxlen == 0) return;

    /* Select target, to populate the one with less data compared to size. */
    if (o->dataset.len != o->dataset.maxlen ||
        o->test.len != o->dataset.len)
    {
        if (o->dataset.maxlen == 0) {
            target = &o->test;
        } else if (o->test.maxlen == 0) {
            target = &o->dataset;
        } else {
            double fill_a = (double)o->dataset.len / o->dataset.maxlen;
            double fill_b = (double)o->test.len / o->test.maxlen;
            target = (fill_a <= fill_b) ? &o->dataset : &o->test;
        }
    }

    /* Append if there is room or substitute with a random entry. */
    size_t idx;
    int j, numin = INPUT_UNITS(o->nn),
           numout = OUTPUT_UNITS(o->nn);

    if (target->maxlen == target->len) {
        idx = rand() % target->maxlen;
    } else {
        idx = target->len;
        target->len++;
        target->inputs = RedisModule_Realloc(target->inputs,
            sizeof(double)*numin*target->len);
        target->outputs = RedisModule_Realloc(target->outputs,
            sizeof(double)*numout*target->len);
    }

    /* Finally store the values at position. */
    for (j = 0; j < numin; j++)
        target->inputs[idx*numin+j] = inputs[j];
    for (j = 0; j < numout; j++)
        target->outputs[idx*numout+j] = outputs[j];
}

/* Free the specified dataset. */
void NRDatasetFree(NRDataset *dset) {
    RedisModule_Free(dset->inputs);
    RedisModule_Free(dset->outputs);
}

/* Free a whole NN object. */
void NRTypeReleaseObject(NRTypeObject *o) {
    AnnFree(o->nn);
    NRDatasetFree(&o->dataset);
    NRDatasetFree(&o->test);
    RedisModule_Free(o);
}

/* ================================ Training =============================== */

/* Clone a neural network object, including the training and test dataset.
 * We use cloning in order to train in a different thread, and later
 * copy the weights back into the original NN.
 *
 * Note when 'newid' is 0, the copied object NN unique ID is the same as the
 * original as normally this is what we want, in order to later match the
 * trained network with the object stored at the specified key
 * in the pending traning structure.
 *
 * However if the copy is performed with other goals, 'newid' should
 * be set to non-zero in order to create a net with a different ID. */
NRTypeObject *NRClone(NRTypeObject *o, int newid) {
    NRTypeObject *copy;
    copy = RedisModule_Calloc(1,sizeof(*o));
    *copy = *o;
    if (newid) copy->id = NRNextId++;
    copy->nn = AnnClone(o->nn);
    copy->dataset = o->dataset;
    copy->test = o->test;

    int ilen = INPUT_UNITS(o->nn);
    int olen = OUTPUT_UNITS(o->nn);
    copy->dataset.inputs = RedisModule_Alloc(sizeof(double)*ilen*o->dataset.len);
    copy->dataset.outputs = RedisModule_Alloc(sizeof(double)*olen*o->dataset.len);
    copy->test.inputs = RedisModule_Alloc(sizeof(double)*ilen*o->test.len);
    copy->test.outputs = RedisModule_Alloc(sizeof(double)*olen*o->test.len);
    memcpy(copy->dataset.inputs,o->dataset.inputs,sizeof(double)*ilen*o->dataset.len);
    memcpy(copy->dataset.outputs,o->dataset.outputs,sizeof(double)*olen*o->dataset.len);
    memcpy(copy->test.inputs,o->test.inputs,sizeof(double)*ilen*o->test.len);
    memcpy(copy->test.outputs,o->test.outputs,sizeof(double)*olen*o->test.len);
    return copy;
}

/* Transfer the weights from the source to the destination NN.
 * This is used after the learning process finished in a different
 * thread in order to transfer the learning back to the orignal
 * NN. */
void NRTransferWeights(RedisModuleCtx *ctx, NRTypeObject *dst, NRTypeObject *src) {
    if (dst->id != src->id) {
        RedisModule_Log(ctx,"warning",
            "NSTransferWeight(): source and destination neural network IDs "
            "don't match. This is unexpected, probably a bug inside the "
            "module. Weights not transferred back to the origina NN.");
        return;
    }

    /* It would be faster to memcpy just the weight array for each layer,
     * however this way we access the NN in a more abstract way, and should
     * be fast enough in most cases. We can always optimized it later. */
    AnnFree(dst->nn);
    dst->nn = AnnClone(src->nn);
    dst->training_total_steps = src->training_total_steps;
    dst->training_total_ms = src->training_total_ms;
    dst->dataset_error = src->dataset_error;
    dst->test_error = src->test_error;
    dst->flags &= ~NR_FLAG_TO_TRANSFER;
    dst->flags |= src->flags & NR_FLAG_TO_TRANSFER;
}

/* Threaded training entry point. */
void *NRTrainingThreadMain(void *arg) {
    NRPendingTraining *pt = arg;
    NRTypeObject *nr = pt->nr;
    int training_iterations = 1;
    double train_error = 0;
    double test_error = 0;
    double past_train_error = 1.0/0;
    double past_test_error = 1.0/0;
    int auto_stop = nr->flags & NR_FLAG_AUTO_STOP;

    uint64_t cycles;
    long long start = NRMilliseconds();
    long long cycle_time;
    int overfitting_count = 0;
    while(1) {
        long long cycle_start = NRMilliseconds();
        train_error = AnnTrain(nr->nn,
                               nr->dataset.inputs,
                               nr->dataset.outputs,
                               0,
                               training_iterations,
                               nr->dataset.len);
        cycle_time = NRMilliseconds() - cycle_start;
        nr->training_total_ms += cycle_time;
        nr->training_total_steps += nr->dataset.len*training_iterations;

        /* Evaluate the error in the case of auto training, stop it
         * once we see that the error in the traning set is decreasing
         * while the one in the test set is not. */
        if (auto_stop) {
            test_error = AnnTestError(nr->nn,
                                      nr->test.inputs,
                                      nr->test.outputs,
                                      nr->test.len);

            if (train_error < past_train_error &&
                test_error > past_test_error)
            {
                overfitting_count++;
                if (overfitting_count == 5) {
                    nr->flags |= NR_FLAG_OF_DETECTED;
                    break;
                }
            } else {
                overfitting_count = 0;
            }

            /* Also stop if the loss is zero in both datasets. */
            if (train_error < 0.000000000000001 && test_error < 0.000000000000001) break;
        }

        cycles++;
        long long total_time = NRMilliseconds()-start;

        /* Cycles and milliseconds stop conditions. */
        if (nr->training_max_cycles && cycles == nr->training_max_cycles)
            break;
        if (nr->training_max_ms && total_time > nr->training_max_ms)
            break;

        /* If this is a long training, to do just a single training iteration
         * for each cycle is not optimal: tune the number of iterations to
         * at least take 100 milliseconds. */
        if (total_time > 10000 && cycle_time < 100) training_iterations++;

        past_train_error = train_error;
        past_test_error = test_error;
    }

    /* If auto stop is disabled, we still need to compute the test error
     * in order to return this information to the main thread. */
    if (!auto_stop) {
        test_error = AnnTestError(nr->nn,
                                  nr->test.inputs,
                                  nr->test.outputs,
                                  nr->test.len);
    }
    nr->dataset_error = train_error;
    nr->test_error = test_error;

    /* Signal that the training process has finished, it's up to the main
     * thread to cleanup this training slot, copying the weights to the
     * original neural network and reclaiming memory for the copy we
     * used to work. */
    pthread_mutex_lock(&NRPendingTrainingMutex);
    pt->in_progress = 0;
    pthread_mutex_unlock(&NRPendingTrainingMutex);
    return NULL;
}

/* Start a background training in another thread. Return REDISMODULE_ERR if
 * there is no free slot for training, as we already reached the maximum of
 * networks we can train in parallel.
 *
 * The 'flags' argument specifies the additional NN flags to pass to the
 * training ruotine:
 *
 *  NR_FLAG_AUTO_STOP -- Automatically stop training on overtraining. */
int NRStartTraining(RedisModuleCtx *ctx, RedisModuleString *key, int dbid, NRTypeObject *nr) {
    pthread_mutex_lock(&NRPendingTrainingMutex);
    if (NRPendingTrainingCount == NR_PENDING_TRAINING_MAX_LEN) {
        pthread_mutex_unlock(&NRPendingTrainingMutex);
        return REDISMODULE_ERR;
    }

    /* Setup our trainig data. */
    NRPendingTraining *pt = &NRTrainings[NRPendingTrainingCount];
    pt->key = RedisModule_CreateStringFromString(ctx,key);
    RedisModule_RetainString(ctx,pt->key);
    pt->db_id = dbid;
    pt->in_progress = 1;
    pt->nr = NRClone(nr,0);
    if (pthread_create(&pt->tid,NULL,NRTrainingThreadMain,pt) != 0) {
        RedisModule_Log(ctx,"warning","Unable to create a new pthread in NRStartTraining()");
        RedisModule_FreeString(ctx,pt->key);
        pt->key = NULL;
        NRTypeReleaseObject(pt->nr);
        pthread_mutex_unlock(&NRPendingTrainingMutex);
        return REDISMODULE_ERR;
    }
    NRPendingTrainingCount++;
    nr->flags |= NR_FLAG_TRAINING;
    pthread_mutex_unlock(&NRPendingTrainingMutex);
    return REDISMODULE_OK;
}

/* Check if there are threads that terminated the NN training, and
 * collect the info they computed (that is the new NN). */
int NRCollectThreads(RedisModuleCtx *ctx) {
    int collected = 0;
    pthread_mutex_lock(&NRPendingTrainingMutex);
    for (int j = 0; j < NRPendingTrainingCount; j++) {
        NRPendingTraining *pt = &NRTrainings[j];
        if (pt->in_progress == 0) {
            /* Training terminated. Let's see if the key
             * is still there and NN ID matches. */
            int orig_id = RedisModule_GetSelectedDb(ctx);
            if (orig_id != pt->db_id) RedisModule_SelectDb(ctx,pt->db_id);
            RedisModuleKey *key = RedisModule_OpenKey(ctx,pt->key,
                REDISMODULE_READ|REDISMODULE_WRITE);
            int type = RedisModule_KeyType(key);
            if (RedisModule_ModuleTypeGetType(key) == NRType) {
                NRTypeObject *nr = RedisModule_ModuleTypeGetValue(key);
                if (nr->id == pt->nr->id) {
                    NRTransferWeights(ctx,nr,pt->nr);
                    nr->flags &= ~NR_FLAG_TRAINING;
                }
                RedisModule_FreeString(ctx,pt->key);
                pt->key = NULL;
                NRTypeReleaseObject(pt->nr);
                NRPendingTrainingCount--;
                memcpy(&NRTrainings[j],&NRTrainings[j+1],
                    (NRPendingTrainingCount-j)*sizeof(NRTrainings[0]));
            }
            if (orig_id != pt->db_id) RedisModule_SelectDb(ctx,orig_id);
            collected++;
        }
    }
    pthread_mutex_unlock(&NRPendingTrainingMutex);
    return collected;
}

/* ================================ Commands =============================== */

/* NR.CREATE <key> <type> <inputs> [<hidden> ...] -> <outputs> [DATASET <items>]
 * [TEST <items>] [NORMALIZE] */
int NRCreate_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    long long dset_size = 0, test_size = 0;
    int layers[NR_MAX_LAYERS], num_layers = 0;
    int flags = NR_FLAG_NONE;
    RedisModule_AutoMemory(ctx);
    NRCollectThreads(ctx);

    if (argc < 6) return RedisModule_WrongArity(ctx);

    const char *nntype = RedisModule_StringPtrLen(argv[2], NULL);
    if (!strcasecmp(nntype,"classifier")) {
        flags |= NR_FLAG_CLASSIFIER;
    } else if (!strcasecmp(nntype,"regressor")) {
        flags |= NR_FLAG_REGRESSOR;
    } else {
        return RedisModule_ReplyWithError(ctx,
            "ERR invalid neural network type. Must be "
            "CLASSIFIER or REGRESSOR");
    }

    /* Parse net layers definition. */
    int j = 3, stop = 0;
    while (j < argc) {
        const char *u = RedisModule_StringPtrLen(argv[j], NULL);
        long long units;

        /* When we see -> the next layer is the final layer (output) layer. */
        if (!strcmp(u,"->")) {
            stop = 1;
            j++;
            continue;
        }
        if (RedisModule_StringToLongLong(argv[j],&units) != REDISMODULE_OK ||
            units <= 0)
        {
            return RedisModule_ReplyWithError(ctx, "ERR invalid units count");
        }
        layers[num_layers++] = units;
        j++;
        if (stop) break;
    }

    /* Our NN library takes the definition of layers in the opposite
     * order, swap the layers array. */
    for (int i = 0; i < num_layers/2; i++) {
        int t = layers[i];
        layers[i] = layers[num_layers-1-i];
        layers[num_layers-1-i] = t;
    }

    /* Parse the remaining options. */
    for (; j < argc; j++) {
        const char *o = RedisModule_StringPtrLen(argv[j], NULL);
        long long v;
        int lastarg = (j == argc-1);
        if ((!strcasecmp(o,"dataset") ||
             !strcasecmp(o,"test")) && !lastarg)
        {
            if ((RedisModule_StringToLongLong(argv[j+1],&v) != REDISMODULE_OK) ||
                 v < 0)
            {
                return RedisModule_ReplyWithError(ctx,
                    "ERR invalid dataset size");
            }
            if (!strcasecmp(o,"dataset"))
                dset_size = v;
            else
                test_size = v;
            j++;
        } else if (!strcasecmp(o,"normalize")) {
            flags |= NR_FLAG_NORMALIZE;
        } else {
            return RedisModule_ReplyWithError(ctx,
                "ERR Syntax error in NR.CREATE");
        }
    }

    /* Open the key, and check that's available. */
    RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
        REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(key);
    if (type != REDISMODULE_KEYTYPE_EMPTY) {
        return RedisModule_ReplyWithError(ctx,"ERR the key name is busy");
    }

    /* We can finally create our neural network. */
    NRTypeObject *nr = createNRTypeObject(flags,layers,num_layers,
                              dset_size,test_size);
    RedisModule_ModuleTypeSetValue(key,NRType,nr);

    RedisModule_ReplyWithLongLong(ctx,AnnCountWeights(nr->nn));
    RedisModule_ReplicateVerbatim(ctx);
    return REDISMODULE_OK;
}

/* NR.RUN key [input1 input2 input3 ... inputN] */
int NRRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    RedisModule_AutoMemory(ctx); /* Use automatic memory management. */
    NRCollectThreads(ctx);

    if (argc < 3) return RedisModule_WrongArity(ctx);
    RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1], REDISMODULE_READ);
    int type = RedisModule_KeyType(key);
    if (RedisModule_ModuleTypeGetType(key) != NRType)
        return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);

    NRTypeObject *nr = RedisModule_ModuleTypeGetValue(key);
    int ilen = INPUT_UNITS(nr->nn);
    if (argc != ilen+2)
        return RedisModule_ReplyWithError(ctx,
            "ERR number of arguments does not "
            "match the number of inputs in the neural network");

    for(int j = 0; j < ilen; j++) {
        double input;
        if (RedisModule_StringToDouble(argv[j+2],&input) != REDISMODULE_OK)
            return RedisModule_ReplyWithError(ctx,
                "ERR invalid neural network input: must be a valid double "
                "precision floating point number");
        INPUT_NODE(nr->nn,j) = input;
    }

    AnnSimulate(nr->nn);

    int olen = OUTPUT_UNITS(nr->nn);
    RedisModule_ReplyWithArray(ctx,olen);
    for(int j = 0; j < olen; j++)
        RedisModule_ReplyWithDouble(ctx, OUTPUT_NODE(nr->nn,j));
    return REDISMODULE_OK;
}

/* NR.OBSERVE key input1 [input2 input3 ... inputN] -> output */
int NRObserve_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    RedisModule_AutoMemory(ctx); /* Use automatic memory management. */
    NRCollectThreads(ctx);

    if (argc < 3) return RedisModule_WrongArity(ctx);
    RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
        REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(key);
    if (RedisModule_ModuleTypeGetType(key) != NRType)
        return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);

    NRTypeObject *nr = RedisModule_ModuleTypeGetValue(key);
    int ilen = INPUT_UNITS(nr->nn);
    int olen = OUTPUT_UNITS(nr->nn);

    if (argc != olen+ilen+3)
        return RedisModule_ReplyWithError(ctx,
            "ERR number of arguments does not "
            "match the number of inputs and outputs in the neural network");

    const char *sep = RedisModule_StringPtrLen(argv[ilen+2], NULL);
    if (strcmp(sep,"->")) {
        return RedisModule_ReplyWithError(ctx,
            "ERR no '->' separtor in the correct position between inputs and "
            "outputs: are you sure your training data is correct?");
    }

    double *inputs = RedisModule_Alloc(sizeof(double)*ilen);
    double *outputs = RedisModule_Alloc(sizeof(double)*olen);

    for(int j = 2; j < argc; j++) {
        double val;
        if (j == ilen+2) continue; /* -> separator. */
        if (RedisModule_StringToDouble(argv[j],&val) != REDISMODULE_OK) {
            RedisModule_Free(inputs);
            RedisModule_Free(outputs);
            return RedisModule_ReplyWithError(ctx,
                "ERR invalid neural network input: must be a valid double "
                "precision floating point number");
        }
        if (j < ilen+2) {
            inputs[j-2] = val;
        } else {
            outputs[j-ilen-3] = val;
        }
    }

    NRTypeInsertData(nr,inputs,outputs);
    RedisModule_Free(inputs);
    RedisModule_Free(outputs);

    RedisModule_ReplyWithArray(ctx,2);
    RedisModule_ReplyWithLongLong(ctx, nr->dataset.len);
    RedisModule_ReplyWithLongLong(ctx, nr->test.len);
    return REDISMODULE_OK;
}

/* NR.TRAIN key [MAXCYCLES <count>] [MAXTIME <count>] [AUTOSTOP] */
int NRTrain_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    RedisModule_AutoMemory(ctx); /* Use automatic memory management. */
    NRCollectThreads(ctx);

    if (argc < 2) return RedisModule_WrongArity(ctx);
    RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
        REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(key);
    if (RedisModule_ModuleTypeGetType(key) != NRType)
        return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);

    NRTypeObject *nr = RedisModule_ModuleTypeGetValue(key);
    if (nr->flags & NR_FLAG_TRAINING)
        return RedisModule_ReplyWithError(ctx,
            "ERR neural network training already in progress");

    nr->training_max_cycles = 0;
    nr->training_max_ms = 10000;
    nr->flags &= ~(NR_FLAG_AUTO_STOP);

    for (int j = 2; j < argc; j++) {
        const char *o = RedisModule_StringPtrLen(argv[j], NULL);
        long long v;
        int lastarg = (j == argc-1);

        if (!strcasecmp(o,"autostop")) {
            nr->flags |= NR_FLAG_AUTO_STOP;
        } else if (!strcasecmp(o,"maxcycles") && !lastarg) {
            if (RedisModule_StringToLongLong(argv[++j],&v) != REDISMODULE_OK) {
                return RedisModule_ReplyWithError(ctx,
                    "ERR invalid number of cycles");
            }
            nr->training_max_cycles = v;
        } else if (!strcasecmp(o,"maxtime") && !lastarg) {
            if (RedisModule_StringToLongLong(argv[++j],&v) != REDISMODULE_OK) {
                return RedisModule_ReplyWithError(ctx,
                    "ERR invalid number of milliseconds of time");
            }
            nr->training_max_ms = v;
        } else {
            return RedisModule_ReplyWithError(ctx,
                "ERR Syntax error in NR.TRAIN");
        }
    }

    if (NRStartTraining(ctx,argv[1],RedisModule_GetSelectedDb(ctx),nr) ==
        REDISMODULE_ERR)
    {
        return RedisModule_ReplyWithError(ctx,
            "ERR Can't train the neural network: "
            "too many NNs already training");
    } else {
        return RedisModule_ReplyWithSimpleString(ctx,"Training has started");
    }
}

/* NR.INFO key */
int NRInfo_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    RedisModule_AutoMemory(ctx); /* Use automatic memory management. */
    NRCollectThreads(ctx);

    if (argc != 2) return RedisModule_WrongArity(ctx);
    RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1], REDISMODULE_READ);
    int type = RedisModule_KeyType(key);
    if (RedisModule_ModuleTypeGetType(key) != NRType)
        return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);

    NRTypeObject *nr = RedisModule_ModuleTypeGetValue(key);

    RedisModule_ReplyWithArray(ctx,2*12);

    RedisModule_ReplyWithSimpleString(ctx,"id");
    RedisModule_ReplyWithLongLong(ctx,nr->id);

    RedisModule_ReplyWithSimpleString(ctx,"training");
    RedisModule_ReplyWithLongLong(ctx,nr->flags & NR_FLAG_TRAINING);

    RedisModule_ReplyWithSimpleString(ctx,"layout");
    RedisModule_ReplyWithArray(ctx,LAYERS(nr->nn));
    for (int i = LAYERS(nr->nn)-1; i >= 0; i--) {
        int units = UNITS(nr->nn,i);
        if (i != 0) units--; /* Don't count the bias unit. */
        RedisModule_ReplyWithLongLong(ctx,units);
    }

    RedisModule_ReplyWithSimpleString(ctx,"training-dataset-maxlen");
    RedisModule_ReplyWithLongLong(ctx,nr->dataset.maxlen);

    RedisModule_ReplyWithSimpleString(ctx,"training-dataset-len");
    RedisModule_ReplyWithLongLong(ctx,nr->dataset.len);

    RedisModule_ReplyWithSimpleString(ctx,"test-dataset-maxlen");
    RedisModule_ReplyWithLongLong(ctx,nr->test.maxlen);

    RedisModule_ReplyWithSimpleString(ctx,"test-dataset-len");
    RedisModule_ReplyWithLongLong(ctx,nr->test.len);

    RedisModule_ReplyWithSimpleString(ctx,"training-total-steps");
    RedisModule_ReplyWithLongLong(ctx,nr->training_total_steps);

    RedisModule_ReplyWithSimpleString(ctx,"training-total-ms");
    RedisModule_ReplyWithLongLong(ctx,nr->training_total_ms);

    RedisModule_ReplyWithSimpleString(ctx,"dataset-error");
    RedisModule_ReplyWithDouble(ctx,nr->dataset_error);

    RedisModule_ReplyWithSimpleString(ctx,"test-error");
    RedisModule_ReplyWithDouble(ctx,nr->test_error);

    RedisModule_ReplyWithSimpleString(ctx,"overfitting-detected");
    RedisModule_ReplyWithSimpleString(ctx, (nr->flags & NR_FLAG_OF_DETECTED) ? "yes" : "no");

    return REDISMODULE_OK;
}

/* =============================== Type methods ============================= */

void *NRTypeRdbLoad(RedisModuleIO *rdb, int encver) {
#if 0
    if (encver != 0) {
        /* RedisModule_Log("warning","Can't load data with version %d", encver);*/
        return NULL;
    }
    uint64_t elements = RedisModule_LoadUnsigned(rdb);
    NRTypeObject *hto = createNRTypeObject();
    while(elements--) {
        int64_t ele = RedisModule_LoadSigned(rdb);
        NRTypeInsert(hto,ele);
    }
    return hto;
#endif
}

void NRTypeRdbSave(RedisModuleIO *rdb, void *value) {
#if 0
    NRTypeObject *hto = value;
    struct NRTypeNode *node = hto->head;
    RedisModule_SaveUnsigned(rdb,hto->len);
    while(node) {
        RedisModule_SaveSigned(rdb,node->value);
        node = node->next;
    }
#endif
}

void NRTypeAofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
#if 0
    NRTypeObject *hto = value;
    struct NRTypeNode *node = hto->head;
    while(node) {
        RedisModule_EmitAOF(aof,"HELLOTYPE.INSERT","sl",key,node->value);
        node = node->next;
    }
#endif
}

void NRTypeDigest(RedisModuleDigest *digest, void *value) {
    /* TODO: The DIGEST module interface is yet not implemented. */
}

void NRTypeFree(void *value) {
    NRTypeReleaseObject(value);
}

/* This function must be present on each Redis module. It is used in order to
 * register the commands into the Redis server. */
int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_Init(ctx,"neuralredis",1,REDISMODULE_APIVER_1)
        == REDISMODULE_ERR) return REDISMODULE_ERR;

    NRType = RedisModule_CreateDataType(ctx,"neural-NN",0,NRTypeRdbLoad,NRTypeRdbSave,NRTypeAofRewrite,NRTypeDigest,NRTypeFree);
    if (NRType == NULL) return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx,"nr.create",
        NRCreate_RedisCommand,"write deny-oom",1,1,1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx,"nr.run",
        NRRun_RedisCommand,"readonly",1,1,1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx,"nr.observe",
        NRObserve_RedisCommand,"write deny-oom",1,1,1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx,"nr.info",
        NRInfo_RedisCommand,"readonly",1,1,1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx,"nr.train",
        NRTrain_RedisCommand,"write",1,1,1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    return REDISMODULE_OK;
}
