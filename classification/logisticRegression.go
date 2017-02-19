package classification

import (
   "fmt"
   "math"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
   "github.com/eriq-augustine/goml/optimize"
   "github.com/eriq-augustine/goml/util"
)

const (
   LR_DEFAULT_L2_PENALTY = 1.0
)

type LogisticRegression struct {
   reducer features.Reducer
   optimizer optimize.Optimizer
   l2Penalty float64

   // [class][feature]
   weights [][]float64
   intercepts []float64
   labels []base.Feature
}

// Note that 0 is a valid value for |l2Penalty|, pass -1 for default.
func NewLogisticRegression(reducer features.Reducer, optimizer optimize.Optimizer, l2Penalty float64) *LogisticRegression {
   if (reducer == nil) {
      reducer = features.NoReducer{};
   }

   if (optimizer == nil) {
      optimizer = optimize.NewSGD(0, 0, 0, 0);
   }

   if (l2Penalty < 0) {
      l2Penalty = LR_DEFAULT_L2_PENALTY;
   }

   var lr LogisticRegression = LogisticRegression{
      reducer: reducer,
      l2Penalty: l2Penalty,
      optimizer: optimizer,
   };

   return &lr;
}

func (this *LogisticRegression) Train(tuples []base.Tuple) {
   if (tuples == nil || len(tuples) == 0) {
      panic("Must provide tuples for training.")
   }

   this.reducer.Init(tuples);
   tuples = this.reducer.Reduce(tuples);

   var numFeatures int = -1;
   var numericTuples []base.NumericTuple = make([]base.NumericTuple, len(tuples));
   for i, tuple := range(tuples) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("LogisticRegression only supports classifying NumericTuple");
      }
      numericTuples[i] = numericTuple;

      if (numFeatures == -1) {
         numFeatures = numericTuple.DataSize();
      } else if (numFeatures != numericTuple.DataSize()) {
         panic(fmt.Sprintf("Inconsistent number of features. Tuple[0]: %d, Tuple[%d]: %d",
               numFeatures, i, numericTuple.DataSize()));
      }
   }

   this.train(numericTuples);
}

func (this *LogisticRegression) train(tuples []base.NumericTuple) {
   this.labels = make([]base.Feature, 0);
   var labelMap map[base.Feature]int = make(map[base.Feature]int);
   for _, tuple := range(tuples) {
      _, ok := labelMap[tuple.GetClass()];
      if (!ok) {
         labelMap[tuple.GetClass()] = len(this.labels);
         this.labels = append(this.labels, tuple.GetClass());
      }
   }

   // Weights (|labels| x |features|) + Intercepts (|labels|)
   var initialParams []float64 = make([]float64, len(this.labels) * (1 + tuples[0].DataSize()));

   if (this.optimizer.SupportsBatch()) {
      this.weights, this.intercepts = this.unpackOptimizerParams(this.optimizer.OptimizeBatch(
         initialParams,
         util.RangeSlice(len(tuples)),
         func(params []float64) float64 { return this.negativeLogLikelihoodOptimize(tuples, params); },
         func(params []float64, points []int) []float64 { return this.negativeLogLikelihoodGradientBatchOptimize(tuples, params, points); },
      ));
   } else {
      this.weights, this.intercepts = this.unpackOptimizerParams(this.optimizer.Optimize(
         initialParams,
         func(params []float64) float64 { return this.negativeLogLikelihoodOptimize(tuples, params); },
         func(params []float64) []float64 { return this.negativeLogLikelihoodGradientOptimize(tuples, params); },
      ));
   }
}

func (this LogisticRegression) Classify(tuples []base.Tuple) ([]base.Feature, []float64) {
   tuples = this.reducer.Reduce(tuples);

   var numericTuples []base.NumericTuple = make([]base.NumericTuple, len(tuples));
   for i, tuple := range(tuples) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("LogisticRegression only supports classifying NumericTuple");
      }
      numericTuples[i] = numericTuple;
   }

   return this.classify(numericTuples);
}

func (this LogisticRegression) classify(tuples []base.NumericTuple) ([]base.Feature, []float64) {
   var probabilities [][]float64 = probabilities(this.weights, this.intercepts, tuples);

   var results []base.Feature = make([]base.Feature, len(tuples));
   var resultProbabilities []float64 = make([]float64, len(tuples));

   for i, instanceProbabilities := range(probabilities) {
      maxProbabilityIndex, maxProbability := util.Max(instanceProbabilities);
      results[i] = this.labels[maxProbabilityIndex];
      resultProbabilities[i] = maxProbability;
   }

   return results, resultProbabilities;
}

// Calculate the probability of each tuple being in each class.
// Returns float64[tuple][class]
// The math comes out to prob(tuple=x,class=k) = exp(Wk dot x - logSumExp(Wj dot x))
// (logSumExp is over all classes j).
func probabilities(weights [][]float64, intercepts[]float64, tuples []base.NumericTuple) [][]float64 {
   var probabilities [][]float64 = make([][]float64, len(tuples));
   for i, _ := range(probabilities) {
      probabilities[i] = make([]float64, len(weights));
   }

   // This will get reset each tuple, but only allocate once.
   var activations []float64 = make([]float64, len(weights));

   for tupleIndex, tuple := range(tuples) {
      for classIndex, _ := range(weights) {
         activations[classIndex] = intercepts[classIndex] + dot(weights[classIndex], tuple);
      }

      var normalization float64 = util.LogSumExp(activations);

      for classIndex, _ := range(weights) {
         probabilities[tupleIndex][classIndex] = math.Exp(activations[classIndex] - normalization);
      }
   }

   return probabilities;
}

// Math comes out to:
// NNL = -[ sum(n over data){ sum(k over classes){ oneHotLabel(n, k) * (Wk dot x - logSumExp(Wj dot x)) } } ]
// NNL = -[ sum(n over data){ sum(k over classes){ oneHotLabel(n, k) * log(prob(Xn, k)) } } ]
func negativeLogLikelihood(
      weights [][]float64, intercepts []float64, l2Penalty float64,
      tuples []base.NumericTuple, labels []base.Feature) float64 {
   var probabilities [][]float64 = probabilities(weights, intercepts, tuples);

   var sum float64 = 0;
   for tupleIndex, tuple := range(tuples) {
      for classIndex, classValue := range(labels) {
         // One hot multiplication, the value is only active if the class is one that we are examining.
         if (tuple.GetClass() == classValue) {
            sum += math.Log(probabilities[tupleIndex][classIndex]);
         }
      }
   }

   // Add an l2 regularizer
   var regularizer float64 = 0;
   for classIndex, _ := range(weights) {
      regularizer += math.Pow(intercepts[classIndex], 2);

      for _, weight := range(weights[classIndex]) {
         regularizer += math.Pow(weight, 2);
      }
   }
   regularizer = l2Penalty / 2.0 * regularizer;

   return -1.0 * sum + regularizer;
}

// Note that the gradient is with respects to each specific weight.
// So, we will return a vector of gradients.
func negativeLogLikelihoodGradient(
      weights [][]float64, intercepts []float64, l2Penalty float64,
      tuples []base.NumericTuple, labels []base.Feature) ([][]float64, []float64) {
   var probabilities [][]float64 = probabilities(weights, intercepts, tuples);

   var gradients [][]float64 = make([][]float64, len(labels));
   for classIndex, _ := range(gradients) {
      gradients[classIndex] = make([]float64, tuples[0].DataSize());
   }

   var interceptGradients []float64 = make([]float64, len(labels));

   for tupleIndex, tuple := range(tuples) {
      for classIndex, classValue := range(labels) {
         var val float64 = probabilities[tupleIndex][classIndex];
         // One hot.
         if (tuple.GetClass() == classValue) {
            val -= 1.0;
         }

         interceptGradients[classIndex] += val;

         for featureIndex := 0; featureIndex < tuple.DataSize(); featureIndex++ {
            gradients[classIndex][featureIndex] += val * tuple.GetNumericData(featureIndex);
         }
      }
   }

   // Add an l2 regularizer.
   for classIndex, _ := range(weights) {
      interceptGradients[classIndex] += intercepts[classIndex] * l2Penalty;
      for featureIndex, _ := range(weights[classIndex]) {
         gradients[classIndex][featureIndex] += weights[classIndex][featureIndex] * l2Penalty;
      }
   }

   return gradients, interceptGradients;
}

// Unpack params from the optimizer into weights and intercepts.
// Params are packed: [
//    intercept[0], intercept[1], ... , intercept[K - 1],
//    weight[0][0], weight[0][1], ..., weight[0][N - 1],
//    ...
//    weight[K - 1][0], weight[K - 1][1], ..., weight[K - 1][M - 1]
// ]
// K - Number of Labels
// N - Number of Features
func (this LogisticRegression) unpackOptimizerParams(params []float64) ([][]float64, []float64) {
   var intercepts []float64 = params[:len(this.labels)];

   var packedWeights []float64 = params[len(this.labels):];
   var numFeatures int = len(packedWeights) / len(this.labels);

   var weights [][]float64 = make([][]float64, len(this.labels));
   for i, _ := range(weights) {
      weights[i] = packedWeights[(i * numFeatures) : ((i + 1) * numFeatures)];
   }

   return weights, intercepts;
}

// A wrapper for an optimizer function for NNL.
// The first param will be curried.
func (this LogisticRegression) negativeLogLikelihoodOptimize(
      tuples []base.NumericTuple,
      params []float64) float64 {
   weights, intercepts := this.unpackOptimizerParams(params);

   return negativeLogLikelihood(weights, intercepts, this.l2Penalty, tuples, this.labels);
}

// A wrapper for an optimizer function for NLL.
// The first param will be curried.
func (this LogisticRegression) negativeLogLikelihoodGradientOptimize(
      tuples []base.NumericTuple,
      params []float64) []float64 {
   weights, intercepts := this.unpackOptimizerParams(params);

   weightGradients, interceptGradients := negativeLogLikelihoodGradient(
         weights, intercepts, this.l2Penalty,
         tuples, this.labels);

   // Packup the gradients.
   var gradients []float64 = make([]float64, len(interceptGradients) + len(weightGradients) * len(weightGradients[0]));
   for i, interceptGradient := range(interceptGradients) {
      gradients[i] = interceptGradient;
   }

   for i, _ := range(weightGradients) {
      for j, weightGradient := range(weightGradients[i]) {
         gradients[len(interceptGradients) + (i * len(weightGradients[i]) + j)] = weightGradient;
      }
   }

   return gradients;
}

// A wrapper for a batch optimizer function for NLL.
// The first param will be curried.
func (this LogisticRegression) negativeLogLikelihoodGradientBatchOptimize(
      tuples []base.NumericTuple,
      params []float64,
      points []int) []float64 {
   return this.negativeLogLikelihoodGradientOptimize(base.SelectNumericTuples(tuples, points), params);
}

// TODO(eriq): Optimize
func dot(a []float64, b base.NumericTuple) float64 {
   if (len(a) != b.DataSize()) {
      panic(fmt.Sprintf("Length of LHS (%d) and length of RHS (%d) must match for a dot.", len(a), b.DataSize()));
   }

   var sum float64 = 0;
   for i, _ := range(a) {
      sum += a[i] * b.GetNumericData(i);
   }

   return sum;
}
