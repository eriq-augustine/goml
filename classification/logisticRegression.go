package classification

import (
   "fmt"
   "math"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
   "github.com/eriq-augustine/goml/optimize"
   "github.com/eriq-augustine/goml/util"

   "github.com/gonum/blas/blas64"
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
   var numericData [][]float64 = make([][]float64, len(tuples));
   var dataLabels []int = make([]int, len(tuples));

   // Note all the labels we have seen and assign them an arbitrary identifier (index into this.labels).
   this.labels = make([]base.Feature, 0);
   var labelMap map[base.Feature]int = make(map[base.Feature]int);

   for i, tuple := range(tuples) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("LogisticRegression only supports classifying NumericTuple");
      }

      _, ok = labelMap[numericTuple.GetClass()];
      if (!ok) {
         labelMap[numericTuple.GetClass()] = len(this.labels);
         this.labels = append(this.labels, numericTuple.GetClass());
      }

      numericData[i] = numericTuple.ToFloatSlice();
      // Convert all labels to their surrogate identifier.
      dataLabels[i] = labelMap[numericTuple.GetClass()];

      // Ensure all vectors are the same size.
      if (numFeatures == -1) {
         numFeatures = numericTuple.DataSize();
      } else if (numFeatures != numericTuple.DataSize()) {
         panic(fmt.Sprintf("Inconsistent number of features. Tuple[0]: %d, Tuple[%d]: %d",
               numFeatures, i, numericTuple.DataSize()));
      }
   }

   this.train(numericData, dataLabels);
}

func (this LogisticRegression) Classify(tuples []base.Tuple) ([]base.Feature, []float64) {
   tuples = this.reducer.Reduce(tuples);

   var numericData [][]float64 = make([][]float64, len(tuples));
   for i, tuple := range(tuples) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("LogisticRegression only supports classifying NumericTuple");
      }
      numericData[i] = numericTuple.ToFloatSlice();
   }

   classIndexes, probabilities := this.classify(numericData);

   // Translate the class ids back to actual features.
   var classes []base.Feature = make([]base.Feature, len(classIndexes));
   for i, classIndex := range(classIndexes) {
      classes[i] = this.labels[classIndex];
   }

   return classes, probabilities;
}

// In the internals of Logistic Regression (typically non-exported functions),
// we don't deal with actual base.Tuple's.
// Just raw slices of doubles and ints (which are the mapped class labels).

func (this *LogisticRegression) train(data [][]float64, dataLabels []int) {
   // Params = Weights                 + Intercepts
   //          (|labels| x |features|) + (|labels|)
   var initialParams []float64 = make([]float64, len(this.labels) * (1 + len(data[0])));

   if (this.optimizer.SupportsBatch()) {
      this.weights, this.intercepts = this.unpackOptimizerParams(this.optimizer.OptimizeBatch(
         initialParams,
         util.RangeSlice(len(data)),
         func(params []float64) float64 {
            return this.negativeLogLikelihoodOptimize(data, dataLabels, params);
         },
         func(params []float64, points []int) []float64 {
            return this.negativeLogLikelihoodGradientBatchOptimize(data, dataLabels, params, points);
         },
      ));
   } else {
      this.weights, this.intercepts = this.unpackOptimizerParams(this.optimizer.Optimize(
         initialParams,
         func(params []float64) float64 {
            return this.negativeLogLikelihoodOptimize(data, dataLabels, params);
         },
         func(params []float64) []float64 {
            return this.negativeLogLikelihoodGradientOptimize(data, dataLabels, params);
         },
      ));
   }
}

func (this LogisticRegression) classify(data [][]float64) ([]int, []float64) {
   var probabilities [][]float64 = probabilities(this.weights, this.intercepts, data);

   var results []int = make([]int, len(data));
   var resultProbabilities []float64 = make([]float64, len(data));

   for i, instanceProbabilities := range(probabilities) {
      maxProbabilityIndex, maxProbability := util.Max(instanceProbabilities);
      results[i] = maxProbabilityIndex;
      resultProbabilities[i] = maxProbability;
   }

   return results, resultProbabilities;
}

// Calculate the probability of each data point being in each class.
// Returns float64[data point][class]
// The math comes out to prob(point=x,class=k) = exp(Wk dot x - logSumExp(Wj dot x))
// (logSumExp is over all classes j).
func probabilities(weights [][]float64, intercepts[]float64, data [][]float64) [][]float64 {
   var probabilities [][]float64 = make([][]float64, len(data));
   for i, _ := range(probabilities) {
      probabilities[i] = make([]float64, len(weights));
   }

   // This will get reset each data point, but only allocated once.
   var activations []float64 = make([]float64, len(weights));

   for dataPointIndex, dataPoint := range(data) {
      for classIndex, _ := range(weights) {
         activations[classIndex] = intercepts[classIndex] + dot(weights[classIndex], dataPoint);
      }

      var normalization float64 = util.LogSumExp(activations);
      for classIndex, _ := range(weights) {
         probabilities[dataPointIndex][classIndex] = math.Exp(activations[classIndex] - normalization);
      }
   }

   return probabilities;
}

// Math comes out to:
// NLL = -[ sum(n over data){ sum(k over classes){ oneHotLabel(n, k) * (Wk dot x - logSumExp(Wj dot x)) } } ]
// NLL = -[ sum(n over data){ sum(k over classes){ oneHotLabel(n, k) * log(prob(Xn, k)) } } ]
func negativeLogLikelihood(
      weights [][]float64, intercepts []float64, l2Penalty float64,
      data [][]float64, dataLabels []int) float64 {
   var probabilities [][]float64 = probabilities(weights, intercepts, data);

   var sum float64 = 0;
   for dataPointIndex, _ := range(data) {
      // One hot multiplication, the value is only active if the class is one that we are examining.
      sum += math.Log(probabilities[dataPointIndex][dataLabels[dataPointIndex]]);
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
      data [][]float64, dataLabels []int) ([][]float64, []float64) {
   var probabilities [][]float64 = probabilities(weights, intercepts, data);

   // TODO(eriq): Allocate once and keep in struct?
   // Note that the number of weight vectors (len(weights)) ==
   // the number of intercepts (len(intercepts)) ==
   // the number of classes.
   // [class][feature]
   var gradients [][]float64 = make([][]float64, len(intercepts));
   for classIndex, _ := range(gradients) {
      gradients[classIndex] = make([]float64, len(data[0]));
   }

   var interceptGradients []float64 = make([]float64, len(intercepts));

   for dataPointIndex, dataPoint := range(data) {
      for classIndex, _ := range(intercepts) {
         var val float64 = probabilities[dataPointIndex][classIndex];

         // One hot.
         if (dataLabels[dataPointIndex] == classIndex) {
            val -= 1.0;
         }

         interceptGradients[classIndex] += val;

         for featureIndex := 0; featureIndex < len(data[0]); featureIndex++ {
            gradients[classIndex][featureIndex] += val * dataPoint[featureIndex];
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

   // TODO(eriq): Avoid this allocation?
   var weights [][]float64 = make([][]float64, len(this.labels));
   for i, _ := range(weights) {
      weights[i] = packedWeights[(i * numFeatures) : ((i + 1) * numFeatures)];
   }

   return weights, intercepts;
}

// A wrapper for an optimizer function for NLL.
// The first two params will be curried.
func (this LogisticRegression) negativeLogLikelihoodOptimize(
      data [][]float64,
      dataLabels []int,
      params []float64) float64 {
   weights, intercepts := this.unpackOptimizerParams(params);

   return negativeLogLikelihood(weights, intercepts, this.l2Penalty, data, dataLabels);
}

// A wrapper for an optimizer function for NLL.
// The first two params will be curried.
func (this LogisticRegression) negativeLogLikelihoodGradientOptimize(
      data [][]float64,
      dataLabels []int,
      params []float64) []float64 {
   weights, intercepts := this.unpackOptimizerParams(params);

   weightGradients, interceptGradients := negativeLogLikelihoodGradient(
         weights, intercepts, this.l2Penalty,
         data, dataLabels);

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
      data [][]float64,
      dataLabels []int,
      params []float64,
      points []int) []float64 {
   return this.negativeLogLikelihoodGradientOptimize(
      util.SelectIndexesFloat2D(data, points), util.SelectIndexesInt(dataLabels, points), params);
}

func dot(a []float64, b []float64) float64 {
   if (len(a) != len(b)) {
      panic(fmt.Sprintf("Length of LHS (%d) and length of RHS (%d) must match for a dot.", len(a), len(b)));
   }

   var aVec blas64.Vector = blas64.Vector{1, a};
   var bVec blas64.Vector = blas64.Vector{1, b};

   return blas64.Dot(len(a), aVec, bVec);
}
