package optimize

import (
   "math"

   "github.com/eriq-augustine/goml/util"
)

const (
   OPTIMIZER_SGD_DEFAULT_MAX_ITERATIONS = 2e3
   OPTIMIZER_SGD_DEFAULT_ALPHA = 1e-4
   OPTIMIZER_SGD_DEFAULT_TOLERENCE = 1e-4
   OPTIMIZER_SGD_DEFAULT_BATCH_SIZE = 128
)

type SGD struct {
   maxIterations int
   alpha float64
   tolerence float64
   batchSize int
}

func NewSGD(maxIterations int, alpha float64, tolerence float64, batchSize int) *SGD {
   if (maxIterations <= 0) {
      maxIterations = OPTIMIZER_SGD_DEFAULT_MAX_ITERATIONS;
   }

   if (alpha <= 0) {
      alpha = OPTIMIZER_SGD_DEFAULT_ALPHA;
   }

   if (tolerence <= 0) {
      tolerence = OPTIMIZER_SGD_DEFAULT_TOLERENCE;
   }

   if (batchSize <= 0) {
      batchSize = OPTIMIZER_SGD_DEFAULT_BATCH_SIZE;
   }

   var sgd SGD = SGD{
      maxIterations: maxIterations,
      alpha: alpha,
      tolerence: tolerence,
      batchSize: batchSize,
   };

   return &sgd;
}

func (this SGD) Optimize(initialParams []float64, objective ObjectiveFunction, gradient ObjectiveFunctionGradient) []float64 {
   panic("StochasticGradientDescent does not support non batches.");
}

func (this SGD) OptimizeBatch(initialParams []float64, initialPoints []int, objective ObjectiveFunction, gradient ObjectiveBatchFunctionGradient) []float64 {
   // Make a copy so we don't trash the caller's.
   var params []float64 = append([]float64{}, initialParams...);
   var points []int = append([]int{}, initialPoints...);

   var firstRun bool = true;
   // Evaluation of the objective function.
   var value float64;

   for iteration := 0; iteration < this.maxIterations; iteration++ {
      util.ShuffleIntSlice(points);

      for batch := 0; batch < int(math.Ceil(float64(len(points)) / float64(this.batchSize))); batch++ {
         var batchStart int = batch * this.batchSize;
         var batchEnd int = util.MinInt(len(points), ((batch + 1) * this.batchSize));
         var batchPoints []int = points[batchStart:batchEnd];

         // Modify weights
         var gradients []float64 = gradient(params, batchPoints);
         for paramIndex, _ := range(params) {
            params[paramIndex] -= this.alpha * gradients[paramIndex];
         }
      }

      var nextValue float64 = objective(params);
      if (!firstRun && math.Abs(nextValue - value) < this.tolerence) {
         break;
      }

      firstRun = false;
      value = nextValue;
   }

   return params;
}

func (this SGD) SupportsBatch() bool {
   return true;
}
