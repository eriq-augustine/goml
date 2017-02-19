package optimize

import (
   "math"
)

const (
   OPTIMIZER_GD_DEFAULT_MAX_ITERATIONS = 1e3
   OPTIMIZER_GD_DEFAULT_ALPHA = 1e-4
   OPTIMIZER_GD_DEFAULT_TOLERENCE = 1e-4
)

type GradientDescent struct {
   maxIterations int
   alpha float64
   tolerence float64
}

func NewGradientDescent(maxIterations int, alpha float64, tolerence float64) *GradientDescent {
   if (maxIterations <= 0) {
      maxIterations = OPTIMIZER_GD_DEFAULT_MAX_ITERATIONS;
   }

   if (alpha <= 0) {
      alpha = OPTIMIZER_GD_DEFAULT_ALPHA;
   }

   if (tolerence <= 0) {
      tolerence = OPTIMIZER_GD_DEFAULT_TOLERENCE;
   }

   var gd GradientDescent = GradientDescent{
      maxIterations: maxIterations,
      alpha: alpha,
      tolerence: tolerence,
   };

   return &gd;
}

func (this GradientDescent) Optimize(initialParams []float64, objective ObjectiveFunction, gradient ObjectiveFunctionGradient) []float64 {
   // Make a copy so we don't trash the caller's.
   var params []float64 = append([]float64{}, initialParams...);

	// First run
   var value float64 = objective(params);
   for iteration := 0; iteration < this.maxIterations; iteration++ {
      // Modify weights
      var gradients []float64 = gradient(params);
      for paramIndex, _ := range(params) {
         params[paramIndex] -= this.alpha * gradients[paramIndex];
      }

      var nextValue float64 = objective(params);
      if (math.Abs(nextValue - value) < this.tolerence) {
         break;
      }

      value = nextValue;
   }

   return params;
}

func (this GradientDescent) OptimizeBatch(initialParams []float64, points []int, objective ObjectiveFunction, gradient ObjectiveBatchFunctionGradient) []float64 {
   panic("GradientDescent does not support batches.");
}

func (this GradientDescent) SupportsBatch() bool {
   return false;
}
