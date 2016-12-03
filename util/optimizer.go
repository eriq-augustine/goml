package util

import (
   "math"
)

const (
   OPTIMIZER_GD_DEFAULT_MAX_ITERATIONS = 1e3
   OPTIMIZER_GD_DEFAULT_ALPHA = 1e-4
   OPTIMIZER_GD_DEFAULT_TOLERENCE = 1e-4
)

type OptimizeFunction func([]float64) float64;
type OptimizeFunctionGradient func([]float64) []float64;

type Optimizer interface {
   Optimize(params []float64, function OptimizeFunction, gradient OptimizeFunctionGradient) []float64
}

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

func (this GradientDescent) Optimize(initialParams []float64, function OptimizeFunction, gradient OptimizeFunctionGradient) []float64 {
   // Make a copy so we don't trash the caller's.
   var params []float64 = append([]float64{}, initialParams...);

	// First run
   var value float64 = function(params);
   for iteration := 0; iteration < this.maxIterations; iteration++ {
      // Modify weights
      var gradients []float64 = gradient(params);
      for paramIndex, _ := range(params) {
         params[paramIndex] -= this.alpha * gradients[paramIndex];
      }

      var nextValue float64 = function(params);
      if (math.Abs(nextValue - value) < this.tolerence) {
         break;
      }

      value = nextValue;
   }

   return params;
}
