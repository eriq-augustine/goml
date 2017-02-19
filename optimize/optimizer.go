package optimize

// The objective function to minimize.
type ObjectiveFunction func(params []float64) float64;
// Returns the gradient for each parameter.
type ObjectiveFunctionGradient func(params []float64) []float64;

// A function that will evaluate the objective for all indexes indicated in |points|.
type ObjectiveBatchFunctionGradient func(params []float64, points []int) []float64;

type Optimizer interface {
   // Not all optimizers must support both batch and non-batch mode, but it better support at least one.
   Optimize(params []float64, objective ObjectiveFunction, gradient ObjectiveFunctionGradient) []float64
   // |points| should be a int identifier (probably index) for each point in the dataset.
   // These will be chunked into batches.
   OptimizeBatch(params []float64, points []int, objective ObjectiveFunction, gradient ObjectiveBatchFunctionGradient) []float64

   SupportsBatch() bool
}
