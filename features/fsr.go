package features

import (
   "github.com/eriq-augustine/goml/base"
)

// Reduces a set of features usually based on some statistical analysis.
type Reducer interface {
   Init([]base.Tuple)
   Reduce([]base.Tuple) []base.Tuple
}
