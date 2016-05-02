package features

import (
	"github.com/eriq-augustine/goml/base"
)

// Reduces a set of features usually based on some statistical analysis.
type Reducer interface {
	Init(Features, []base.Tuple)
	Reduce(Features) Features
}
