package features

import (
	"github.com/eriq-augustine/goml/base"
)

type NoReducer struct{}

func (reducer NoReducer) Init(features Features, tuples []base.Tuple) {}

func (reducer NoReducer) Reduce(features Features) Features {
	return features
}
