package features

import (
   "github.com/eriq-augustine/goml/base"
)

type NoReducer struct{}

func (reducer NoReducer) Init(tuples []base.Tuple) {}

func (reducer NoReducer) Reduce(tuples []base.Tuple) []base.Tuple {
   return tuples;
}
