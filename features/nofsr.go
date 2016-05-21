package features

import (
   "github.com/eriq-augustine/goml/base"
)

type NoReducer struct{}

func (reducer NoReducer) Init(tuples []base.Tuple) {}

func (reducer NoReducer) Reduce(tuple base.Tuple) base.Tuple {
   return tuple;
}
