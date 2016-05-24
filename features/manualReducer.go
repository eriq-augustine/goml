package features

import (
   "github.com/eriq-augustine/goml/base"
)

// A ManualReducer is initialized with the columns to choose.
// This should mainly be used for testing.
type ManualReducer struct{
   features []int
}

func NewManualReducer(selectedFeatures []int) ManualReducer {
   if (selectedFeatures == nil) {
      selectedFeatures = make([]int, 0);
   }

   return ManualReducer{selectedFeatures};
}

func (this ManualReducer) Init(tuples []base.Tuple) {}

func (this ManualReducer) Reduce(tuples []base.Tuple) []base.Tuple {
   return SelectFeatures(tuples, this.features);
}
