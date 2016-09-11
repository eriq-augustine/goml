package classification

import (
   "github.com/eriq-augustine/goml/base"
)

type Classifier interface {
   Train([]base.Tuple)
   // The second return is a confidence value for each classification.
   // If the Classifier does not support it, nil will always be returned.
   // There are no bounds on this value and they may not be comparable
   // between different classifiers or even instances of the same classifier.
   Classify([]base.Tuple) ([]base.Feature, []float64)
}
