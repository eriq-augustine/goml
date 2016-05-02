package classification

import (
	"github.com/eriq-augustine/goml/base"
)

type Classifier interface {
	Train([]base.Tuple)
	Classify(base.Tuple) interface{}
}
