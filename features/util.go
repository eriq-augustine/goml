package features

import (
	"fmt"

	"github.com/eriq-augustine/goml/base"
)

func InferFeatureType(value interface{}) FeatureType {
	if value == nil {
		panic("Cannot infer feature type of nil tuple value")
	}

	switch valueType := value.(type) {
	case int, int32, int64, uint, uint32, uint64:
		return Numeric
	case bool:
		return Boolean
	case float32, float64:
		return Numeric
	case string:
		return String
	default:
		panic(fmt.Sprintf("Unknown type for feature inference: %T", valueType))
	}
}

// Returns (data feature types, class label feature type)
func InferFeatures(tuple base.Tuple) (Features, FeatureType) {
	var features Features = make([]FeatureType, len(tuple.Data))
	var classLabelType FeatureType = String

	for i, value := range tuple.Data {
		features[i] = InferFeatureType(value)
	}

	if tuple.Class != nil {
		classLabelType = InferFeatureType(tuple.Class)
	}

	return features, classLabelType
}
