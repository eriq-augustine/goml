package classification

import (
	"testing"

	"github.com/eriq-augustine/goml/base"
)

func TestBase(t *testing.T) {
	var knn Classifier = NewKnn(3, nil)

	var trainingData []base.Tuple = []base.Tuple{
		base.Tuple{[]interface{}{10, 10}, "A"},
		base.Tuple{[]interface{}{9, 9}, "A"},
		base.Tuple{[]interface{}{11, 11}, "A"},
		base.Tuple{[]interface{}{-10, -10}, "B"},
		base.Tuple{[]interface{}{-9, -9}, "B"},
		base.Tuple{[]interface{}{-11, -11}, "B"},
	}

	knn.Train(trainingData)

	var testData []base.Tuple = []base.Tuple{
		base.Tuple{[]interface{}{8, 8}, "A"},
		base.Tuple{[]interface{}{-8, -8}, "B"},
	}

	for _, tuple := range testData {
		var actual interface{} = knn.Classify(tuple)
		if actual != tuple.Class {
			t.Errorf("Bad classification. Expected: %v, Got: %v", tuple.Class, actual)
		}
	}
}
